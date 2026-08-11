"""
Receiver / display client for the orbital simulation TCP streaming system.

Connects to the video and telemetry TCP servers, decodes incoming frames,
and (optionally) displays them in an OpenCV window with telemetry overlay.
"""

import asyncio
import io
import json
import logging

import numpy as np
from PIL import Image

from .protocol import (
    HEADER_SIZE,
    decode_header,
)

logger = logging.getLogger(__name__)


class StreamingClient:
    """Connects to an orbital-simulation streaming server and receives
    video frames and telemetry data."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, host: str = "localhost", video_port: int = 9100,
                 telemetry_port: int = 9101) -> None:
        self._host = host
        self._video_port = video_port
        self._telemetry_port = telemetry_port

        self._video_reader: asyncio.StreamReader | None = None
        self._video_writer: asyncio.StreamWriter | None = None
        self._telemetry_reader: asyncio.StreamReader | None = None
        self._telemetry_writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        """Open TCP connections to both the video and telemetry servers."""
        try:
            self._video_reader, self._video_writer = await asyncio.open_connection(
                self._host, self._video_port
            )
            logger.info("Connected to video server at %s:%d",
                        self._host, self._video_port)
        except OSError as exc:
            logger.error("Failed to connect to video server: %s", exc)
            raise

        try:
            self._telemetry_reader, self._telemetry_writer = await asyncio.open_connection(
                self._host, self._telemetry_port
            )
            logger.info("Connected to telemetry server at %s:%d",
                        self._host, self._telemetry_port)
        except OSError as exc:
            logger.error("Failed to connect to telemetry server: %s", exc)
            # Clean up the video connection that already succeeded.
            if self._video_writer is not None:
                self._video_writer.close()
            raise

    async def disconnect(self) -> None:
        """Close both connections gracefully."""
        for writer in (self._video_writer, self._telemetry_writer):
            if writer is not None:
                try:
                    writer.close()
                    await writer.wait_closed()
                except OSError:
                    pass
        self._video_reader = None
        self._video_writer = None
        self._telemetry_reader = None
        self._telemetry_writer = None
        logger.info("Disconnected from streaming server")

    # ------------------------------------------------------------------
    # Low-level frame reading
    # ------------------------------------------------------------------

    async def _read_frame(self, reader: asyncio.StreamReader) -> dict | None:
        """Read one complete protocol frame from *reader*.

        Returns a dict with keys ``channel``, ``data`` (raw payload bytes),
        ``seq``, and ``sim_time``, or ``None`` on EOF / protocol error.
        """
        try:
            header_bytes = await reader.readexactly(HEADER_SIZE)
        except (asyncio.IncompleteReadError, ConnectionError):
            return None

        header = decode_header(header_bytes)
        if header is None:
            logger.warning("Received frame with invalid magic bytes")
            return None

        try:
            payload = await reader.readexactly(header["length"])
        except (asyncio.IncompleteReadError, ConnectionError):
            return None

        return {
            "channel": header["channel"],
            "data": payload,
            "seq": header["seq"],
            "sim_time": header["sim_time"],
        }

    # ------------------------------------------------------------------
    # High-level receive helpers
    # ------------------------------------------------------------------

    async def receive_video(self) -> np.ndarray | None:
        """Receive the next video frame and decode it to a NumPy RGB array.

        Returns ``None`` on connection loss.
        """
        if self._video_reader is None:
            raise RuntimeError("Not connected -- call connect() first")

        frame = await self._read_frame(self._video_reader)
        if frame is None:
            return None

        img = Image.open(io.BytesIO(frame["data"]))
        return np.asarray(img)

    async def receive_telemetry(self) -> dict | None:
        """Receive the next telemetry frame and return it as a dict.

        Returns ``None`` on connection loss.
        """
        if self._telemetry_reader is None:
            raise RuntimeError("Not connected -- call connect() first")

        frame = await self._read_frame(self._telemetry_reader)
        if frame is None:
            return None

        return {
            "telemetry": json.loads(frame["data"].decode("utf-8")),
            "seq": frame["seq"],
            "sim_time": frame["sim_time"],
        }

    # ------------------------------------------------------------------
    # Display loop (requires OpenCV)
    # ------------------------------------------------------------------

    async def run_display(self) -> None:
        """Main loop: receive video frames, overlay the latest telemetry,
        and display them in an OpenCV window.  Press 'q' to quit.

        OpenCV (``cv2``) is imported lazily so the rest of the client can
        be used without it.
        """
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV (cv2) is required for run_display()")
            raise

        await self.connect()

        latest_telemetry: dict = {}
        window_name = "Orbital Simulation"

        # Launch a background task that continuously updates telemetry.
        async def _telemetry_loop() -> None:
            nonlocal latest_telemetry
            while True:
                telem = await self.receive_telemetry()
                if telem is None:
                    break
                latest_telemetry = telem.get("telemetry", {})

        telemetry_task = asyncio.create_task(_telemetry_loop())

        try:
            while True:
                rgb = await self.receive_video()
                if rgb is None:
                    logger.info("Video stream ended")
                    break

                # OpenCV uses BGR ordering.
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # Overlay telemetry text.
                y_offset = 30
                for key, value in latest_telemetry.items():
                    text = f"{key}: {value}"
                    cv2.putText(
                        bgr, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        cv2.LINE_AA,
                    )
                    y_offset += 20

                cv2.imshow(window_name, bgr)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit requested by user")
                    break
        finally:
            telemetry_task.cancel()
            try:
                await telemetry_task
            except asyncio.CancelledError:
                pass
            cv2.destroyAllWindows()
            await self.disconnect()
