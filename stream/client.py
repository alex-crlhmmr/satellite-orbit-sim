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
    CHANNEL_TELEMETRY,
    CHANNEL_VIDEO,
    HEADER_SIZE,
    MAX_TELEMETRY_PAYLOAD,
    MAX_VIDEO_PAYLOAD,
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
                try:
                    await self._video_writer.wait_closed()
                except OSError:
                    pass
                self._video_reader = None
                self._video_writer = None
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

    async def _read_frame(
        self,
        reader: asyncio.StreamReader,
        expected_channel: int,
        max_payload: int,
    ) -> dict | None:
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
            logger.warning("Received invalid protocol header")
            return None
        if header["channel"] != expected_channel:
            logger.warning(
                "Received channel %d on channel %d connection",
                header["channel"], expected_channel,
            )
            return None
        if header["length"] > max_payload:
            logger.warning(
                "Rejected %d-byte payload (limit %d)",
                header["length"], max_payload,
            )
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

    async def receive_video_frame(self) -> dict | None:
        """Receive video pixels together with sequence and simulation time.

        Returns ``None`` on connection loss.
        """
        if self._video_reader is None:
            raise RuntimeError("Not connected -- call connect() first")

        frame = await self._read_frame(
            self._video_reader, CHANNEL_VIDEO, MAX_VIDEO_PAYLOAD
        )
        if frame is None:
            return None

        with Image.open(io.BytesIO(frame["data"])) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        return {
            "image": rgb,
            "seq": frame["seq"],
            "sim_time": frame["sim_time"],
        }

    async def receive_video(self) -> np.ndarray | None:
        """Receive the next video frame as an RGB array."""
        frame = await self.receive_video_frame()
        return None if frame is None else frame["image"]

    async def receive_telemetry(self) -> dict | None:
        """Receive the next telemetry frame and return it as a dict.

        Returns ``None`` on connection loss.
        """
        if self._telemetry_reader is None:
            raise RuntimeError("Not connected -- call connect() first")

        frame = await self._read_frame(
            self._telemetry_reader, CHANNEL_TELEMETRY, MAX_TELEMETRY_PAYLOAD
        )
        if frame is None:
            return None

        return {
            "telemetry": json.loads(frame["data"].decode("utf-8")),
            "seq": frame["seq"],
            "sim_time": frame["sim_time"],
        }
