"""
Streaming server for orbital simulation video and telemetry.

Three concurrent servers run inside one StreamingServer instance:

* binary TCP video       (port 9100, default) — legacy Python viewer
* binary TCP telemetry   (port 9101, default) — legacy Python viewer
* HTTP browser endpoint  (port 8080, default) — open in any browser
    GET /              -> HTML viewer page
    GET /video.mjpg    -> multipart/x-mixed-replace MJPEG stream
    GET /telemetry.sse -> Server-Sent Events (JSON)
* optional WebRTC signalling (port 8081, default) — Jetson hardware H.264

Every video and telemetry client gets a single-slot latest-value buffer
that drops stale data if the consumer falls behind. The simulation loop
is therefore never blocked by a slow viewer.
"""

import asyncio
import io
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .protocol import encode_telemetry

logger = logging.getLogger(__name__)


_HTML_PATH = Path(__file__).parent / "viewer.html"
_MJPEG_BOUNDARY = b"orbitalframe"


# ---------------------------------------------------------------------------
# Per-client one-slot channel
# ---------------------------------------------------------------------------

class _FrameChannel:
    """
    Single-slot 'latest frame' buffer for one client.

    Producer just calls put(); if the previous frame hasn't been sent
    yet it is overwritten. The consumer task per channel drains the
    slot and writes to the wire.
    """

    __slots__ = ("writer", "frame_wrap", "latest", "event", "closed",
                 "addr", "name", "dropped", "sent")

    def __init__(self, writer: asyncio.StreamWriter, frame_wrap, name: str):
        self.writer = writer
        self.frame_wrap = frame_wrap        # bytes -> bytes
        self.latest: Optional[bytes] = None
        self.event = asyncio.Event()
        self.closed = False
        self.addr = writer.get_extra_info("peername")
        self.name = name
        self.dropped = 0
        self.sent = 0

    def put(self, payload: bytes) -> None:
        if self.closed:
            return
        if self.latest is not None:
            self.dropped += 1
        self.latest = payload
        self.event.set()

    async def run(self) -> None:
        """Drain the slot to the wire. Returns when the client disconnects."""
        try:
            while not self.closed:
                await self.event.wait()
                if self.closed:
                    break
                payload = self.latest
                self.latest = None
                self.event.clear()
                if payload is None:
                    continue
                try:
                    self.writer.write(self.frame_wrap(payload))
                    await self.writer.drain()
                    self.sent += 1
                except (ConnectionError, BrokenPipeError, asyncio.CancelledError):
                    self.closed = True
                    break
                except Exception:
                    self.closed = True
                    break
        finally:
            try:
                self.writer.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# StreamingServer
# ---------------------------------------------------------------------------

class StreamingServer:
    """Broadcasts video frames and telemetry to connected clients.

    Public API matches the previous version:
        await server.start()
        await server.send_video_frame(rgb_array, seq, sim_time)
        await server.send_telemetry(telemetry_dict, seq, sim_time)
        await server.stop()
        v, t = server.client_count()
    """

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        video_port: int = 9100,
        telemetry_port: int = 9101,
        http_port: int = 8080,
        jpeg_quality: int = 85,
        webrtc_enabled: bool = False,
        webrtc_port: int = 8081,
        webrtc_width: int = 1280,
        webrtc_height: int = 720,
        webrtc_fps: int = 15,
        webrtc_bitrate: int = 3_000_000,
    ) -> None:
        self._bind_host = bind_host
        self._video_port = video_port
        self._telemetry_port = telemetry_port
        self._http_port = http_port
        self._jpeg_quality = jpeg_quality
        self._webrtc = None
        self._webrtc_config = (
            webrtc_enabled, webrtc_port, webrtc_width, webrtc_height,
            webrtc_fps, webrtc_bitrate,
        )

        self._binary_video_channels: List[_FrameChannel] = []
        self._http_video_channels: List[_FrameChannel] = []

        self._binary_telemetry_channels: List[_FrameChannel] = []
        self._http_telemetry_channels: List[_FrameChannel] = []

        self._servers: List[asyncio.AbstractServer] = []
        self._tasks: List[asyncio.Task] = []

        try:
            self._viewer_html = _HTML_PATH.read_bytes()
        except FileNotFoundError:
            self._viewer_html = b"<h1>viewer.html missing</h1>"

    # ----------------------------- lifecycle ------------------------------

    async def start(self) -> None:
        try:
            self._servers.append(await asyncio.start_server(
                self._handle_binary_video, self._bind_host, self._video_port
            ))
            self._servers.append(await asyncio.start_server(
                self._handle_binary_telemetry, self._bind_host, self._telemetry_port
            ))
            self._servers.append(await asyncio.start_server(
                self._handle_http, self._bind_host, self._http_port
            ))
            enabled, port, width, height, fps, bitrate = self._webrtc_config
            if enabled:
                from .webrtc import WebRTCServer
                self._webrtc = WebRTCServer(
                    self._bind_host, port, width, height, fps, bitrate
                )
                await self._webrtc.start()
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        if self._webrtc is not None:
            await self._webrtc.stop()
            self._webrtc = None
        for server in self._servers:
            try:
                server.close()
                await server.wait_closed()
            except Exception:
                pass
        self._servers.clear()
        for ch in (self._binary_video_channels + self._http_video_channels
                   + self._binary_telemetry_channels
                   + self._http_telemetry_channels):
            ch.closed = True
            ch.event.set()
        self._binary_video_channels.clear()
        self._http_video_channels.clear()
        self._binary_telemetry_channels.clear()
        self._http_telemetry_channels.clear()
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

    # ----------------------------- producers ------------------------------

    async def send_video_frame(
        self, rgb_array: np.ndarray, seq: int, sim_time: float
    ) -> None:
        if self._webrtc is not None:
            self._webrtc.push_frame(rgb_array)
        has_binary = bool(self._binary_video_channels)
        has_http = bool(self._http_video_channels)
        if not (has_binary or has_http):
            return

        # Encode JPEG once (used by both transports).
        img = Image.fromarray(rgb_array, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._jpeg_quality)
        jpeg = buf.getvalue()

        if has_binary:
            from .protocol import CHANNEL_VIDEO, encode_frame
            binary_payload = encode_frame(CHANNEL_VIDEO, jpeg, seq, sim_time)
            for ch in list(self._binary_video_channels):
                if ch.closed:
                    self._binary_video_channels.remove(ch)
                else:
                    ch.put(binary_payload)

        if has_http:
            for ch in list(self._http_video_channels):
                if ch.closed:
                    self._http_video_channels.remove(ch)
                else:
                    ch.put(jpeg)

    async def send_telemetry(
        self, telemetry_dict: dict, seq: int, sim_time: float
    ) -> None:
        binary_clients = bool(self._binary_telemetry_channels)
        http_clients = bool(self._http_telemetry_channels)
        if not (binary_clients or http_clients):
            return

        if binary_clients:
            data = encode_telemetry(telemetry_dict, seq, sim_time)
            for ch in list(self._binary_telemetry_channels):
                if ch.closed:
                    self._binary_telemetry_channels.remove(ch)
                else:
                    ch.put(data)

        if http_clients:
            # The browser HUD consumes only the selected target's top-level
            # fields. Avoid duplicating the complete constellation over SSE;
            # binary telemetry retains the full schema for API consumers.
            browser_telemetry = {
                key: value for key, value in telemetry_dict.items()
                if key != "satellites"
            }
            payload = json.dumps(
                {**browser_telemetry, "seq": seq}, separators=(",", ":")
            ).encode("utf-8")
            for ch in list(self._http_telemetry_channels):
                if ch.closed:
                    self._http_telemetry_channels.remove(ch)
                else:
                    ch.put(payload)

    # ------------------------- binary TCP handlers -----------------------

    async def _handle_binary_video(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        addr = writer.get_extra_info("peername")
        ch = _FrameChannel(writer, lambda b: b, name="bin-video")
        self._binary_video_channels.append(ch)
        print(f"  [stream] Video client connected: {addr} "
              f"(binary, total: {len(self._binary_video_channels)})")
        try:
            task = asyncio.create_task(ch.run())
            self._tasks.append(task)
            # Watch for client disconnect.
            await reader.read(-1)
        except Exception:
            pass
        finally:
            ch.closed = True
            ch.event.set()
            if ch in self._binary_video_channels:
                self._binary_video_channels.remove(ch)
            print(f"  [stream] Video client disconnected: {addr} "
                  f"(sent={ch.sent}, dropped={ch.dropped})")

    async def _handle_binary_telemetry(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        addr = writer.get_extra_info("peername")
        ch = _FrameChannel(writer, lambda b: b, name="bin-telemetry")
        self._binary_telemetry_channels.append(ch)
        print(f"  [stream] Telemetry client connected: {addr} "
              f"(binary, total: {len(self._binary_telemetry_channels)})")
        try:
            task = asyncio.create_task(ch.run())
            self._tasks.append(task)
            await reader.read(-1)
        except Exception:
            pass
        finally:
            ch.closed = True
            ch.event.set()
            if ch in self._binary_telemetry_channels:
                self._binary_telemetry_channels.remove(ch)
            print(f"  [stream] Telemetry client disconnected: {addr} "
                  f"(sent={ch.sent}, dropped={ch.dropped})")

    # ------------------------------ HTTP ---------------------------------

    async def _handle_http(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        addr = writer.get_extra_info("peername")
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not request_line:
                return
            # Drain headers.
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                if not line or line in (b"\r\n", b"\n"):
                    break
            parts = request_line.split()
            if len(parts) < 2 or parts[0] != b"GET":
                await self._http_send_simple(writer, 405, b"method not allowed")
                return
            path = parts[1].split(b"?", 1)[0]

            if path == b"/" or path == b"/index.html":
                await self._http_send_html(writer)
            elif path == b"/video.mjpg":
                await self._http_serve_mjpeg(writer, reader, addr)
            elif path == b"/telemetry.sse":
                await self._http_serve_sse(writer, reader, addr)
            else:
                await self._http_send_simple(writer, 404, b"not found")
        except (asyncio.TimeoutError, ConnectionError):
            pass
        except Exception as e:
            logger.warning("HTTP handler error from %s: %s", addr, e)
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _http_send_simple(
        self, writer: asyncio.StreamWriter, status: int, body: bytes
    ) -> None:
        reason = {200: "OK", 404: "Not Found", 405: "Method Not Allowed"}.get(
            status, "Error"
        )
        head = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n"
        ).encode("ascii")
        try:
            writer.write(head + body)
            await writer.drain()
        except Exception:
            pass

    async def _http_send_html(self, writer: asyncio.StreamWriter) -> None:
        body = self._viewer_html
        head = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        try:
            writer.write(head + body)
            await writer.drain()
        except Exception:
            pass

    async def _http_serve_mjpeg(
        self,
        writer: asyncio.StreamWriter,
        reader: asyncio.StreamReader,
        addr,
    ) -> None:
        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; "
            f"boundary={_MJPEG_BOUNDARY.decode()}\r\n"
            "Cache-Control: no-store\r\n"
            "Pragma: no-cache\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        try:
            writer.write(header)
            await writer.drain()
        except Exception:
            return

        def wrap(jpeg: bytes) -> bytes:
            return (
                b"--" + _MJPEG_BOUNDARY + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )

        ch = _FrameChannel(writer, wrap, name="http-mjpeg")
        self._http_video_channels.append(ch)
        print(f"  [stream] Video client connected: {addr} "
              f"(http/mjpeg, total: {len(self._http_video_channels)})")
        run_task = asyncio.create_task(ch.run())
        try:
            # When the client closes the TCP connection, reader.read returns b''.
            await reader.read(-1)
        except Exception:
            pass
        finally:
            ch.closed = True
            ch.event.set()
            try:
                await asyncio.wait_for(run_task, timeout=0.5)
            except Exception:
                run_task.cancel()
            if ch in self._http_video_channels:
                self._http_video_channels.remove(ch)
            print(f"  [stream] Video client disconnected: {addr} "
                  f"(http/mjpeg, sent={ch.sent}, dropped={ch.dropped})")

    async def _http_serve_sse(
        self,
        writer: asyncio.StreamWriter,
        reader: asyncio.StreamReader,
        addr,
    ) -> None:
        header = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/event-stream; charset=utf-8\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        try:
            writer.write(header)
            await writer.drain()
        except Exception:
            return

        def wrap(payload: bytes) -> bytes:
            return b"data: " + payload + b"\n\n"

        ch = _FrameChannel(writer, wrap, name="http-sse")
        self._http_telemetry_channels.append(ch)
        print(f"  [stream] Telemetry client connected: {addr} "
              f"(http/sse, total: {len(self._http_telemetry_channels)})")
        run_task = asyncio.create_task(ch.run())
        try:
            await reader.read(-1)
        except Exception:
            pass
        finally:
            ch.closed = True
            ch.event.set()
            try:
                await asyncio.wait_for(run_task, timeout=0.5)
            except Exception:
                run_task.cancel()
            if ch in self._http_telemetry_channels:
                self._http_telemetry_channels.remove(ch)
            print(f"  [stream] Telemetry client disconnected: {addr} "
                  f"(http/sse, sent={ch.sent}, dropped={ch.dropped})")

    # ------------------------------ stats --------------------------------

    def client_count(self) -> Tuple[int, int]:
        webrtc_clients = 0 if self._webrtc is None else self._webrtc.client_count()
        v = (len(self._binary_video_channels) + len(self._http_video_channels)
             + webrtc_clients)
        t = len(self._binary_telemetry_channels) + len(self._http_telemetry_channels)
        return v, t
