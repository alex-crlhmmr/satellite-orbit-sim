"""
Asyncio TCP streaming server for orbital simulation video and telemetry.

Runs two TCP servers (one for video, one for telemetry). Connected clients
receive broadcast frames produced by the simulation loop.
"""

import asyncio
import logging
from typing import List, Tuple

import numpy as np

from .protocol import encode_telemetry, encode_video_frame

logger = logging.getLogger(__name__)


class StreamingServer:
    """Broadcasts video frames and telemetry to connected TCP clients."""

    def __init__(self, video_port: int = 9100,
                 telemetry_port: int = 9101) -> None:
        self._video_port = video_port
        self._telemetry_port = telemetry_port

        self._video_clients: List[asyncio.StreamWriter] = []
        self._telemetry_clients: List[asyncio.StreamWriter] = []

        self._video_server: asyncio.AbstractServer | None = None
        self._telemetry_server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._video_server = await asyncio.start_server(
            self._handle_video_client, "0.0.0.0", self._video_port
        )
        self._telemetry_server = await asyncio.start_server(
            self._handle_telemetry_client, "0.0.0.0", self._telemetry_port
        )

    async def stop(self) -> None:
        for server in (self._video_server, self._telemetry_server):
            if server is not None:
                server.close()
                await server.wait_closed()
        for writer in self._video_clients + self._telemetry_clients:
            try:
                writer.close()
            except Exception:
                pass
        self._video_clients.clear()
        self._telemetry_clients.clear()

    async def _handle_video_client(self, reader: asyncio.StreamReader,
                                   writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        self._video_clients.append(writer)
        logger.warning("Video client connected: %s (total: %d)", addr, len(self._video_clients))
        print(f"  [stream] Video client connected: {addr}")
        try:
            # Hold connection open until client disconnects
            await reader.read(-1)
        except Exception:
            pass
        finally:
            if writer in self._video_clients:
                self._video_clients.remove(writer)
            try:
                writer.close()
            except Exception:
                pass
            print(f"  [stream] Video client disconnected: {addr}")

    async def _handle_telemetry_client(self, reader: asyncio.StreamReader,
                                       writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        self._telemetry_clients.append(writer)
        logger.warning("Telemetry client connected: %s (total: %d)", addr, len(self._telemetry_clients))
        print(f"  [stream] Telemetry client connected: {addr}")
        try:
            await reader.read(-1)
        except Exception:
            pass
        finally:
            if writer in self._telemetry_clients:
                self._telemetry_clients.remove(writer)
            try:
                writer.close()
            except Exception:
                pass
            print(f"  [stream] Telemetry client disconnected: {addr}")

    async def send_video_frame(self, rgb_array: np.ndarray, seq: int,
                               sim_time: float) -> None:
        if not self._video_clients:
            return
        data = encode_video_frame(rgb_array, seq, sim_time)
        await self._broadcast(data, self._video_clients)

    async def send_telemetry(self, telemetry_dict: dict, seq: int,
                             sim_time: float) -> None:
        if not self._telemetry_clients:
            return
        data = encode_telemetry(telemetry_dict, seq, sim_time)
        await self._broadcast(data, self._telemetry_clients)

    async def _broadcast(self, data: bytes,
                         clients: List[asyncio.StreamWriter]) -> None:
        dead: List[asyncio.StreamWriter] = []
        for writer in clients:
            try:
                writer.write(data)
                await writer.drain()
            except Exception:
                dead.append(writer)
        for writer in dead:
            if writer in clients:
                clients.remove(writer)
            try:
                writer.close()
            except Exception:
                pass

    def client_count(self) -> Tuple[int, int]:
        return len(self._video_clients), len(self._telemetry_clients)
