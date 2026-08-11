"""Tests for the wire protocol and non-blocking streaming primitives."""

import asyncio
import json

import pytest

from stream.protocol import (
    CHANNEL_TELEMETRY,
    HEADER_SIZE,
    decode_header,
    encode_frame,
)
from stream.server import StreamingServer, _FrameChannel


class _FakeWriter:
    def __init__(self):
        self.writes = []
        self.closed = False

    def get_extra_info(self, name):
        return ("test", 0) if name == "peername" else None

    def write(self, data):
        self.writes.append(data)

    async def drain(self):
        return None

    def close(self):
        self.closed = True


def test_protocol_header_roundtrip():
    payload = json.dumps({"altitude_km": 400.0}).encode()
    frame = encode_frame(CHANNEL_TELEMETRY, payload, 42, 123.5)

    header = decode_header(frame[:HEADER_SIZE])

    assert header == {
        "channel": CHANNEL_TELEMETRY,
        "length": len(payload),
        "seq": 42,
        "sim_time": 123.5,
    }
    assert frame[HEADER_SIZE:] == payload


def test_latest_value_channel_drops_stale_payload():
    async def scenario():
        writer = _FakeWriter()
        channel = _FrameChannel(writer, lambda data: data, "test")
        channel.put(b"old")
        channel.put(b"new")
        task = asyncio.create_task(channel.run())
        await asyncio.sleep(0)
        channel.closed = True
        channel.event.set()
        await task
        return channel, writer

    channel, writer = asyncio.run(scenario())
    assert channel.dropped == 1
    assert writer.writes == [b"new"]


def test_start_rolls_back_partially_opened_servers(monkeypatch):
    class _FakeServer:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

        async def wait_closed(self):
            return None

    first = _FakeServer()
    calls = 0

    async def fake_start_server(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        raise OSError("port already in use")

    monkeypatch.setattr(asyncio, "start_server", fake_start_server)
    server = StreamingServer()

    with pytest.raises(OSError, match="port already in use"):
        asyncio.run(server.start())

    assert first.closed
    assert server._servers == []


def test_browser_viewer_reconnects_video_stream():
    html = (StreamingServer()._viewer_html).decode("utf-8")

    assert "video.onerror" in html
    assert "setTimeout(connectVideo" in html
    assert "/video.mjpg?t=${Date.now()}" in html
