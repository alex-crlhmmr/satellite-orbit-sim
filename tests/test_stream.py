"""Tests for the wire protocol and non-blocking streaming primitives."""

import asyncio
import json

import numpy as np
import pytest

from stream.client import StreamingClient
from stream.protocol import (
    CHANNEL_TELEMETRY,
    CHANNEL_VIDEO,
    HEADER_SIZE,
    MAX_TELEMETRY_PAYLOAD,
    decode_header,
    encode_frame,
    encode_video_frame,
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


def test_protocol_rejects_unknown_channels():
    with pytest.raises(ValueError, match="unsupported channel id"):
        encode_frame(99, b"payload", 1, 0.0)


def test_client_rejects_wrong_channel_and_oversized_payload():
    async def read(data, expected_channel, maximum):
        reader = asyncio.StreamReader()
        reader.feed_data(data)
        reader.feed_eof()
        return await StreamingClient()._read_frame(
            reader, expected_channel, maximum
        )

    wrong_channel = encode_frame(CHANNEL_VIDEO, b"jpeg", 1, 0.0)
    assert asyncio.run(read(
        wrong_channel, CHANNEL_TELEMETRY, MAX_TELEMETRY_PAYLOAD
    )) is None

    oversized = encode_frame(
        CHANNEL_TELEMETRY, b"x" * 8, 1, 0.0
    )
    assert asyncio.run(read(oversized, CHANNEL_TELEMETRY, 4)) is None


def test_client_video_decoder_preserves_frame_metadata():
    async def scenario():
        encoded = encode_video_frame(
            np.zeros((4, 6, 3), dtype=np.uint8), seq=17, sim_time=42.5
        )
        reader = asyncio.StreamReader()
        reader.feed_data(encoded)
        reader.feed_eof()
        client = StreamingClient()
        client._video_reader = reader
        return await client.receive_video_frame()

    frame = asyncio.run(scenario())
    assert frame["image"].shape == (4, 6, 3)
    assert frame["seq"] == 17
    assert frame["sim_time"] == 42.5


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


def test_browser_viewer_prefers_webrtc_and_retains_mjpeg_fallback():
    html = (StreamingServer()._viewer_html).decode("utf-8")

    assert "RTCPeerConnection" in html
    assert 'type: "offer"' in html
    assert "setTimeout(connectMJPEG" in html
    assert "/video.mjpg?t=${Date.now()}" in html
    assert "setTimeout(connectTelemetry, 1000)" in html


def test_webrtc_only_frame_avoids_jpeg_encoding(monkeypatch):
    class _FakeWebRTC:
        def __init__(self):
            self.frames = []

        def push_frame(self, frame):
            self.frames.append(frame)

    server = StreamingServer()
    server._webrtc = _FakeWebRTC()
    monkeypatch.setattr(
        "stream.server.Image.fromarray",
        lambda *args, **kwargs: pytest.fail("JPEG encoding should not run"),
    )
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    asyncio.run(server.send_video_frame(frame, 1, 0.0))
    assert len(server._webrtc.frames) == 1
    assert server._webrtc.frames[0] is frame


def test_browser_telemetry_omits_constellation_payload():
    class _FakeChannel:
        closed = False

        def __init__(self):
            self.payload = None

        def put(self, payload):
            self.payload = payload

    server = StreamingServer()
    channel = _FakeChannel()
    server._http_telemetry_channels.append(channel)
    telemetry = {
        "altitude_km": 550.0,
        "target_id": "sat-01",
        "satellite_count": 20,
        "satellites": [{"id": f"sat-{index:02d}"} for index in range(20)],
    }

    asyncio.run(server.send_telemetry(telemetry, seq=7, sim_time=35.0))

    browser_payload = json.loads(channel.payload)
    assert browser_payload == {
        "altitude_km": 550.0,
        "target_id": "sat-01",
        "satellite_count": 20,
        "seq": 7,
    }
