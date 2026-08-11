"""
Binary wire protocol for orbital simulation video and telemetry streaming.

Header format (21 bytes, big-endian):
    magic       4 bytes   b"ORBT"
    channel_id  1 byte    uint8   (1=video, 2=telemetry)
    payload_len 4 bytes   uint32
    sequence    4 bytes   uint32
    sim_time    8 bytes   float64 (double)
"""

import io
import json
import struct

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC = b"ORBT"

CHANNEL_VIDEO = 1
CHANNEL_TELEMETRY = 2
VALID_CHANNELS = frozenset({CHANNEL_VIDEO, CHANNEL_TELEMETRY})

# Defensive receive limits. These are intentionally much larger than normal
# simulator messages while preventing an invalid header from requesting an
# unbounded allocation/read.
MAX_VIDEO_PAYLOAD = 32 * 1024 * 1024
MAX_TELEMETRY_PAYLOAD = 1024 * 1024

# >  = big-endian
# 4s = 4-byte char[] (magic)
# B  = uint8          (channel_id)
# I  = uint32         (payload_len)
# I  = uint32         (sequence)
# d  = float64        (sim_time)
_HEADER_FMT = ">4sBIId"
HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 21


# ---------------------------------------------------------------------------
# Low-level encode / decode
# ---------------------------------------------------------------------------

def encode_frame(channel_id: int, payload_bytes: bytes, seq: int,
                 sim_time: float) -> bytes:
    """Pack a protocol header in front of *payload_bytes* and return the
    complete frame as a single ``bytes`` object."""
    if channel_id not in VALID_CHANNELS:
        raise ValueError(f"unsupported channel id: {channel_id}")
    header = struct.pack(
        _HEADER_FMT,
        MAGIC,
        channel_id,
        len(payload_bytes),
        seq,
        sim_time,
    )
    return header + payload_bytes


def decode_header(data: bytes) -> dict | None:
    """Unpack a protocol header from *data* (must be at least HEADER_SIZE
    bytes).  Returns a dict with keys ``channel``, ``length``, ``seq``,
    ``sim_time``, or ``None`` if the magic bytes do not match."""
    if len(data) < HEADER_SIZE:
        return None

    magic, channel_id, payload_len, seq, sim_time = struct.unpack(
        _HEADER_FMT, data[:HEADER_SIZE]
    )

    if magic != MAGIC or channel_id not in VALID_CHANNELS:
        return None

    return {
        "channel": channel_id,
        "length": payload_len,
        "seq": seq,
        "sim_time": sim_time,
    }


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def encode_video_frame(rgb_array: np.ndarray, seq: int, sim_time: float,
                       quality: int = 85) -> bytes:
    """Encode a NumPy RGB array to JPEG, wrap it in a protocol frame, and
    return the raw bytes ready for transmission."""
    img = Image.fromarray(rgb_array, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    jpeg_bytes = buf.getvalue()
    return encode_frame(CHANNEL_VIDEO, jpeg_bytes, seq, sim_time)


def encode_telemetry(telemetry_dict: dict, seq: int,
                     sim_time: float) -> bytes:
    """JSON-encode a telemetry dictionary and wrap it in a protocol frame."""
    payload = json.dumps(telemetry_dict, separators=(",", ":")).encode("utf-8")
    return encode_frame(CHANNEL_TELEMETRY, payload, seq, sim_time)
