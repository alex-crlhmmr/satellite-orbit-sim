"""End-to-end WebRTC H.264 decode/cadence probe for a running simulator."""

import argparse
import asyncio
import json
import statistics
import time

import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp


async def probe(uri: str, frame_count: int) -> None:
    peer = RTCPeerConnection()
    peer.addTransceiver("video", direction="recvonly")
    arrival_times = []
    complete = asyncio.Event()

    @peer.on("track")
    def on_track(track):
        print(f"track received: {track.kind}", flush=True)
        async def receive():
            while len(arrival_times) < frame_count:
                await track.recv()
                arrival_times.append(time.perf_counter())
            complete.set()

        asyncio.create_task(receive())

    async with websockets.connect(uri, ping_interval=None) as websocket:
        offer = await peer.createOffer()
        await peer.setLocalDescription(offer)
        await websocket.send(json.dumps({
            "type": "offer", "sdp": peer.localDescription.sdp,
        }))
        pending_candidates = []

        async def signalling():
            async for raw in websocket:
                message = json.loads(raw)
                if message["type"] == "answer":
                    await peer.setRemoteDescription(RTCSessionDescription(
                        sdp=message["sdp"], type="answer"
                    ))
                    for candidate in pending_candidates:
                        await peer.addIceCandidate(candidate)
                    pending_candidates.clear()
                elif message["type"] == "ice":
                    text = message["candidate"]
                    candidate_text = (
                        text.split(":", 1)[1] if text.startswith("candidate:") else text
                    )
                    if len(candidate_text.split()) < 8:
                        continue
                    candidate = candidate_from_sdp(
                        candidate_text
                    )
                    candidate.sdpMLineIndex = message["sdpMLineIndex"]
                    candidate.sdpMid = message.get("sdpMid", "0")
                    if peer.remoteDescription is None:
                        pending_candidates.append(candidate)
                    else:
                        await peer.addIceCandidate(candidate)

        signalling_task = asyncio.create_task(signalling())
        timeout = max(20.0, frame_count / 20.0 + 15.0)
        await asyncio.wait_for(complete.wait(), timeout=timeout)
        signalling_task.cancel()

    await peer.close()
    intervals = [b - a for a, b in zip(arrival_times, arrival_times[1:])]
    print(json.dumps({
        "decoded_frames": len(arrival_times),
        "mean_fps": 1.0 / statistics.mean(intervals),
        "p95_interval_ms": statistics.quantiles(intervals, n=20)[18] * 1000,
        "max_interval_ms": max(intervals) * 1000,
        "intervals_over_100ms": sum(value > 0.1 for value in intervals),
        "intervals_over_500ms": sum(value > 0.5 for value in intervals),
        "intervals_over_1s": sum(value > 1.0 for value in intervals),
        "intervals_over_2s": sum(value > 2.0 for value in intervals),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="ws://127.0.0.1:8085")
    parser.add_argument("--frames", type=int, default=150)
    args = parser.parse_args()
    asyncio.run(probe(args.uri, args.frames))


if __name__ == "__main__":
    main()
