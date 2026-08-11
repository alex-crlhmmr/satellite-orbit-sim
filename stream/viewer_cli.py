"""Command-line viewer for the simulator's binary streaming protocol."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import TextIO

from PIL import Image

from .client import StreamingClient


async def run_viewer(
    host: str,
    video_port: int,
    telemetry_port: int,
    headless: bool,
    save_frames: bool,
    save_dir: str,
    save_telemetry: bool = False,
    telemetry_file: str = "telemetry.jsonl",
) -> None:
    """Connect, display or record frames, and close all resources cleanly."""
    cv2 = None
    if not headless:
        try:
            import cv2 as imported_cv2

            cv2 = imported_cv2
        except ImportError:
            print("OpenCV unavailable; using headless mode")
            headless = True

    save_path = Path(save_dir) if save_frames else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)

    telemetry_log: TextIO | None = None
    client = StreamingClient(host, video_port, telemetry_port)
    telemetry_task: asyncio.Task | None = None
    latest_telemetry: dict = {}
    telemetry_count = 0

    try:
        if save_telemetry:
            telemetry_log = Path(telemetry_file).open("w", encoding="utf-8")
        await client.connect()
        print(f"Connected to {host} (video {video_port}, telemetry {telemetry_port})")

        async def receive_telemetry() -> None:
            nonlocal latest_telemetry, telemetry_count
            while True:
                frame = await client.receive_telemetry()
                if frame is None:
                    return
                latest_telemetry = frame["telemetry"]
                if telemetry_log is not None:
                    record = {
                        "seq": frame["seq"],
                        "sim_time": frame["sim_time"],
                        **latest_telemetry,
                    }
                    telemetry_log.write(json.dumps(record) + "\n")
                    telemetry_count += 1
                    if telemetry_count % 100 == 0:
                        telemetry_log.flush()

        telemetry_task = asyncio.create_task(receive_telemetry())
        frame_count = 0
        fps_started = time.monotonic()

        while True:
            frame = await client.receive_video_frame()
            if frame is None:
                print("\nVideo stream ended")
                break
            image = frame["image"]
            seq = frame["seq"]
            sim_time = frame["sim_time"]
            frame_count += 1
            elapsed = max(time.monotonic() - fps_started, 1e-9)
            fps = frame_count / elapsed

            altitude = latest_telemetry.get("altitude_km", 0.0)
            if headless:
                if frame_count % 30 == 1:
                    print(
                        f"\rFrame {seq:5d} | t={sim_time:8.0f}s | "
                        f"alt={altitude:7.1f} km | {fps:.1f} fps",
                        end="",
                        flush=True,
                    )
            else:
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                lines = [
                    f"t = {sim_time:.0f}s ({sim_time / 3600:.1f}h)",
                    f"Alt: {altitude:.1f} km",
                    f"SMA: {latest_telemetry.get('semi_major_axis_km', 0.0):.1f} km",
                    f"Ecc: {latest_telemetry.get('eccentricity', 0.0):.6f}",
                    f"Inc: {latest_telemetry.get('inclination_deg', 0.0):.2f} deg",
                    f"Speed: {latest_telemetry.get('speed_ms', 0.0):.0f} m/s",
                    f"FPS: {fps:.1f}",
                ]
                for index, line in enumerate(lines, start=1):
                    cv2.putText(
                        bgr, line, (10, 3 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1, cv2.LINE_AA,
                    )
                cv2.imshow("Orbital Simulation", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    cv2.imwrite(f"screenshot_{seq}.png", bgr)

            if save_path is not None and frame_count % 30 == 0:
                Image.fromarray(image).save(save_path / f"frame_{seq:06d}.png")
    finally:
        if telemetry_task is not None:
            telemetry_task.cancel()
            try:
                await telemetry_task
            except asyncio.CancelledError:
                pass
        if cv2 is not None:
            cv2.destroyAllWindows()
        await client.disconnect()
        if telemetry_log is not None:
            telemetry_log.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orbital Simulation Viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--video-port", type=int, default=9100)
    parser.add_argument("--telemetry-port", type=int, default=9101)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--save-dir", default="captured_frames")
    parser.add_argument("--save-telemetry", action="store_true")
    parser.add_argument("--telemetry-file", default="telemetry.jsonl")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(run_viewer(
            args.host,
            args.video_port,
            args.telemetry_port,
            args.headless,
            args.save_frames,
            args.save_dir,
            args.save_telemetry,
            args.telemetry_file,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted")


if __name__ == "__main__":
    main()
