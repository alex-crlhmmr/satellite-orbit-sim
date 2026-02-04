"""
Remote viewer for orbital simulation streaming server.

Usage:
    python viewer.py --host 128.12.11.135
    python viewer.py --host 128.12.11.135 --save-frames
    python viewer.py --host 128.12.11.135 --headless
"""

import asyncio
import argparse
import struct
import io
import json
import sys
import time
import numpy as np
from PIL import Image

MAGIC = b"ORBT"
HEADER_FMT = ">4sBIId"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


async def read_frame(reader):
    """Read one protocol frame. Returns (channel, payload, seq, sim_time) or None."""
    try:
        hdr = await reader.readexactly(HEADER_SIZE)
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    magic, ch, plen, seq, sim_time = struct.unpack(HEADER_FMT, hdr)
    if magic != MAGIC:
        return None
    try:
        payload = await reader.readexactly(plen)
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    return ch, payload, seq, sim_time


async def run_viewer(host, video_port, telemetry_port, headless, save_frames, save_dir,
                     save_telemetry=False, telemetry_file="telemetry.jsonl"):
    print(f"Connecting to {host}...")

    try:
        vr, vw = await asyncio.open_connection(host, video_port)
        print(f"  Video connected ({host}:{video_port})")
    except OSError as e:
        print(f"  Video connection failed: {e}")
        return

    try:
        tr, tw = await asyncio.open_connection(host, telemetry_port)
        print(f"  Telemetry connected ({host}:{telemetry_port})")
    except OSError as e:
        print(f"  Telemetry connection failed: {e}")
        vw.close()
        return

    # Try importing OpenCV for live display
    cv2 = None
    if not headless:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError:
            print("  OpenCV not available — falling back to headless mode")
            headless = True

    if save_frames:
        from pathlib import Path
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"  Saving frames to {save_path}/")

    telem_log = None
    if save_telemetry:
        telem_log = open(telemetry_file, "w")
        print(f"  Recording telemetry to {telemetry_file}")

    # Background telemetry reader
    latest_telem = {}
    telem_count = 0

    async def telem_loop():
        nonlocal latest_telem, telem_count
        while True:
            result = await read_frame(tr)
            if result is None:
                break
            _, payload, seq_t, sim_t = result
            try:
                latest_telem = json.loads(payload.decode("utf-8"))
                if telem_log is not None:
                    record = {"seq": seq_t, "sim_time": sim_t, **latest_telem}
                    telem_log.write(json.dumps(record) + "\n")
                    telem_count += 1
                    if telem_count % 100 == 0:
                        telem_log.flush()
            except json.JSONDecodeError:
                pass

    telem_task = asyncio.create_task(telem_loop())

    frame_count = 0
    fps_start = time.monotonic()
    fps_frames = 0

    try:
        while True:
            result = await read_frame(vr)
            if result is None:
                print("\nVideo stream ended")
                break

            _, payload, seq, sim_time = result
            img = np.array(Image.open(io.BytesIO(payload)))
            frame_count += 1
            fps_frames += 1

            # Compute display FPS
            now = time.monotonic()
            elapsed = now - fps_start
            fps = fps_frames / elapsed if elapsed > 0 else 0
            if elapsed > 2.0:
                fps_start = now
                fps_frames = 0

            alt = latest_telem.get("altitude_km", 0)
            inc = latest_telem.get("inclination_deg", 0)
            ecc = latest_telem.get("eccentricity", 0)
            sma = latest_telem.get("semi_major_axis_km", 0)
            spd = latest_telem.get("speed_ms", 0)

            if headless:
                if frame_count % 30 == 1:
                    print(f"\r  Frame {seq:5d} | t={sim_time:8.0f}s | "
                          f"alt={alt:7.1f}km | sma={sma:8.1f}km | "
                          f"e={ecc:.6f} | i={inc:5.1f}° | "
                          f"v={spd:.0f}m/s | {fps:.1f}fps", end="", flush=True)
            else:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Telemetry overlay
                overlay_lines = [
                    f"t = {sim_time:.0f}s ({sim_time/3600:.1f}h)",
                    f"Alt: {alt:.1f} km",
                    f"SMA: {sma:.1f} km",
                    f"Ecc: {ecc:.6f}",
                    f"Inc: {inc:.2f} deg",
                    f"Speed: {spd:.0f} m/s",
                    f"FPS: {fps:.1f}",
                ]
                y = 25
                for line in overlay_lines:
                    cv2.putText(bgr, line, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                                cv2.LINE_AA)
                    y += 22

                cv2.imshow("Orbital Simulation", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\nQuit requested")
                    break
                elif key == ord("s"):
                    fname = f"screenshot_{seq}.png"
                    cv2.imwrite(fname, bgr)
                    print(f"\n  Saved {fname}")

            if save_frames and frame_count % 30 == 0:
                Image.fromarray(img).save(f"{save_dir}/frame_{seq:06d}.png")

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        telem_task.cancel()
        try:
            await telem_task
        except asyncio.CancelledError:
            pass
        if cv2 is not None:
            cv2.destroyAllWindows()
        vw.close()
        tw.close()

    if telem_log is not None:
        telem_log.close()
        print(f"Saved {telem_count} telemetry records to {telemetry_file}")
    print(f"Received {frame_count} frames total")


def main():
    parser = argparse.ArgumentParser(description="Orbital Simulation Viewer")
    parser.add_argument("--host", default="128.12.11.135",
                        help="Jetson IP address (default: 128.12.11.135)")
    parser.add_argument("--video-port", type=int, default=9100)
    parser.add_argument("--telemetry-port", type=int, default=9101)
    parser.add_argument("--headless", action="store_true",
                        help="Terminal-only mode (no OpenCV window)")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save every 30th frame as PNG")
    parser.add_argument("--save-dir", default="captured_frames",
                        help="Directory for saved frames")
    parser.add_argument("--save-telemetry", action="store_true",
                        help="Record all telemetry to a JSONL file")
    parser.add_argument("--telemetry-file", default="telemetry.jsonl",
                        help="Output file for telemetry (default: telemetry.jsonl)")
    args = parser.parse_args()

    asyncio.run(run_viewer(
        args.host, args.video_port, args.telemetry_port,
        args.headless, args.save_frames, args.save_dir,
        args.save_telemetry, args.telemetry_file,
    ))


if __name__ == "__main__":
    main()
