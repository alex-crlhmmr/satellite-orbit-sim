"""Jetson hardware-H.264 WebRTC transport.

This adapter deliberately uses the system GStreamer installation supplied by
JetPack. It is optional: callers can retain MJPEG on machines without the
NVIDIA encoder or GStreamer WebRTC plugins.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from dataclasses import dataclass

import numpy as np
import websockets

logger = logging.getLogger(__name__)


def _load_gstreamer():
    # Ubuntu installs PyGObject here, outside uv's isolated environment.
    system_packages = "/usr/lib/python3/dist-packages"
    if system_packages not in sys.path:
        sys.path.append(system_packages)
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstSdp", "1.0")
    gi.require_version("GstWebRTC", "1.0")
    from gi.repository import GLib, Gst, GstSdp, GstWebRTC

    Gst.init(None)
    required = ("nvv4l2h264enc", "nvvidconv", "webrtcbin", "rtph264pay")
    missing = [name for name in required if Gst.ElementFactory.find(name) is None]
    if missing:
        raise RuntimeError("missing GStreamer elements: " + ", ".join(missing))
    return GLib, Gst, GstSdp, GstWebRTC


@dataclass
class _Peer:
    websocket: object
    pipeline: object
    appsrc: object
    webrtc: object
    payloader: object
    rtp_caps: object
    frame_index: int = 0
    streaming: bool = False
    rtp_buffers: int = 0
    pushed_frames: int = 0
    stats_scheduled: bool = False


class WebRTCServer:
    """WebSocket-signalled WebRTC peers fed by RGB render frames."""

    def __init__(self, host: str, port: int, width: int, height: int,
                 fps: int, bitrate: int = 3_000_000, max_clients: int = 2):
        self.host = host
        self.port = int(port)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self.bitrate = int(bitrate)
        self.max_clients = int(max_clients)
        self.GLib, self.Gst, self.GstSdp, self.GstWebRTC = _load_gstreamer()
        self._loop = None
        self._server = None
        self._peer_by_id: dict[int, _Peer] = {}
        self._glib_loop = self.GLib.MainLoop()
        self._glib_thread = threading.Thread(
            target=self._glib_loop.run, name="gstreamer-main", daemon=True
        )

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._glib_thread.start()
        self._server = await websockets.serve(
            self._handle_client, self.host, self.port, max_size=1_000_000
        )

    async def stop(self) -> None:
        for peer in list(self._peer_by_id.values()):
            if not peer.streaming:
                continue
            self._close_peer(peer)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._glib_loop.quit()

    def client_count(self) -> int:
        return len(self._peer_by_id)

    def push_frame(self, rgb: np.ndarray) -> None:
        if not self._peer_by_id:
            return
        frame = np.ascontiguousarray(rgb, dtype=np.uint8)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(
                f"WebRTC frame must be {(self.height, self.width, 3)}, got {frame.shape}"
            )
        payload = frame.tobytes()
        duration = self.Gst.SECOND // self.fps
        for peer in list(self._peer_by_id.values()):
            buffer = self.Gst.Buffer.new_allocate(None, len(payload), None)
            buffer.fill(0, payload)
            buffer.pts = peer.frame_index * duration
            buffer.dts = buffer.pts
            buffer.duration = duration
            peer.frame_index += 1
            result = peer.appsrc.emit("push-buffer", buffer)
            if result != self.Gst.FlowReturn.OK:
                logger.warning("WebRTC appsrc returned %s", result)
            else:
                peer.pushed_frames += 1

    def _make_peer(self, websocket) -> _Peer:
        pipeline = self.Gst.parse_launch(
            "webrtcbin name=peer bundle-policy=max-bundle latency=0 "
            "appsrc name=source is-live=true format=time do-timestamp=false "
            f"caps=video/x-raw,format=RGB,width={self.width},height={self.height},"
            f"framerate={self.fps}/1 ! "
            "videoconvert ! video/x-raw,format=RGBA ! nvvidconv ! "
            "video/x-raw(memory:NVMM),format=NV12 ! "
            f"nvv4l2h264enc bitrate={self.bitrate} control-rate=1 "
            f"iframeinterval={self.fps} insert-sps-pps=true ! "
            "h264parse config-interval=-1 ! "
            "video/x-h264,profile=constrained-baseline,"
            "stream-format=byte-stream,alignment=au ! "
            "rtph264pay name=pay aggregate-mode=zero-latency "
            "config-interval=-1 pt=96 ! "
            "capsfilter name=rtpcaps caps=\"application/x-rtp,media=video,"
            "encoding-name=H264,clock-rate=90000,payload=96,packetization-mode=1\" ! "
            "peer."
        )
        peer = _Peer(
            websocket=websocket,
            pipeline=pipeline,
            appsrc=pipeline.get_by_name("source"),
            webrtc=pipeline.get_by_name("peer"),
            payloader=pipeline.get_by_name("pay"),
            rtp_caps=pipeline.get_by_name("rtpcaps"),
        )
        peer.webrtc.connect("on-ice-candidate", self._on_ice_candidate, peer)
        peer.webrtc.connect(
            "notify::ice-connection-state", self._on_state_changed, peer
        )
        peer.webrtc.connect(
            "notify::connection-state", self._on_state_changed, peer
        )
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::error", self._on_pipeline_error, peer)
        peer.payloader.get_static_pad("src").add_probe(
            self.Gst.PadProbeType.BUFFER, self._on_rtp_buffer, peer
        )
        pipeline.set_state(self.Gst.State.READY)
        return peer

    async def _handle_client(self, websocket) -> None:
        if len(self._peer_by_id) >= self.max_clients:
            await websocket.close(code=1013, reason="WebRTC client limit reached")
            return
        peer = self._make_peer(websocket)
        peer_id = id(peer)
        self._peer_by_id[peer_id] = peer
        logger.info("WebRTC client connected")
        try:
            async for raw in websocket:
                message = json.loads(raw)
                if message.get("type") == "offer":
                    print("  [webrtc] SDP offer received")
                    self._set_offer(peer, message["sdp"])
                elif message.get("type") == "ice":
                    peer.webrtc.emit(
                        "add-ice-candidate", int(message["sdpMLineIndex"]),
                        message["candidate"],
                    )
                elif message.get("type") == "client-stats":
                    logger.info("WebRTC browser stats: %s", message.get("stats", {}))
        finally:
            self._peer_by_id.pop(peer_id, None)
            self._close_peer(peer)
            logger.info("WebRTC client disconnected")

    def _set_offer(self, peer: _Peer, sdp_text: str) -> None:
        h264_payloads = []
        fmtp_by_payload = {}
        for line in sdp_text.splitlines():
            if line.startswith("a=rtpmap:") and " H264/90000" in line.upper():
                h264_payloads.append(int(line.split(":", 1)[1].split()[0]))
            elif line.startswith("a=fmtp:"):
                payload, parameters = line.split(":", 1)[1].split(" ", 1)
                fmtp_by_payload[int(payload)] = parameters.lower()
        compatible = []
        for payload in h264_payloads:
            parameters = fmtp_by_payload.get(payload, "")
            if "packetization-mode=1" not in parameters:
                continue
            profile = next(
                (part.split("=", 1)[1] for part in parameters.split(";")
                 if part.startswith("profile-level-id=")),
                "",
            )
            if profile.startswith("42"):
                compatible.append((profile.startswith("42e"), payload, profile))
        if not compatible:
            raise ValueError("browser offer does not contain H.264")
        _, h264_payload, profile_level_id = max(compatible)
        peer.payloader.set_property("pt", h264_payload)
        rtp_caps = (
            "application/x-rtp,media=video,encoding-name=H264,"
            f"clock-rate=90000,payload={h264_payload},packetization-mode=(string)1,"
            f"profile-level-id=(string){profile_level_id},"
            "level-asymmetry-allowed=(string)1"
        )
        peer.rtp_caps.set_property(
            "caps", self.Gst.Caps.from_string(rtp_caps)
        )
        print(
            f"  [webrtc] selected H.264 payload {h264_payload}, "
            f"profile {profile_level_id}"
        )
        filtered_lines = []
        for line in sdp_text.splitlines():
            if line.startswith("m=video "):
                fields = line.split()
                line = " ".join(fields[:3] + [str(h264_payload)])
            elif line.startswith(("a=rtpmap:", "a=fmtp:", "a=rtcp-fb:")):
                payload_text = line.split(":", 1)[1].split()[0]
                if payload_text != "*" and int(payload_text) != h264_payload:
                    continue
            filtered_lines.append(line)
        filtered_offer = "\r\n".join(filtered_lines) + "\r\n"
        result, sdp = self.GstSdp.SDPMessage.new_from_text(filtered_offer)
        if result != self.GstSdp.SDPResult.OK:
            raise ValueError("invalid WebRTC SDP offer")
        # GstWebRTCSessionDescription is available from the element's namespace.
        offer = self.GstWebRTC.WebRTCSessionDescription.new(
            self.GstWebRTC.WebRTCSDPType.OFFER, sdp
        )
        promise = self.Gst.Promise.new_with_change_func(
            self._on_offer_set, peer, None
        )
        peer.webrtc.emit("set-remote-description", offer, promise)

    def _on_offer_set(self, _promise, peer: _Peer, _unused) -> None:
        promise = self.Gst.Promise.new_with_change_func(
            self._on_answer_created, peer, None
        )
        peer.webrtc.emit("create-answer", None, promise)

    def _on_answer_created(self, promise, peer: _Peer, _unused) -> None:
        reply = promise.get_reply()
        answer = reply.get_value("answer")
        # Queue SDP before set-local-description starts ICE candidate emission.
        self._send(peer, {"type": "answer", "sdp": answer.sdp.as_text()})
        peer.webrtc.emit(
            "set-local-description", answer, self.Gst.Promise.new()
        )
        peer.pipeline.set_state(self.Gst.State.PLAYING)
        peer.streaming = True
        print("  [webrtc] SDP answer sent")
        for line in answer.sdp.as_text().splitlines():
            if line.startswith(("m=video", "a=rtpmap", "a=fmtp", "a=send")):
                print(f"  [webrtc] {line}")

    def _on_ice_candidate(self, _webrtc, mline_index: int,
                          candidate: str, peer: _Peer) -> None:
        self._send(peer, {
            "type": "ice", "candidate": candidate,
            "sdpMLineIndex": int(mline_index),
        })

    def _send(self, peer: _Peer, message: dict) -> None:
        if self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(
            peer.websocket.send(json.dumps(message)), self._loop
        )
        future.add_done_callback(self._report_send_failure)

    @staticmethod
    def _report_send_failure(future) -> None:
        error = future.exception()
        if error is not None:
            logger.warning("WebRTC signalling send failed: %s", error)

    def _on_state_changed(self, webrtc, _property, peer: _Peer) -> None:
        state = webrtc.get_property("connection-state").value_nick
        print(
            "  [webrtc] state="
            f"{state}, ice="
            f"{webrtc.get_property('ice-connection-state').value_nick}"
        )
        if state == "connected" and not peer.stats_scheduled:
            peer.stats_scheduled = True
            self.GLib.timeout_add_seconds(3, self._request_stats, peer)

    def _request_stats(self, peer: _Peer) -> bool:
        if id(peer) not in self._peer_by_id:
            return False
        promise = self.Gst.Promise.new_with_change_func(
            self._on_stats, peer, None
        )
        peer.webrtc.emit("get-stats", None, promise)
        print(
            f"  [webrtc] frames accepted: {peer.pushed_frames}; "
            f"RTP buffers produced: {peer.rtp_buffers}"
        )
        return False

    def _on_rtp_buffer(self, _pad, _info, peer: _Peer):
        peer.rtp_buffers += 1
        return self.Gst.PadProbeReturn.OK

    @staticmethod
    def _on_stats(promise, _peer: _Peer, _unused) -> None:
        reply = promise.get_reply()
        text = reply.to_string()
        fields = [
            token for token in text.replace(",", " ").split()
            if any(name in token for name in (
                "packets-sent", "bytes-sent", "frames-encoded", "frames-sent"
            ))
        ]
        print("  [webrtc] outbound stats: " + (" ".join(fields) or text[:500]))

    @staticmethod
    def _on_pipeline_error(_bus, message, _peer: _Peer) -> None:
        error, debug = message.parse_error()
        print(f"  [webrtc] pipeline error: {error}; {debug}")

    def _close_peer(self, peer: _Peer) -> None:
        try:
            peer.appsrc.emit("end-of-stream")
            peer.pipeline.set_state(self.Gst.State.NULL)
        except Exception:
            pass
