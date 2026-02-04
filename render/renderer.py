"""Main headless renderer using moderngl with EGL backend."""

import os
import numpy as np
import moderngl

from render.camera import Camera
from render.earth import Earth

# Directory containing the GLSL shader source files.
_SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")


class Renderer:
    """Headless 3D renderer for the orbital simulation.

    Uses moderngl with the EGL backend to render frames entirely on the GPU
    without a display server.  Each call to :meth:`render_frame` produces an
    RGB uint8 numpy array suitable for encoding to video or display.
    """

    def __init__(self, width: int = 1280, height: int = 720, config=None):
        """Initialise the EGL context, framebuffer, shaders, and scene objects.

        Args:
            width: Framebuffer width in pixels.
            height: Framebuffer height in pixels.
            config: Optional dict with configuration overrides.  Recognised
                keys include ``day_texture``, ``night_texture``.
        """
        self.width = width
        self.height = height
        self.config = config or {}

        # Create headless EGL context
        self.ctx = moderngl.create_context(standalone=True, backend="egl")
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # Off-screen framebuffer
        self.color_attachment = self.ctx.texture((width, height), 3)
        self.depth_attachment = self.ctx.depth_renderbuffer((width, height))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_attachment],
            depth_attachment=self.depth_attachment,
        )

        # Shaders
        self.earth_prog = self._load_shader("earth")
        self.orbit_prog = self._load_shader("orbit")

        # Scene objects
        self.camera = Camera(aspect=width / height)
        self.earth = Earth(self.ctx)

        # Load Earth textures (graceful fallback to procedural)
        day_path = self.config.get(
            "day_texture",
            os.path.join(os.path.dirname(_SHADER_DIR), "..", "assets", "earth_day.jpg"),
        )
        night_path = self.config.get(
            "night_texture",
            os.path.join(os.path.dirname(_SHADER_DIR), "..", "assets", "earth_night.jpg"),
        )
        self.earth.load_textures(self.ctx, day_path, night_path)

        # Reusable satellite point-sprite buffer (single vertex)
        self._sat_vbo = self.ctx.buffer(reserve=4 * 4)  # vec3 position + float alpha

    # ------------------------------------------------------------------
    # Shader loading
    # ------------------------------------------------------------------

    def _load_shader(self, name: str):
        """Read vertex and fragment shader sources and compile a program.

        Args:
            name: Base name of the shader pair (e.g. ``"earth"`` loads
                ``earth.vert`` and ``earth.frag``).

        Returns:
            A compiled ``moderngl.Program``.
        """
        vert_path = os.path.join(_SHADER_DIR, f"{name}.vert")
        frag_path = os.path.join(_SHADER_DIR, f"{name}.frag")

        with open(vert_path, "r") as fh:
            vert_src = fh.read()
        with open(frag_path, "r") as fh:
            frag_src = fh.read()

        return self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_camera_mode(self, mode: str):
        """Switch camera mode.

        Args:
            mode: 'tracking' (follows satellite), 'fixed' (inertial overview),
                  or 'split' (both views side by side).
        """
        self.camera_mode = mode

    def _render_scene(self, sat_positions, sun_dir, gmst, trail_positions, vp):
        """Draw the full scene (Earth + trail + satellites) with a given VP matrix."""
        self._draw_earth(gmst, sun_dir, vp)
        if trail_positions is not None:
            self._draw_trail(trail_positions, vp)
        for pos in sat_positions:
            self._draw_satellite(pos, vp)

    def render_frame(self, sat_positions, sun_pos, gmst: float,
                     trail_positions=None) -> np.ndarray:
        """Render a complete frame and return the pixel data.

        Args:
            sat_positions: Satellite ECI positions — a single (3,) array or
                an (N, 3) array for multiple satellites [m].
            sun_pos: Sun position in ECI [m] (used to derive direction).
            gmst: Greenwich Mean Sidereal Time angle [rad].
            trail_positions: Optional (M, 3) array of past positions for
                drawing the orbit trail.

        Returns:
            RGB uint8 numpy array of shape ``(height, width, 3)``.
        """
        sat_positions = np.atleast_2d(np.asarray(sat_positions, dtype=np.float64))

        # Normalised sun direction
        sun_pos = np.asarray(sun_pos, dtype=np.float64)
        sun_len = np.linalg.norm(sun_pos)
        sun_dir = sun_pos / sun_len if sun_len > 0 else np.array([1, 0, 0], dtype=np.float64)

        mode = getattr(self, 'camera_mode', 'tracking')

        if mode == 'split':
            # Render both views side by side into one frame
            # Left half: tracking view, Right half: fixed inertial view
            half_w = self.width // 2

            # --- Left: tracking ---
            self.fbo.use()
            self.ctx.clear(0.0, 0.0, 0.02, 1.0)
            self.ctx.viewport = (0, 0, half_w, self.height)
            self.camera.track_satellite(sat_positions[0])
            # Adjust aspect ratio for half-width
            old_aspect = self.camera.aspect
            self.camera.aspect = (half_w / self.height)
            self.camera._dirty_proj = True
            vp = self.camera.vp_matrix()
            self._render_scene(sat_positions, sun_dir, gmst, trail_positions, vp)

            # --- Right: fixed inertial ---
            self.ctx.viewport = (half_w, 0, self.width - half_w, self.height)
            self.camera.fixed_inertial(sat_positions[0])
            self.camera._dirty_proj = True
            vp = self.camera.vp_matrix()
            self._render_scene(sat_positions, sun_dir, gmst, trail_positions, vp)

            # Restore
            self.camera.aspect = old_aspect
            self.camera._dirty_proj = True
            self.ctx.viewport = (0, 0, self.width, self.height)
        else:
            # Single camera mode
            self.fbo.use()
            self.ctx.clear(0.0, 0.0, 0.02, 1.0)

            if mode == 'fixed':
                self.camera.fixed_inertial(sat_positions[0])
            else:
                self.camera.track_satellite(sat_positions[0])
            vp = self.camera.vp_matrix()
            self._render_scene(sat_positions, sun_dir, gmst, trail_positions, vp)

        # Read pixels
        raw = self.fbo.read(components=3)
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        # OpenGL origin is bottom-left; flip to top-left
        frame = np.flipud(frame).copy()
        return frame

    # ------------------------------------------------------------------
    # Internal draw helpers
    # ------------------------------------------------------------------

    def _draw_earth(self, gmst: float, sun_dir, vp: np.ndarray):
        """Render the Earth sphere."""
        model = Earth.get_model_matrix(gmst)
        self.earth.render(self.earth_prog, vp, model, sun_dir)

    def _draw_trail(self, trail_positions, vp: np.ndarray):
        """Render the orbit trail as a fading line strip."""
        trail = np.asarray(trail_positions, dtype=np.float32)
        if trail.ndim != 2 or trail.shape[0] < 2:
            return

        n = trail.shape[0]
        # Alpha ramps from 0 (oldest) to 1 (newest)
        alphas = np.linspace(0.0, 1.0, n, dtype=np.float32)

        # Interleave: x, y, z, alpha per vertex
        data = np.empty((n, 4), dtype=np.float32)
        data[:, :3] = trail
        data[:, 3] = alphas

        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(
            self.orbit_prog,
            [(vbo, "3f 1f", "position", "alpha")],
        )

        self.orbit_prog["mvp"].write(vp.astype(np.float32).T.tobytes())

        vao.render(moderngl.LINE_STRIP)
        vao.release()
        vbo.release()

    def _draw_satellite(self, sat_pos, vp: np.ndarray):
        """Render a satellite as a bright point sprite."""
        pos = np.asarray(sat_pos, dtype=np.float32).reshape(3)
        # Reuse orbit shader with full alpha for a single point
        data = np.empty(4, dtype=np.float32)
        data[:3] = pos
        data[3] = 1.0

        self._sat_vbo.write(data.tobytes())
        vao = self.ctx.vertex_array(
            self.orbit_prog,
            [(self._sat_vbo, "3f 1f", "position", "alpha")],
        )
        self.orbit_prog["mvp"].write(vp.astype(np.float32).T.tobytes())

        self.ctx.point_size = 6.0
        vao.render(moderngl.POINTS)
        vao.release()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Release all GPU resources."""
        for resource in (
            self.color_attachment,
            self.depth_attachment,
            self.fbo,
            self.earth_prog,
            self.orbit_prog,
            self.earth.vbo,
            self.earth.ibo,
            self._sat_vbo,
        ):
            try:
                resource.release()
            except Exception:
                pass

        if self.earth.day_texture is not None:
            try:
                self.earth.day_texture.release()
            except Exception:
                pass
        if self.earth.night_texture is not None:
            try:
                self.earth.night_texture.release()
            except Exception:
                pass

        try:
            self.ctx.release()
        except Exception:
            pass
