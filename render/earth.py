"""Earth mesh generation and texture management for orbital rendering."""

import os

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

from core.constants import R_EARTH, R_EARTH_POLAR


class Earth:
    """Textured WGS-84 ellipsoid with day/night shading.

    Generates an ellipsoid, loads (or creates) day and night textures, and
    renders via moderngl with the supplied shader program.
    """

    def __init__(self, ctx, lat_divs: int = 64, lon_divs: int = 128):
        """Create Earth mesh and GPU buffers.

        Args:
            ctx: A moderngl.Context.
            lat_divs: Number of latitude divisions for the UV sphere.
            lon_divs: Number of longitude divisions for the UV sphere.
        """
        self.radius = R_EARTH
        self.polar_radius = R_EARTH_POLAR
        vertices, indices = self._generate_ellipsoid(lat_divs, lon_divs)
        self.num_indices = len(indices)
        self._create_buffers(ctx, vertices, indices)
        self._vao = None

        self.day_texture = None
        self.night_texture = None

    # ------------------------------------------------------------------
    # Mesh generation
    # ------------------------------------------------------------------

    def _generate_ellipsoid(self, lat_divs: int, lon_divs: int):
        """Generate a geodetically parameterized WGS-84 ellipsoid mesh.

        Returns:
            vertices: ndarray of shape (N, 8) — x,y,z, nx,ny,nz, u,v
            indices: ndarray of triangle indices (uint32).
        """
        verts = []
        for i in range(lat_divs + 1):
            latitude = np.pi / 2.0 - np.pi * i / lat_divs
            cos_latitude = np.cos(latitude)
            sin_latitude = np.sin(latitude)
            v_coord = i / lat_divs

            for j in range(lon_divs + 1):
                # Equirectangular maps run from -180 degrees at the left edge
                # through Greenwich at the centre to +180 at the right edge.
                longitude = -np.pi + 2.0 * np.pi * j / lon_divs
                sin_longitude = np.sin(longitude)
                cos_longitude = np.cos(longitude)
                u_coord = j / lon_divs

                x = self.radius * cos_latitude * cos_longitude
                y = self.radius * cos_latitude * sin_longitude
                z = self.polar_radius * sin_latitude

                normal = np.array([
                    x / self.radius**2,
                    y / self.radius**2,
                    z / self.polar_radius**2,
                ])
                normal /= np.linalg.norm(normal)
                nx, ny, nz = normal

                verts.append([x, y, z, nx, ny, nz, u_coord, v_coord])

        vertices = np.array(verts, dtype=np.float32)

        # Build triangle indices
        idx = []
        for i in range(lat_divs):
            for j in range(lon_divs):
                p0 = i * (lon_divs + 1) + j
                p1 = p0 + 1
                p2 = p0 + (lon_divs + 1)
                p3 = p2 + 1

                idx.extend([p0, p2, p1])
                idx.extend([p1, p2, p3])

        indices = np.array(idx, dtype=np.uint32)
        return vertices, indices

    # ------------------------------------------------------------------
    # Buffer creation
    # ------------------------------------------------------------------

    def _create_buffers(self, ctx, vertices: np.ndarray, indices: np.ndarray):
        """Create moderngl vertex buffer and index buffer.

        Args:
            ctx: moderngl.Context
            vertices: (N, 8) float32 array
            indices: uint32 index array
        """
        self.vbo = ctx.buffer(vertices.tobytes())
        self.ibo = ctx.buffer(indices.tobytes())

    # ------------------------------------------------------------------
    # Textures
    # ------------------------------------------------------------------

    def load_textures(self, ctx, day_path: str, night_path: str):
        """Load day and night Earth textures.

        If the files do not exist, procedural fallback textures are created.

        Args:
            ctx: moderngl.Context
            day_path: Path to daytime Earth texture image.
            night_path: Path to nighttime Earth texture image.
        """
        self.day_texture = self._load_or_generate(ctx, day_path, kind="day")
        self.night_texture = self._load_or_generate(ctx, night_path, kind="night")

    def _load_or_generate(self, ctx, path: str, kind: str):
        """Attempt to load a texture from *path*; fall back to procedural."""
        if os.path.isfile(path) and Image is not None:
            try:
                img = Image.open(path).convert("RGB")
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                data = img.tobytes()
                tex = ctx.texture(img.size, 3, data)
                tex.build_mipmaps()
                tex.filter = (ctx.LINEAR_MIPMAP_LINEAR, ctx.LINEAR)
                return tex
            except Exception:
                pass  # fall through to procedural

        # Procedural fallback
        if kind == "day":
            return self._procedural_day(ctx)
        return self._procedural_night(ctx)

    def _procedural_day(self, ctx, width: int = 512, height: int = 256):
        """Generate a simple blue/green day-side Earth texture."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Ocean blue
        img[:, :] = [30, 80, 160]
        # Green land bands (crude)
        for y in range(height):
            lat_frac = y / height
            # Rough land between 20-60 % latitude band
            if 0.2 < lat_frac < 0.45 or 0.55 < lat_frac < 0.8:
                for x in range(width):
                    lon_frac = x / width
                    # Simple continental shapes via modular arithmetic
                    if (int(lon_frac * 7) + int(lat_frac * 5)) % 3 == 0:
                        img[y, x] = [40, 130, 50]
        tex = ctx.texture((width, height), 3, img.tobytes())
        tex.build_mipmaps()
        return tex

    def _procedural_night(self, ctx, width: int = 512, height: int = 256):
        """Generate a dark-blue night-side texture with city-light dots."""
        rng = np.random.RandomState(42)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = [4, 4, 20]
        # Scatter "city lights"
        num_lights = 800
        ys = rng.randint(int(height * 0.15), int(height * 0.85), num_lights)
        xs = rng.randint(0, width, num_lights)
        for y, x in zip(ys, xs):
            brightness = rng.randint(120, 255)
            img[y, x] = [brightness, brightness, int(brightness * 0.7)]
        tex = ctx.texture((width, height), 3, img.tobytes())
        tex.build_mipmaps()
        return tex

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, prog, vp_matrix: np.ndarray, model_matrix: np.ndarray,
               sun_dir):
        """Render the Earth sphere.

        Args:
            prog: Compiled moderngl shader program (earth shader).
            vp_matrix: 4x4 view-projection matrix (column-major float32).
            model_matrix: 4x4 model matrix (column-major float32).
            sun_dir: Normalised sun direction in world space (3-element).
        """
        mvp = vp_matrix @ model_matrix

        prog["mvp"].write(mvp.astype(np.float32).T.tobytes())
        prog["model"].write(model_matrix.astype(np.float32).T.tobytes())

        sun_dir = np.asarray(sun_dir, dtype=np.float32)
        prog["sun_direction"].value = tuple(sun_dir)

        if self.day_texture is not None:
            self.day_texture.use(location=0)
            prog["day_texture"].value = 0
        if self.night_texture is not None:
            self.night_texture.use(location=1)
            prog["night_texture"].value = 1

        if self._vao is None:
            self._vao = prog.ctx.vertex_array(
                prog,
                [(self.vbo, "3f 3f 2f", "position", "normal", "texcoord")],
                index_buffer=self.ibo,
            )
        self._vao.render()

    @staticmethod
    def get_model_matrix(itrf_to_gcrf: np.ndarray) -> np.ndarray:
        """Embed the full ITRF-to-GCRF rotation in a 4x4 model matrix.

        Args:
            itrf_to_gcrf: Orthogonal 3x3 Earth-orientation matrix.

        Returns:
            4x4 column-major rotation matrix (float32).
        """
        rotation = np.asarray(itrf_to_gcrf, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("ITRF-to-GCRF rotation must have shape (3, 3)")
        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = rotation
        return m
