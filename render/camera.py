"""Camera system with view and projection matrices for orbital rendering.

Matrices are built in **row-major** convention (standard numpy layout).
Before uploading to GLSL (which reads column-major), callers must
transpose: ``mat.T.astype(np.float32).tobytes()``.
"""

import numpy as np


class Camera:
    def __init__(self, fov: float = 45.0, aspect: float = 16 / 9,
                 near: float = 1e4, far: float = 5e8):
        self.fov = fov
        self.aspect = aspect
        self.near = near
        self.far = far

        self._eye = np.array([0.0, 0.0, 3e7], dtype=np.float64)
        self._target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        self._view = None
        self._proj = None
        self._dirty_view = True
        self._dirty_proj = True

    def look_at(self, eye, target, up=(0, 0, 1)):
        self._eye = np.asarray(eye, dtype=np.float64)
        self._target = np.asarray(target, dtype=np.float64)
        self._up = np.asarray(up, dtype=np.float64)
        self._dirty_view = True

    def perspective_matrix(self) -> np.ndarray:
        if self._dirty_proj:
            self._proj = self._build_perspective()
            self._dirty_proj = False
        return self._proj

    def view_matrix(self) -> np.ndarray:
        if self._dirty_view:
            self._view = self._build_look_at()
            self._dirty_view = False
        return self._view

    def vp_matrix(self) -> np.ndarray:
        """Return combined projection @ view (row-major)."""
        return self.perspective_matrix() @ self.view_matrix()

    def track_satellite(self, sat_pos, earth_center=(0, 0, 0),
                        distance_scale: float = 2.0):
        """Position camera behind the satellite, looking toward Earth.

        Camera follows the satellite so both the satellite and Earth are
        visible. The planet fills a good portion of the view.
        """
        sat_pos = np.asarray(sat_pos, dtype=np.float64)
        earth_center = np.asarray(earth_center, dtype=np.float64)

        direction = sat_pos - earth_center
        dist = np.linalg.norm(direction)
        if dist < 1.0:
            return

        direction_unit = direction / dist

        # Place camera behind the satellite, looking toward Earth
        eye = earth_center + direction_unit * dist * distance_scale

        # Up vector: use Z-axis, fall back to Y if parallel
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(direction_unit, up)) > 0.95:
            up = np.array([0.0, 1.0, 0.0])

        self.look_at(eye, earth_center, up)

    def fixed_inertial(self, sat_pos, earth_center=(0, 0, 0),
                       view_distance: float = 2.5e7):
        """Fixed camera in the inertial frame, looking at Earth center.

        The camera stays at a fixed position in ECI space so that
        Earth's rotation and the satellite's orbital motion are both
        clearly visible. The view distance is chosen so that the full
        orbit and the Earth globe are in frame.
        """
        sat_pos = np.asarray(sat_pos, dtype=np.float64)

        # Fixed position: above the orbital plane on the +Z side,
        # offset slightly in Y so the orbit isn't edge-on.
        eye = np.array([0.0, -view_distance * 0.4, view_distance * 0.9])

        up = np.array([0.0, 0.0, 1.0])
        self.look_at(eye, earth_center, up)

    def _build_perspective(self) -> np.ndarray:
        """Standard row-major perspective projection matrix."""
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        n, fa = self.near, self.far

        m = np.zeros((4, 4), dtype=np.float64)
        m[0, 0] = f / self.aspect
        m[1, 1] = f
        m[2, 2] = -(fa + n) / (fa - n)
        m[2, 3] = -(2.0 * fa * n) / (fa - n)
        m[3, 2] = -1.0
        return m

    def _build_look_at(self) -> np.ndarray:
        """Standard row-major look-at view matrix."""
        forward = self._target - self._eye
        forward_len = np.linalg.norm(forward)
        if forward_len < 1e-12:
            return np.eye(4, dtype=np.float64)
        forward = forward / forward_len

        right = np.cross(forward, self._up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-12:
            fallback = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(forward, fallback)) > 0.99:
                fallback = np.array([1.0, 0.0, 0.0])
            right = np.cross(forward, fallback)
            right_len = np.linalg.norm(right)
        right = right / right_len

        true_up = np.cross(right, forward)

        m = np.eye(4, dtype=np.float64)
        m[0, :3] = right
        m[1, :3] = true_up
        m[2, :3] = -forward
        m[0, 3] = -np.dot(right, self._eye)
        m[1, 3] = -np.dot(true_up, self._eye)
        m[2, 3] = np.dot(forward, self._eye)
        return m
