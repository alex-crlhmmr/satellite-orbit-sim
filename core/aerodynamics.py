"""Attitude-aware projected-area models for free-molecular LEO drag.

These models compute geometry only.  They deliberately do not claim to model
surface chemistry, accommodation, or a physical drag coefficient; those
remain explicit inputs to the force model and its validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def lvlh_body_to_eci(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """Return body-to-ECI DCM for +X along-track, +Z nadir LVLH attitude."""
    z_body = -r_eci / np.linalg.norm(r_eci)
    orbit_normal = np.cross(r_eci, v_eci)
    orbit_normal /= np.linalg.norm(orbit_normal)
    x_body = np.cross(z_body, orbit_normal)
    x_body /= np.linalg.norm(x_body)
    y_body = np.cross(z_body, x_body)
    return np.column_stack((x_body, y_body, z_body))


@dataclass(frozen=True)
class BoxWingGeometry:
    """Convex box plus optional two-sided flat panels.

    ``box_dimensions_m`` are lengths along body X/Y/Z. Each panel entry has a
    body-frame unit normal and total exposed area. Shadowing and self-occlusion
    are outside this deliberately auditable first-order model.
    """

    box_dimensions_m: np.ndarray
    panel_normals_body: np.ndarray
    panel_areas_m2: np.ndarray

    @classmethod
    def from_config(cls, config: dict) -> "BoxWingGeometry":
        dimensions = np.asarray(config["box_dimensions_m"], dtype=np.float64)
        if dimensions.shape != (3,) or np.any(dimensions <= 0):
            raise ValueError("box_dimensions_m must contain three positive lengths")
        panels = config.get("panels", [])
        normals = np.asarray([item["normal_body"] for item in panels], dtype=np.float64)
        areas = np.asarray([item["area_m2"] for item in panels], dtype=np.float64)
        if not panels:
            normals = np.empty((0, 3), dtype=np.float64)
            areas = np.empty(0, dtype=np.float64)
        if normals.shape != (len(panels), 3) or np.any(areas < 0):
            raise ValueError("invalid panel geometry")
        if len(normals):
            lengths = np.linalg.norm(normals, axis=1)
            if np.any(lengths == 0):
                raise ValueError("panel normals must be nonzero")
            normals = normals / lengths[:, None]
        return cls(dimensions, normals, areas)

    def projected_area(self, flow_direction_body: np.ndarray) -> float:
        """Orthographic area normal to a unit relative-flow direction [m²]."""
        direction = np.asarray(flow_direction_body, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("flow direction must be nonzero")
        ux, uy, uz = np.abs(direction / norm)
        lx, ly, lz = self.box_dimensions_m
        box = ly * lz * ux + lx * lz * uy + lx * ly * uz
        panels = float(np.sum(self.panel_areas_m2 *
                              np.abs(self.panel_normals_body @ (direction / norm))))
        return float(box + panels)

    def area_mass_ratio_lvlh(self, r_eci: np.ndarray, v_eci: np.ndarray,
                             relative_velocity_eci: np.ndarray,
                             mass_kg: float) -> float:
        if mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        body_to_eci = lvlh_body_to_eci(r_eci, v_eci)
        flow_body = body_to_eci.T @ relative_velocity_eci
        return self.projected_area(flow_body) / mass_kg
