"""Scientific-integrity checks for Earth visualization geometry."""

import hashlib
from pathlib import Path

import brahe as bh
import numpy as np
import pytest

from core.constants import R_EARTH, R_EARTH_POLAR
from render.earth import Earth
from render.ephemeris import scene_ephemeris
from render.renderer import Renderer

ASSET_HASHES = {
    "earth_day.jpg": "a9f0088972dee0254610af851c4d6838ca3f2cf79176987e0a5713e2c15ec042",
    "earth_night.jpg": "373e5a08c9f378a2ce6320214a613148e4b1e3946b3f39a516c9093b76cb7124",
}


def _earth_without_gpu() -> Earth:
    earth = Earth.__new__(Earth)
    earth.radius = R_EARTH
    earth.polar_radius = R_EARTH_POLAR
    return earth


def test_earth_mesh_is_wgs84_and_texture_longitude_is_geographic():
    lat_divs, lon_divs = 4, 8
    vertices, _ = _earth_without_gpu()._generate_ellipsoid(lat_divs, lon_divs)
    vertices = vertices.reshape(lat_divs + 1, lon_divs + 1, 8)

    north_pole = vertices[0, lon_divs // 2]
    anti_meridian = vertices[lat_divs // 2, 0]
    greenwich = vertices[lat_divs // 2, lon_divs // 2]

    assert np.allclose(north_pole[:3], [0.0, 0.0, R_EARTH_POLAR], atol=1.0)
    assert np.allclose(anti_meridian[:3], [-R_EARTH, 0.0, 0.0], atol=1.0)
    assert np.allclose(greenwich[:3], [R_EARTH, 0.0, 0.0], atol=1.0)
    assert greenwich[6] == 0.5

    positions = vertices[..., :3].reshape(-1, 3)
    ellipsoid = (
        (positions[:, 0] ** 2 + positions[:, 1] ** 2) / R_EARTH**2
        + positions[:, 2] ** 2 / R_EARTH_POLAR**2
    )
    assert np.allclose(ellipsoid, 1.0, atol=2e-7)
    assert np.allclose(np.linalg.norm(vertices[..., 3:6], axis=-1), 1.0, atol=1e-6)


def test_model_matrix_embeds_full_earth_orientation():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    model = Earth.get_model_matrix(rotation)

    assert np.allclose(model[:3, :3], rotation)
    assert np.allclose(model[3], [0.0, 0.0, 0.0, 1.0])


def test_scene_ephemeris_is_orthogonal_and_uses_physical_solar_distance():
    scene = scene_ephemeris(2460390.0)

    assert np.allclose(
        scene.itrf_to_gcrf.T @ scene.itrf_to_gcrf, np.eye(3), atol=1e-12
    )
    assert np.isclose(np.linalg.det(scene.itrf_to_gcrf), 1.0, atol=1e-12)
    assert 1.45e11 < np.linalg.norm(scene.sun_position_gcrf_m) < 1.52e11

    epoch = bh.Epoch.from_jd(2460390.0, bh.TimeSystem.UTC)
    greenwich = np.array([R_EARTH, 0.0, 0.0])
    assert np.allclose(
        scene.itrf_to_gcrf @ greenwich,
        bh.position_itrf_to_gcrf(epoch, greenwich),
        atol=1e-8,
    )


def test_grounded_shader_has_no_artificial_daylight_or_rim_glow():
    shader = (
        Path(__file__).parents[1] / "render" / "shaders" / "earth.frag"
    ).read_text(encoding="utf-8")

    assert "day_col * 0.35" not in shader
    assert "atmosphere" not in shader
    assert "solar_incidence" in shader


def test_documented_visual_assets_are_immutable():
    assets = Path(__file__).parents[1] / "assets"
    for name, expected in ASSET_HASHES.items():
        assert hashlib.sha256((assets / name).read_bytes()).hexdigest() == expected


def test_renderer_rejects_unknown_constellation_target():
    renderer = Renderer.__new__(Renderer)
    with pytest.raises(ValueError, match="target_index"):
        renderer.render_frame(
            np.zeros((2, 3)), np.ones(3), np.eye(3), target_index=2
        )


def test_renderer_draws_each_constellation_trail(monkeypatch):
    renderer = Renderer.__new__(Renderer)
    trails_drawn = []
    satellites_drawn = []
    monkeypatch.setattr(renderer, "_draw_earth", lambda *args: None)
    monkeypatch.setattr(
        renderer, "_draw_trail", lambda trail, vp: trails_drawn.append(trail)
    )
    monkeypatch.setattr(
        renderer, "_draw_satellite", lambda pos, vp: satellites_drawn.append(pos)
    )
    trails = [np.zeros((2, 3)), np.ones((2, 3))]
    satellites = np.zeros((2, 3))
    renderer._render_scene(
        satellites, np.ones(3), np.eye(3), trails, np.eye(4)
    )
    assert len(trails_drawn) == 2
    assert len(satellites_drawn) == 2
