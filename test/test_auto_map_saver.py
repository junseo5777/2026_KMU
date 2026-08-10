"""Tests for the Nav2-compatible occupancy grid writer."""

import math
from types import SimpleNamespace

from cartographer_2d_slam.map_io import occupancy_grid_to_pgm
from cartographer_2d_slam.map_io import quaternion_to_yaw
from cartographer_2d_slam.map_io import save_occupancy_grid
import pytest


def _map_message():
    orientation = SimpleNamespace(
        x=0.0,
        y=0.0,
        z=math.sin(math.pi / 4.0),
        w=math.cos(math.pi / 4.0),
    )
    origin = SimpleNamespace(
        position=SimpleNamespace(x=1.25, y=-2.5),
        orientation=orientation,
    )
    info = SimpleNamespace(
        width=2,
        height=2,
        resolution=0.05,
        origin=origin,
    )
    return SimpleNamespace(
        info=info,
        data=[0, 100, -1, 50],
    )


def test_pgm_pixels_are_thresholded_and_vertically_flipped():
    content = occupancy_grid_to_pgm(
        width=2,
        height=2,
        data=[0, 100, -1, 50],
    )

    pixels = content.split(b'\n', 4)[-1]
    assert pixels == bytes([205, 205, 254, 0])


def test_quaternion_to_yaw():
    orientation = _map_message().info.origin.orientation
    assert quaternion_to_yaw(orientation) == pytest.approx(math.pi / 2.0)


def test_save_occupancy_grid_writes_nav2_files(tmp_path):
    yaml_path, image_path = save_occupancy_grid(
        _map_message(),
        str(tmp_path / 'parking_map.yaml'),
    )

    yaml_content = yaml_path.read_text(encoding='utf-8')
    assert image_path.is_file()
    assert 'image: "parking_map.pgm"' in yaml_content
    assert 'resolution: 0.05' in yaml_content
    assert 'origin: [1.25, -2.5, 1.57079632679]' in yaml_content
    assert 'occupied_thresh: 0.65' in yaml_content
    assert 'free_thresh: 0.25' in yaml_content
