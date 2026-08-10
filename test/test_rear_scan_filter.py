"""Tests for filtering vehicle-occluded rear LiDAR measurements."""

import math

from cartographer_2d_slam.rear_scan_filter import is_front_angle
from cartographer_2d_slam.rear_scan_filter import mask_rear_ranges
import pytest


def test_front_sector_boundaries_are_kept():
    assert is_front_angle(math.radians(-90.0))
    assert is_front_angle(0.0)
    assert is_front_angle(math.radians(90.0))
    assert not is_front_angle(math.radians(-90.1))
    assert not is_front_angle(math.radians(90.1))
    assert not is_front_angle(math.pi)


def test_rear_half_is_replaced_by_infinity():
    ranges = [1.0, 2.0, 3.0, 4.0, 5.0]
    filtered = mask_rear_ranges(
        angle_min=-math.pi,
        angle_increment=math.pi / 2.0,
        ranges=ranges,
    )

    assert math.isinf(filtered[0])
    assert filtered[1:4] == pytest.approx([2.0, 3.0, 4.0])
    assert math.isinf(filtered[4])
    assert ranges == [1.0, 2.0, 3.0, 4.0, 5.0]
