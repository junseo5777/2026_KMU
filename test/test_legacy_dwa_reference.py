"""Tests for the selectable legacy DWA reference point."""

import math

from kmu.dwa import DWA

import numpy as np

import pytest


def _path_for_steer(paths, target):
    return min(paths, key=lambda item: abs(item[0] - target))[1]


def test_default_dwa_keeps_the_nuclear_lidar_origin():
    """DWA() without track-drive geometry retains its previous origin."""
    dwa = DWA()
    straight = _path_for_steer(dwa.predict_path(0.0), 0.0)

    assert straight[0] == pytest.approx((0.0, 0.0, math.pi / 2))


def test_track_drive_path_starts_at_rear_axle_behind_lidar():
    """Track-drive geometry places the rear axle 0.410 m behind the scan."""
    dwa = DWA(wheel_base=0.345, sensor_forward_offset=0.410)
    straight = _path_for_steer(dwa.predict_path(0.0), 0.0)

    assert straight[0] == pytest.approx((0.0, -0.410, math.pi / 2))
    assert straight[-1] == pytest.approx(
        (0.0, -0.410 + 5 * dwa.step, math.pi / 2)
    )


def test_obstacle_cost_uses_the_same_rear_axle_points_shown_in_rviz():
    """Cost checks use rear-axle path points without LiDAR projection."""
    dwa = DWA(wheel_base=0.345, sensor_forward_offset=0.410)
    expected = [
        (
            path[idx],
            dwa.middle_safe_dist if idx == 2 else dwa.end_safe_dist,
        )
        for _, path in dwa.predict_path(0.0)
        for idx in (2, 5)
    ]
    checked = []

    def record_cost(point, yaw, obs, obs_weight, safe_dist):
        checked.append((point, safe_dist))
        return 0.0

    dwa.cost_function = record_cost
    dwa.dwa(0.0, None)

    assert checked == expected


def test_middle_and_end_points_use_independent_safe_distances():
    """Middle and end thresholds remain independently configurable."""
    dwa = DWA()
    point = (0.0, 0.0, math.pi / 2)
    probe_distance = (dwa.middle_safe_dist + dwa.end_safe_dist) / 2.0
    obstacle = np.asarray([(probe_distance, 0.0)], dtype=float)

    middle_cost = dwa.cost_function(
        point, point[2], obstacle, dwa.w2, dwa.middle_safe_dist)
    end_cost = dwa.cost_function(
        point, point[2], obstacle, dwa.w3, dwa.end_safe_dist)

    assert dwa.middle_safe_dist < probe_distance < dwa.end_safe_dist
    assert middle_cost == pytest.approx(0.0)
    assert end_cost > 0.0


def test_optimized_cost_matches_the_original_full_obstacle_scan():
    """Spatial prefiltering must preserve the original cost equation."""
    dwa = DWA(wheel_base=0.345, sensor_forward_offset=0.410)
    rng = np.random.default_rng(2026)
    obstacles = rng.uniform((-0.7, 0.05), (0.7, 1.05), size=(1200, 2))

    def original_cost(point, yaw, obs, obs_weight, safe_dist):
        p = np.array(point[:2])
        distances = np.linalg.norm(np.array(obs) - p, axis=1)
        dis_min = min(distances)
        obs_cost = 1.0 / dis_min if dis_min < safe_dist else 0.0
        heading_cost = abs(yaw - math.pi / 2)
        return dwa.w1 * heading_cost + obs_weight * obs_cost

    for _, path in dwa.predict_path(0.0):
        for idx, weight, safe_dist in (
                (2, dwa.w2, dwa.middle_safe_dist),
                (5, dwa.w3, dwa.end_safe_dist)):
            actual = dwa.cost_function(
                path[idx], path[idx][2], obstacles, weight, safe_dist)
            expected = original_cost(
                path[idx], path[idx][2], obstacles, weight, safe_dist)
            assert actual == pytest.approx(expected, rel=1e-15, abs=1e-15)


def test_rear_axle_turns_are_mirrored_about_vehicle_centerline():
    """Equal left and right steering produce mirrored rear-axle paths."""
    dwa = DWA(wheel_base=0.345, sensor_forward_offset=0.410)
    paths = dwa.predict_path(0.0)
    left = _path_for_steer(paths, dwa.max_steer)[-1]
    right = _path_for_steer(paths, -dwa.max_steer)[-1]

    assert left[0] == pytest.approx(-right[0])
    assert left[1] == pytest.approx(right[1])
    assert left[2] - math.pi / 2 == pytest.approx(
        -(right[2] - math.pi / 2)
    )
