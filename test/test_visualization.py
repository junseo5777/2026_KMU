"""Tests for the legacy track-drive RViz visualization."""

from builtin_interfaces.msg import Time

from kmu.visualization import RvizVisualizer

import pytest

from visualization_msgs.msg import Marker


class _Now:

    def to_msg(self):
        return Time()


class _Clock:

    def now(self):
        return _Now()


class _Node:

    def get_clock(self):
        return _Clock()


class _Publisher:

    def __init__(self):
        self.message = None

    def publish(self, message):
        self.message = message


def _visualizer():
    return RvizVisualizer(
        _Node(),
        _Publisher(),
        _Publisher(),
        roi_half_width=0.7,
        roi_min_y=0.02,
        roi_max_y=1.3,
        regular_roi_half_width=0.7,
        regular_roi_min_y=0.02,
        regular_roi_max_y=2.5,
    )


def _xy(marker):
    return [(point.x, point.y) for point in marker.points]


def test_roi_markers_match_track_drive_filter_bounds():
    """ROI rectangles use the exact filtering bounds from track_drive."""
    visualizer = _visualizer()

    dwa_roi = visualizer.dwa_roi_marker('laser_frame')
    regular_roi = visualizer.regular_roi_marker('laser_frame')

    assert dwa_roi.type == Marker.LINE_STRIP
    assert dwa_roi.ns == 'dwa_roi'
    assert _xy(dwa_roi) == pytest.approx([
        (-0.7, 0.02),
        (0.7, 0.02),
        (0.7, 1.3),
        (-0.7, 1.3),
        (-0.7, 0.02),
    ])
    assert regular_roi.ns == 'regular_roi'
    assert _xy(regular_roi) == pytest.approx([
        (-0.7, 0.02),
        (0.7, 0.02),
        (0.7, 2.5),
        (-0.7, 2.5),
        (-0.7, 0.02),
    ])


def test_lidar_markers_contain_both_roi_boundaries():
    """The LiDAR topic shows DWA and classification ROI boundaries."""
    visualizer = _visualizer()

    visualizer.publish_lidar_markers([], 'laser_frame')

    namespaces = {
        marker.ns for marker in visualizer.lidar_marker_pub.message.markers
    }
    assert {'scan_points', 'dwa_roi', 'regular_roi'} <= namespaces


def test_dwa_topic_also_contains_dwa_roi_boundary():
    """The DWA topic remains useful without the LiDAR marker display."""
    visualizer = _visualizer()

    visualizer.publish_dwa_visualization(
        None,
        [],
        None,
        end_safe_dist=0.33,
    )

    namespaces = {
        marker.ns for marker in visualizer.dwa_marker_pub.message.markers
    }
    assert 'dwa_roi' in namespaces


def test_dwa_topic_only_draws_end_safe_distance():
    visualizer = _visualizer()

    visualizer.publish_dwa_visualization(
        None,
        [],
        [(0.0, 0.5)],
        end_safe_dist=0.34,
    )

    namespaces = {
        marker.ns for marker in visualizer.dwa_marker_pub.message.markers
    }
    assert 'dwa_end_safe_distance' in namespaces
    assert 'dwa_middle_safe_distance' not in namespaces
