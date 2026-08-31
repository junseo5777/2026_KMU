#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration


class RvizVisualizer:
    """DWA/라이다 관련 Rviz 마커 생성 및 퍼블리시를 담당하는 모듈"""

    def __init__(self, node, dwa_marker_pub, lidar_marker_pub,
                 marker_frame='laser_frame', marker_lifetime=0.2,
                 roi_half_width=0.7, roi_min_y=0.02, roi_max_y=1.7,
                 regular_roi_half_width=None, regular_roi_min_y=None,
                 regular_roi_max_y=None):
        self.node = node
        self.dwa_marker_pub = dwa_marker_pub
        self.lidar_marker_pub = lidar_marker_pub
        self.marker_frame = marker_frame
        self.marker_lifetime = marker_lifetime
        self.roi_half_width = roi_half_width
        self.roi_min_y = roi_min_y
        self.roi_max_y = roi_max_y
        self.regular_roi_half_width = regular_roi_half_width
        self.regular_roi_min_y = regular_roi_min_y
        self.regular_roi_max_y = regular_roi_max_y

    def make_dwa_marker(self, ns, marker_id, marker_type):
        marker = Marker()
        marker.header.frame_id = self.marker_frame
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(seconds=self.marker_lifetime).to_msg()
        return marker

    @staticmethod
    def set_marker_color(marker, r, g, b, a):
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a

    def path_marker(self, ns, marker_id, path, width, color):
        marker = self.make_dwa_marker(ns, marker_id, Marker.LINE_STRIP)
        marker.scale.x = width
        self.set_marker_color(marker, *color)
        for x, y, _ in path:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            marker.points.append(p)
        return marker

    def lidar_points_marker(self, points, frame_id):
        marker = Marker()
        marker.header.frame_id = frame_id or self.marker_frame
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = 'scan_points'
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(seconds=self.marker_lifetime).to_msg()
        marker.scale.x = 0.04
        marker.scale.y = 0.04
        self.set_marker_color(marker, 0.0, 0.8, 1.0, 0.8)

        for x, y in points:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.02
            marker.points.append(point)
        return marker

    def roi_marker(self, ns, frame_id, half_width, min_y, max_y, color,
                   z=0.03):
        marker = Marker()
        marker.header.frame_id = frame_id or self.marker_frame
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(seconds=self.marker_lifetime).to_msg()
        marker.scale.x = 0.03
        self.set_marker_color(marker, *color)

        corners = (
            (-half_width, min_y),
            (half_width, min_y),
            (half_width, max_y),
            (-half_width, max_y),
            (-half_width, min_y),
        )
        for x, y in corners:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = float(z)
            marker.points.append(point)

        return marker

    def dwa_roi_marker(self, frame_id):
        """Return the yellow boundary used to select DWA obstacles."""
        return self.roi_marker(
            'dwa_roi',
            frame_id,
            self.roi_half_width,
            self.roi_min_y,
            self.roi_max_y,
            (1.0, 1.0, 0.0, 0.95),
            z=0.05,
        )

    def regular_roi_marker(self, frame_id):
        """Return the magenta boundary used for object classification."""
        if any(value is None for value in (
                self.regular_roi_half_width,
                self.regular_roi_min_y,
                self.regular_roi_max_y)):
            return None

        return self.roi_marker(
            'regular_roi',
            frame_id,
            self.regular_roi_half_width,
            self.regular_roi_min_y,
            self.regular_roi_max_y,
            (1.0, 0.1, 0.8, 0.85),
            z=0.04,
        )

    def safe_distance_marker(self, obs, safe_dist, ns, color):
        marker = self.make_dwa_marker(ns, 0, Marker.LINE_LIST)
        marker.scale.x = 0.025
        self.set_marker_color(marker, *color)

        radius = float(safe_dist)
        segments = 32
        circle_angles = np.linspace(0.0, 2.0 * np.pi, segments + 1)

        for ox, oy in obs:
            for idx in range(segments):
                p0 = Point()
                p0.x = float(ox + radius * np.cos(circle_angles[idx]))
                p0.y = float(oy + radius * np.sin(circle_angles[idx]))
                p0.z = 0.03

                p1 = Point()
                p1.x = float(ox + radius * np.cos(circle_angles[idx + 1]))
                p1.y = float(oy + radius * np.sin(circle_angles[idx + 1]))
                p1.z = 0.03

                marker.points.extend((p0, p1))

        return marker

    def publish_lidar_markers(self, points, frame_id):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id or self.marker_frame
        delete_marker.header.stamp = self.node.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        marker_array.markers.append(self.lidar_points_marker(points, frame_id))
        regular_roi = self.regular_roi_marker(frame_id)
        if regular_roi is not None:
            marker_array.markers.append(regular_roi)
        marker_array.markers.append(self.dwa_roi_marker(frame_id))
        self.lidar_marker_pub.publish(marker_array)

    def publish_dwa_visualization(
            self, best_path, all_path, obs,
            end_safe_dist):
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = self.marker_frame
        delete_marker.header.stamp = self.node.get_clock().now().to_msg()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # /dwa_markers 만 RViz에 추가해도 DWA 입력 범위를 함께 볼 수 있다.
        marker_array.markers.append(self.dwa_roi_marker(self.marker_frame))

        for idx, (_, path) in enumerate(all_path):
            marker_array.markers.append(
                self.path_marker(
                    'dwa_candidate_paths',
                    idx,
                    path,
                    0.025,
                    (0.1, 0.45, 1.0, 0.35),
                )
            )

        if best_path:
            marker_array.markers.append(
                self.path_marker(
                    'dwa_selected_path',
                    0,
                    best_path,
                    0.04,
                    (0.0, 1.0, 0.2, 0.95),
                )
            )

        if obs is not None and len(obs) > 0:
            marker_array.markers.append(
                self.safe_distance_marker(
                    obs,
                    end_safe_dist,
                    'dwa_end_safe_distance',
                    (1.0, 0.55, 0.0, 0.75),
                )
            )
        self.dwa_marker_pub.publish(marker_array)
