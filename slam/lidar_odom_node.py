#!/usr/bin/env python3

# Copyright 2026 junseo5777 and kimmc53
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Publish odom->base_link using IMU-assisted scan-to-scan ICP."""

from collections import deque
import math
import time

from diagnostic_msgs.msg import DiagnosticArray
from diagnostic_msgs.msg import DiagnosticStatus
from diagnostic_msgs.msg import KeyValue
from geometry_msgs.msg import TransformStamped

from cartographer_2d_slam.lidar_odometry import adaptive_imu_yaw_weight
from cartographer_2d_slam.lidar_odometry import constrain_nonholonomic
from cartographer_2d_slam.lidar_odometry import deskew_points
from cartographer_2d_slam.lidar_odometry import estimate_normals
from cartographer_2d_slam.lidar_odometry import fuse_relative_yaw
from cartographer_2d_slam.lidar_odometry import hybrid_icp
from cartographer_2d_slam.lidar_odometry import icp_match
from cartographer_2d_slam.lidar_odometry import matrix_pose
from cartographer_2d_slam.lidar_odometry import normalize_angle
from cartographer_2d_slam.lidar_odometry import point_to_line_icp
from cartographer_2d_slam.lidar_odometry import pose_matrix
from cartographer_2d_slam.lidar_odometry import SubmapIndex
from cartographer_2d_slam.lidar_odometry import transform_points
from cartographer_2d_slam.lidar_odometry import voxel_downsample
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from tf2_ros import TransformBroadcaster


def _stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1.0e-9


def _yaw_quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class LidarOdomNode(Node):
    """Estimate locally continuous planar odometry from LiDAR and gyro."""

    def __init__(self):
        """Create subscriptions, publishers, and the rolling ICP state."""
        super().__init__('lidar_odom_node')
        self._declare_parameters()
        self._read_parameters()

        self._tf_broadcaster = TransformBroadcaster(self)
        self._odom_publisher = self.create_publisher(
            Odometry, self.odom_topic, 10
        )
        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        if self.use_imu:
            self.create_subscription(
                Imu,
                self.imu_topic,
                self._imu_callback,
                qos_profile_sensor_data,
            )
        self._diagnostics_publisher = self.create_publisher(
            DiagnosticArray,
            '/localization_diagnostics',
            10,
        )

        self._base_to_laser = pose_matrix(
            self.lidar_x, self.lidar_y, self.lidar_yaw
        )
        self._laser_to_base = np.linalg.inv(self._base_to_laser)
        self._pose = np.eye(3, dtype=np.float64)
        self._last_keyframe_pose = self._pose.copy()
        self._keyframes = deque(maxlen=self.submap_keyframes)
        self._submap = np.empty((0, 2), dtype=np.float64)
        self._submap_index = None
        self._previous_scan_index = None
        self._initialized = False
        self._last_scan_time = None
        self._last_scan_imu_yaw = 0.0
        self._failure_count = 0
        self._fallback_count = 0
        self._recovery_count = 0
        self._wrong_frame_logged = False
        self._wrong_imu_frame_logged = False
        self._body_velocity_x = 0.0
        self._body_velocity_y = 0.0
        self._body_yaw_rate = 0.0

        self._imu_last_time = None
        self._imu_yaw = 0.0
        self._imu_yaw_rate = 0.0
        self._gyro_samples = []
        self._gyro_calibration_start = None
        self._gyro_calibrated = self.gyro_calibration_duration <= 0.0

        self._statistics_start = time.monotonic()
        self._scan_count = 0
        self._accepted_count = 0
        self._rejected_count = 0
        self._stationary_count = 0
        self._dropped_count = 0
        self._imu_count = 0
        self._processing_total_ms = 0.0
        self._processing_max_ms = 0.0
        self._last_point_count = 0
        self._last_quality = None
        self._last_lateral_rejection = 0.0
        self._last_imu_fusion_weight = 0.0
        self._imu_fused_count = 0
        self._last_scan_stamp = None
        self._last_imu_stamp = None
        if self.diagnostics_period_sec > 0.0:
            self.create_timer(
                self.diagnostics_period_sec,
                self._publish_diagnostics,
            )

        imu_text = (
            f'{self.imu_topic}(yaw_sign={self.imu_yaw_sign:+.1f})'
            if self.use_imu else 'disabled'
        )
        if self.use_hybrid_icp:
            matcher_text = 'hybrid'
        elif self.use_point_to_line:
            matcher_text = 'point_to_line'
        else:
            matcher_text = 'point_to_point'
        self.get_logger().info(
            f'LiDAR odometry ready: scan={self.scan_topic}, imu={imu_text}, '
            f'matcher={matcher_text}, '
            f'{self.odom_frame_id}->{self.base_frame_id}'
        )
        if self.use_imu and not self._gyro_calibrated:
            self.get_logger().info(
                'Keep the vehicle still for '
                f'{self.gyro_calibration_duration:.1f}s to calibrate gyro z.'
            )

    def _declare_parameters(self):
        self.declare_parameter('scan_topic', '/scan_slam')
        self.declare_parameter('imu_topic', '/imu')
        self.declare_parameter('odom_topic', '/lidar_odom')
        self.declare_parameter('odom_frame_id', 'odom')
        self.declare_parameter('base_frame_id', 'base_link')
        self.declare_parameter('laser_frame_id', 'laser_frame')
        self.declare_parameter('imu_frame_id', 'base_imu_link')
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('use_imu', True)
        self.declare_parameter('imu_yaw_sign', -1.0)
        self.declare_parameter('gyro_bias_z', 0.0)
        self.declare_parameter('gyro_calibration_duration', 2.0)
        self.declare_parameter('imu_yaw_weight', 0.30)
        self.declare_parameter('imu_timeout_sec', 0.20)
        self.declare_parameter('imu_yaw_fusion_weight', 0.30)
        self.declare_parameter('imu_yaw_low_quality_weight', 0.70)
        self.declare_parameter('imu_yaw_good_inlier_ratio', 0.65)
        self.declare_parameter('imu_yaw_good_rmse', 0.08)
        self.declare_parameter('deskew_scan', True)
        self.declare_parameter('lidar_x', 0.410)
        self.declare_parameter('lidar_y', 0.0)
        self.declare_parameter('lidar_yaw', 0.0)
        self.declare_parameter('min_range', 0.15)
        self.declare_parameter('max_range', 20.0)
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('max_scan_points', 400)
        self.declare_parameter('max_submap_points', 2400)
        self.declare_parameter('submap_keyframes', 15)
        self.declare_parameter('keyframe_translation', 0.15)
        self.declare_parameter('keyframe_rotation', 0.08)
        self.declare_parameter('icp_max_iterations', 20)
        self.declare_parameter('max_correspondence_distance', 0.35)
        self.declare_parameter('trim_fraction', 0.80)
        self.declare_parameter('min_correspondences', 30)
        self.declare_parameter('min_inlier_ratio', 0.30)
        self.declare_parameter('max_rmse', 0.15)
        self.declare_parameter('stationary_translation_deadband', 0.004)
        self.declare_parameter('stationary_rotation_deadband', 0.003)
        self.declare_parameter('enforce_nonholonomic', True)
        self.declare_parameter('use_hybrid_icp', False)
        self.declare_parameter('hybrid_max_iterations', 12)
        self.declare_parameter('hybrid_line_weight', 1.0)
        self.declare_parameter('hybrid_point_weight', 1.0)
        self.declare_parameter('hybrid_min_group_correspondences', 5)
        self.declare_parameter('hybrid_min_rank', 3)
        self.declare_parameter('use_point_to_line', False)
        self.declare_parameter('point_to_line_max_iterations', 12)
        self.declare_parameter('max_point_to_line_rmse', 0.08)
        self.declare_parameter('normal_neighbors', 8)
        self.declare_parameter('normal_search_radius', 0.40)
        self.declare_parameter('normal_min_neighbors', 5)
        self.declare_parameter('normal_min_linearity', 0.80)
        self.declare_parameter(
            'normal_max_angle', math.radians(45.0)
        )
        self.declare_parameter('point_to_line_svd_rcond', 1.0e-3)
        self.declare_parameter(
            'point_to_line_max_condition_number', 1.0e4
        )
        self.declare_parameter('fallback_to_point_to_point', True)
        self.declare_parameter('max_prediction_correction', 0.30)
        self.declare_parameter(
            'max_prediction_rotation_correction', 0.20
        )
        self.declare_parameter('recovery_after_failures', 3)
        self.declare_parameter('max_translation_per_scan', 0.80)
        self.declare_parameter('max_rotation_per_scan', 0.50)
        self.declare_parameter('diagnostics_period_sec', 5.0)

    def _read_parameters(self):
        names = [
            'scan_topic', 'imu_topic', 'odom_topic', 'odom_frame_id',
            'base_frame_id', 'laser_frame_id', 'imu_frame_id',
            'publish_tf', 'use_imu',
            'imu_yaw_sign', 'gyro_bias_z',
            'gyro_calibration_duration', 'imu_yaw_weight',
            'imu_timeout_sec', 'imu_yaw_fusion_weight',
            'imu_yaw_low_quality_weight',
            'imu_yaw_good_inlier_ratio', 'imu_yaw_good_rmse', 'lidar_x',
            'lidar_y', 'lidar_yaw', 'deskew_scan', 'min_range', 'max_range',
            'voxel_size', 'max_scan_points', 'max_submap_points',
            'submap_keyframes',
            'keyframe_translation', 'keyframe_rotation',
            'icp_max_iterations', 'max_correspondence_distance',
            'trim_fraction', 'min_correspondences', 'min_inlier_ratio',
            'max_rmse', 'stationary_translation_deadband',
            'stationary_rotation_deadband',
            'enforce_nonholonomic',
            'use_hybrid_icp', 'hybrid_max_iterations',
            'hybrid_line_weight', 'hybrid_point_weight',
            'hybrid_min_group_correspondences', 'hybrid_min_rank',
            'use_point_to_line',
            'point_to_line_max_iterations', 'max_point_to_line_rmse',
            'normal_neighbors', 'normal_search_radius',
            'normal_min_neighbors', 'normal_min_linearity',
            'normal_max_angle', 'point_to_line_svd_rcond',
            'point_to_line_max_condition_number',
            'fallback_to_point_to_point', 'max_prediction_correction',
            'max_prediction_rotation_correction',
            'recovery_after_failures',
            'max_translation_per_scan',
            'max_rotation_per_scan',
            'diagnostics_period_sec',
        ]
        for name in names:
            setattr(self, name, self.get_parameter(name).value)

        if not 0.0 < self.trim_fraction <= 1.0:
            raise ValueError('trim_fraction must be in (0, 1]')
        if self.submap_keyframes < 1:
            raise ValueError('submap_keyframes must be positive')
        if self.min_correspondences < 2:
            raise ValueError('min_correspondences must be at least two')
        if self.stationary_translation_deadband < 0.0:
            raise ValueError(
                'stationary_translation_deadband must be non-negative'
            )
        if self.stationary_rotation_deadband < 0.0:
            raise ValueError(
                'stationary_rotation_deadband must be non-negative'
            )
        if not 0.0 <= self.imu_yaw_weight <= 1.0:
            raise ValueError('imu_yaw_weight must be in [0, 1]')
        if not math.isclose(abs(self.imu_yaw_sign), 1.0):
            raise ValueError('imu_yaw_sign must be either -1.0 or 1.0')
        if self.imu_timeout_sec <= 0.0:
            raise ValueError('imu_timeout_sec must be positive')
        if not (
                0.0 <= self.imu_yaw_fusion_weight
                <= self.imu_yaw_low_quality_weight <= 1.0):
            raise ValueError(
                'IMU fusion weights must satisfy '
                '0 <= nominal <= low_quality <= 1'
            )
        if self.imu_yaw_good_inlier_ratio <= self.min_inlier_ratio:
            raise ValueError(
                'imu_yaw_good_inlier_ratio must exceed min_inlier_ratio'
            )
        if self.imu_yaw_good_rmse >= self.max_rmse:
            raise ValueError('imu_yaw_good_rmse must be less than max_rmse')
        if self.hybrid_max_iterations < 1:
            raise ValueError('hybrid_max_iterations must be positive')
        if (
                self.hybrid_line_weight <= 0.0
                or self.hybrid_point_weight <= 0.0):
            raise ValueError('hybrid residual weights must be positive')
        if self.hybrid_min_group_correspondences < 2:
            raise ValueError(
                'hybrid_min_group_correspondences must be at least two'
            )
        if not 2 <= self.hybrid_min_rank <= 3:
            raise ValueError('hybrid_min_rank must be two or three')
        if self.normal_neighbors < 2:
            raise ValueError('normal_neighbors must be at least two')
        if not 2 <= self.normal_min_neighbors <= self.normal_neighbors:
            raise ValueError(
                'normal_min_neighbors must be in [2, normal_neighbors]'
            )
        if self.normal_search_radius <= 0.0:
            raise ValueError('normal_search_radius must be positive')
        if not 0.0 <= self.normal_min_linearity <= 1.0:
            raise ValueError('normal_min_linearity must be in [0, 1]')
        if not 0.0 <= self.normal_max_angle <= math.pi / 2.0:
            raise ValueError('normal_max_angle must be in [0, pi/2]')
        if self.point_to_line_max_iterations < 1:
            raise ValueError(
                'point_to_line_max_iterations must be positive'
            )
        if self.max_point_to_line_rmse <= 0.0:
            raise ValueError('max_point_to_line_rmse must be positive')
        if not 0.0 < self.point_to_line_svd_rcond < 1.0:
            raise ValueError('point_to_line_svd_rcond must be in (0, 1)')
        if self.point_to_line_max_condition_number < 1.0:
            raise ValueError(
                'point_to_line_max_condition_number must be at least one'
            )
        if self.max_prediction_correction <= 0.0:
            raise ValueError('max_prediction_correction must be positive')
        if self.max_prediction_rotation_correction <= 0.0:
            raise ValueError(
                'max_prediction_rotation_correction must be positive'
            )
        if self.recovery_after_failures < 1:
            raise ValueError('recovery_after_failures must be positive')
        if self.diagnostics_period_sec < 0.0:
            raise ValueError('diagnostics_period_sec must be non-negative')

    def _imu_callback(self, message):
        received_frame = message.header.frame_id.lstrip('/')
        expected_frame = self.imu_frame_id.lstrip('/')
        if expected_frame and received_frame != expected_frame:
            if not self._wrong_imu_frame_logged:
                self.get_logger().error(
                    f'Ignoring IMU in frame {message.header.frame_id!r}; '
                    f'expected {self.imu_frame_id!r}.'
                )
                self._wrong_imu_frame_logged = True
            return
        sample_time = _stamp_seconds(message.header.stamp)
        self._imu_count += 1
        self._last_imu_stamp = sample_time
        if self._imu_last_time is None:
            self._imu_last_time = sample_time
            self._gyro_calibration_start = sample_time
            return

        delta_time = sample_time - self._imu_last_time
        self._imu_last_time = sample_time
        if delta_time <= 0.0 or delta_time > 0.5:
            return

        # The live vehicle test showed that the driver publishes clockwise
        # rotation as positive. Convert it to ROS +z/CCW before calibration,
        # integration, scan prediction, yaw fusion, and scan deskew.
        signed_yaw_rate = self.imu_yaw_sign * message.angular_velocity.z

        if not self._gyro_calibrated:
            self._gyro_samples.append(signed_yaw_rate)
            elapsed = sample_time - self._gyro_calibration_start
            if elapsed >= self.gyro_calibration_duration:
                measured_bias = sum(self._gyro_samples) / len(
                    self._gyro_samples
                )
                self.gyro_bias_z += measured_bias
                self._gyro_calibrated = True
                self._imu_yaw = 0.0
                self._last_scan_imu_yaw = 0.0
                self.get_logger().info(
                    f'Gyro calibration complete: bias_z='
                    f'{self.gyro_bias_z:.6f} rad/s'
                )
            return

        yaw_rate = signed_yaw_rate - self.gyro_bias_z
        self._imu_yaw_rate = yaw_rate
        self._imu_yaw += yaw_rate * delta_time

    def _scan_points(self, message):
        ranges = np.asarray(message.ranges, dtype=np.float64)
        angles = (
            message.angle_min
            + np.arange(len(ranges), dtype=np.float64)
            * message.angle_increment
        )
        lower = max(float(message.range_min), self.min_range)
        sensor_upper = float(message.range_max)
        upper = min(sensor_upper, self.max_range)
        valid = np.isfinite(ranges) & (ranges >= lower) & (ranges <= upper)
        beam_indices = np.arange(len(ranges), dtype=np.float64)[valid]
        ranges = ranges[valid]
        angles = angles[valid]
        points = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles),
        ))
        if self.deskew_scan:
            points = self._deskew_points(points, beam_indices, message)
        return voxel_downsample(
            points, self.voxel_size, self.max_scan_points
        )

    def _deskew_points(self, points, beam_indices, message):
        """Express moving-scan beams in the laser frame of the first ray."""
        time_increment = float(message.time_increment)
        if time_increment <= 0.0 and len(message.ranges) > 1:
            time_increment = float(message.scan_time) / len(message.ranges)
        if time_increment <= 0.0:
            return points

        offsets = beam_indices * time_increment
        yaw_rate = (
            self._imu_yaw_rate
            if self.use_imu and self._gyro_calibrated
            else self._body_yaw_rate
        )
        return deskew_points(
            points,
            offsets,
            self._base_to_laser,
            self._body_velocity_x,
            self._body_velocity_y,
            yaw_rate,
        )

    def _imu_delta_for_scan(self, scan_time):
        """Return fresh bias-corrected gyro rotation since the last scan."""
        if (
                not self.use_imu
                or not self._gyro_calibrated
                or self._last_imu_stamp is None
                or abs(scan_time - self._last_imu_stamp)
                > self.imu_timeout_sec):
            return None
        return normalize_angle(self._imu_yaw - self._last_scan_imu_yaw)

    def _predict_pose(self, delta_time, imu_delta):
        """Predict from body velocity and a fresh relative gyro rotation."""
        predicted_yaw_delta = self._body_yaw_rate * delta_time
        if imu_delta is not None:
            predicted_yaw_delta = normalize_angle(
                predicted_yaw_delta
                + self.imu_yaw_weight
                * normalize_angle(imu_delta - predicted_yaw_delta)
            )
        local_motion = pose_matrix(
            self._body_velocity_x * delta_time,
            self._body_velocity_y * delta_time,
            predicted_yaw_delta,
        )
        return self._pose @ local_motion

    def _fuse_matched_yaw(
            self, old_pose, matched_base_pose, quality, imu_delta):
        """Keep ICP translation while adaptively anchoring yaw to the gyro."""
        if imu_delta is None:
            self._last_imu_fusion_weight = 0.0
            return matched_base_pose

        relative_motion = np.linalg.inv(old_pose) @ matched_base_pose
        dx, dy, lidar_yaw_delta = matrix_pose(relative_motion)
        fusion_weight = adaptive_imu_yaw_weight(
            quality.get('inlier_ratio', 0.0),
            quality.get('rmse', math.inf),
            self.min_inlier_ratio,
            self.imu_yaw_good_inlier_ratio,
            self.imu_yaw_good_rmse,
            self.max_rmse,
            self.imu_yaw_fusion_weight,
            self.imu_yaw_low_quality_weight,
        )
        fused_yaw_delta = fuse_relative_yaw(
            lidar_yaw_delta,
            imu_delta,
            fusion_weight,
        )
        quality['lidar_yaw_delta'] = lidar_yaw_delta
        quality['imu_yaw_delta'] = imu_delta
        quality['fused_yaw_delta'] = fused_yaw_delta
        quality['imu_yaw_fusion_weight'] = fusion_weight
        self._last_imu_fusion_weight = fusion_weight
        self._imu_fused_count += 1
        return old_pose @ pose_matrix(dx, dy, fused_yaw_delta)

    def _evaluate_match(
            self, matched_laser_pose, quality, initial_laser_pose):
        """Apply geometry, residual, and prediction-correction gates."""
        matched_base_pose = matched_laser_pose @ self._laser_to_base
        motion = np.linalg.inv(self._pose) @ matched_base_pose
        dx, dy, dyaw = matrix_pose(motion)
        plausible_motion = (
            math.hypot(dx, dy) <= self.max_translation_per_scan
            and abs(dyaw) <= self.max_rotation_per_scan
        )

        correction = np.linalg.inv(initial_laser_pose) @ matched_laser_pose
        correction_x, correction_y, correction_yaw = matrix_pose(correction)
        plausible_correction = (
            math.hypot(correction_x, correction_y)
            <= self.max_prediction_correction
            and abs(correction_yaw)
            <= self.max_prediction_rotation_correction
        )

        accepted = (
            quality.get('valid', False)
            and quality['correspondences'] >= self.min_correspondences
            and quality['inlier_ratio'] >= self.min_inlier_ratio
            and quality['rmse'] <= self.max_rmse
            and plausible_motion
            and plausible_correction
        )
        if quality['method'] in {'hybrid', 'point_to_line'}:
            required_rank = (
                self.hybrid_min_rank
                if quality['method'] == 'hybrid'
                else 2
            )
            line_residual_ok = (
                not quality.get('line_group_active', True)
                or quality['point_to_line_rmse']
                <= self.max_point_to_line_rmse
            )
            accepted = (
                accepted
                and quality['rank'] >= required_rank
                and line_residual_ok
                and quality['condition_number']
                <= self.point_to_line_max_condition_number
            )
        return matched_base_pose, accepted

    @staticmethod
    def _quality_text(quality):
        """Format compact scan-matcher diagnostics for throttled logs."""
        if quality is None:
            return 'no primary quality'
        method = quality['method']
        inlier_ratio = quality['inlier_ratio']
        rmse = quality['rmse']
        text = (
            f'method={method}, '
            f'inliers={inlier_ratio:.2f}, '
            f'rmse={rmse:.3f} m'
        )
        if quality['method'] in {'hybrid', 'point_to_line'}:
            line_rmse = quality['point_to_line_rmse']
            rank = quality['rank']
            text += (
                f', line_rmse={line_rmse:.3f} m, '
                f'rank={rank}'
            )
        if quality['method'] == 'hybrid':
            line_count = quality['line_correspondences']
            point_count = quality['point_correspondences']
            text += (
                f', groups={line_count}L/'
                f'{point_count}P'
            )
        if 'imu_yaw_fusion_weight' in quality:
            imu_weight = quality['imu_yaw_fusion_weight']
            text += (
                f', imu_yaw_weight={imu_weight:.2f}'
            )
        return text

    def _match_scan(self, points, initial_laser_pose):
        """Try shared-query hybrid ICP, then the configured fallbacks."""
        primary_quality = None
        if self.use_hybrid_icp or self.use_point_to_line:
            source_normals, source_normal_valid, _ = estimate_normals(
                points,
                neighbor_count=self.normal_neighbors,
                search_radius=self.normal_search_radius,
                min_neighbors=self.normal_min_neighbors,
                min_linearity=self.normal_min_linearity,
            )

        if self.use_hybrid_icp:
            matched_laser_pose, quality = hybrid_icp(
                points,
                self._submap_index,
                initial_laser_pose,
                source_normals=source_normals,
                source_normal_valid=source_normal_valid,
                max_iterations=self.hybrid_max_iterations,
                max_correspondence_distance=(
                    self.max_correspondence_distance
                ),
                max_normal_angle=self.normal_max_angle,
                trim_fraction=self.trim_fraction,
                min_correspondences=self.min_correspondences,
                min_group_correspondences=(
                    self.hybrid_min_group_correspondences
                ),
                line_weight=self.hybrid_line_weight,
                point_weight=self.hybrid_point_weight,
                svd_rcond=self.point_to_line_svd_rcond,
                max_condition_number=(
                    self.point_to_line_max_condition_number
                ),
                max_translation_step=self.max_prediction_correction,
                max_rotation_step=(
                    self.max_prediction_rotation_correction
                ),
            )
            matched_base_pose, accepted = self._evaluate_match(
                matched_laser_pose,
                quality,
                initial_laser_pose,
            )
            if accepted:
                return matched_base_pose, quality, True, None
            primary_quality = quality
            if not self.fallback_to_point_to_point:
                return matched_base_pose, quality, False, primary_quality

        elif self.use_point_to_line:
            matched_laser_pose, quality = point_to_line_icp(
                points,
                self._submap_index,
                initial_laser_pose,
                source_normals=source_normals,
                source_normal_valid=source_normal_valid,
                max_iterations=self.point_to_line_max_iterations,
                max_correspondence_distance=(
                    self.max_correspondence_distance
                ),
                max_normal_angle=self.normal_max_angle,
                trim_fraction=self.trim_fraction,
                min_correspondences=self.min_correspondences,
                svd_rcond=self.point_to_line_svd_rcond,
                max_condition_number=(
                    self.point_to_line_max_condition_number
                ),
                max_translation_step=self.max_prediction_correction,
                max_rotation_step=(
                    self.max_prediction_rotation_correction
                ),
            )
            matched_base_pose, accepted = self._evaluate_match(
                matched_laser_pose,
                quality,
                initial_laser_pose,
            )
            if accepted:
                return matched_base_pose, quality, True, None
            primary_quality = quality
            if not self.fallback_to_point_to_point:
                return matched_base_pose, quality, False, primary_quality

        matched_laser_pose, quality = icp_match(
            points,
            self._submap_index,
            initial_laser_pose,
            max_iterations=self.icp_max_iterations,
            max_correspondence_distance=(
                self.max_correspondence_distance
            ),
            trim_fraction=self.trim_fraction,
            min_correspondences=self.min_correspondences,
        )
        if self.use_hybrid_icp or self.use_point_to_line:
            quality['method'] = 'point_to_point_fallback'
        matched_base_pose, accepted = self._evaluate_match(
            matched_laser_pose,
            quality,
            initial_laser_pose,
        )
        return matched_base_pose, quality, accepted, primary_quality

    def _match_consecutive_scan(self, points, predicted_base_pose):
        """Estimate motion relative to the last confirmed LiDAR scan."""
        previous_laser_pose = self._pose @ self._base_to_laser
        predicted_laser_pose = predicted_base_pose @ self._base_to_laser
        initial_relative_pose = (
            np.linalg.inv(previous_laser_pose) @ predicted_laser_pose
        )
        relative_laser_pose, quality = icp_match(
            points,
            self._previous_scan_index,
            initial_relative_pose,
            max_iterations=self.icp_max_iterations,
            max_correspondence_distance=(
                self.max_correspondence_distance
            ),
            trim_fraction=self.trim_fraction,
            min_correspondences=self.min_correspondences,
        )
        quality['method'] = 'point_to_point_scan_to_scan'
        matched_laser_pose = previous_laser_pose @ relative_laser_pose
        matched_base_pose, accepted = self._evaluate_match(
            matched_laser_pose,
            quality,
            predicted_laser_pose,
        )
        return matched_base_pose, quality, accepted

    def _set_previous_scan(self, points):
        """Cache the latest confirmed scan for the next relative match."""
        self._previous_scan_index = SubmapIndex(
            points,
            compute_normals=False,
        )

    def _stationary_motion(self, matched_base_pose, imu_delta):
        """Return whether matched motion is inside measured sensor noise."""
        relative_motion = np.linalg.inv(self._pose) @ matched_base_pose
        dx, dy, dyaw = matrix_pose(relative_motion)
        icp_stationary = (
            math.hypot(dx, dy) <= self.stationary_translation_deadband
            and abs(dyaw) <= self.stationary_rotation_deadband
        )
        if not icp_stationary:
            return False
        if imu_delta is not None:
            return abs(imu_delta) <= self.stationary_rotation_deadband
        return True

    def _recover_without_prediction(self, points):
        """Retry Point-to-Point ICP from the last confirmed vehicle pose."""
        initial_laser_pose = self._pose @ self._base_to_laser
        matched_laser_pose, quality = icp_match(
            points,
            self._submap_index,
            initial_laser_pose,
            max_iterations=self.icp_max_iterations,
            max_correspondence_distance=(
                self.max_correspondence_distance
            ),
            trim_fraction=self.trim_fraction,
            min_correspondences=self.min_correspondences,
        )
        quality['method'] = 'point_to_point_recovery'
        matched_base_pose, accepted = self._evaluate_match(
            matched_laser_pose,
            quality,
            initial_laser_pose,
        )
        return matched_base_pose, quality, accepted

    def _scan_callback(self, message):
        processing_start = time.perf_counter()
        self._scan_count += 1
        self._last_scan_stamp = _stamp_seconds(message.header.stamp)
        received_frame = message.header.frame_id.lstrip('/')
        expected_frame = self.laser_frame_id.lstrip('/')
        if received_frame != expected_frame:
            if not self._wrong_frame_logged:
                self.get_logger().error(
                    f'Ignoring scan in frame {message.header.frame_id!r}; '
                    f'expected {self.laser_frame_id!r}. Update '
                    'laser_frame_id or the LiDAR driver frame.'
                )
                self._wrong_frame_logged = True
            self._dropped_count += 1
            self._record_processing_time(processing_start)
            return

        scan_time = self._last_scan_stamp
        points = self._scan_points(message)
        self._last_point_count = len(points)
        if len(points) < self.min_correspondences:
            self.get_logger().warning(
                f'Ignoring scan with only {len(points)} usable points.'
            )
            self._dropped_count += 1
            self._record_processing_time(processing_start)
            return

        if not self._initialized:
            self._initialized = True
            self._last_scan_time = scan_time
            self._last_scan_imu_yaw = self._imu_yaw
            self._add_keyframe(points)
            self._set_previous_scan(points)
            self._publish(message.header.stamp, None, 0.0)
            self.get_logger().info(
                f'LiDAR odometry initialized with {len(points)} points.'
            )
            self._record_processing_time(processing_start)
            return

        delta_time = scan_time - self._last_scan_time
        if delta_time <= 0.0 or delta_time > 1.0:
            self._last_scan_time = scan_time
            self._last_scan_imu_yaw = self._imu_yaw
            self._dropped_count += 1
            self._record_processing_time(processing_start)
            return

        imu_delta = self._imu_delta_for_scan(scan_time)
        prediction = self._predict_pose(delta_time, imu_delta)
        (
            matched_base_pose,
            quality,
            accepted,
        ) = self._match_consecutive_scan(
            points,
            prediction,
        )
        primary_quality = None

        old_pose = self._pose.copy()
        recovery_used = False
        if (
                not accepted
                and self._failure_count + 1
                >= self.recovery_after_failures):
            rejected_quality = quality
            (
                recovery_pose,
                recovery_quality,
                recovery_accepted,
            ) = self._recover_without_prediction(points)
            if recovery_accepted:
                matched_base_pose = recovery_pose
                quality = recovery_quality
                accepted = True
                recovery_used = True
                primary_quality = rejected_quality

        if accepted:
            matched_base_pose = self._fuse_matched_yaw(
                old_pose,
                matched_base_pose,
                quality,
                imu_delta,
            )
            if self.enforce_nonholonomic:
                matched_base_pose, lateral = constrain_nonholonomic(
                    old_pose,
                    matched_base_pose,
                )
                self._last_lateral_rejection = lateral
            if self._stationary_motion(matched_base_pose, imu_delta):
                matched_base_pose = old_pose
                quality['stationary_deadband'] = True
                self._stationary_count += 1
            self._pose = matched_base_pose
            self._accepted_count += 1
            self._failure_count = 0
            relative_output = np.linalg.inv(old_pose) @ self._pose
            output_dx, _, output_dyaw = matrix_pose(relative_output)
            self._body_velocity_x = output_dx / delta_time
            # The KMU vehicle is nonholonomic. ICP lateral motion is matching
            # noise and must not feed the next prediction or scan deskew.
            self._body_velocity_y = 0.0
            self._body_yaw_rate = output_dyaw / delta_time
            self._set_previous_scan(points)
            if recovery_used:
                self._recovery_count += 1
                self.get_logger().warning(
                    'Recovered with prediction-free Point-to-Point ICP '
                    f'after rejected prediction '
                    f'({self._quality_text(primary_quality)}).'
                )
            if quality['method'] == 'point_to_point_fallback':
                self._fallback_count += 1
                if (
                        self._fallback_count == 1
                        or self._fallback_count % 20 == 0):
                    self.get_logger().warning(
                        'Primary ICP rejected; accepted Point-to-Point '
                        f'fallback ({self._quality_text(primary_quality)}).'
                    )
        else:
            self._last_imu_fusion_weight = 0.0
            self._failure_count += 1
            self._rejected_count += 1
            # Prediction is only an ICP initial guess. Without a confirmed
            # match, hold the last pose and remove stale motion so one bad
            # estimate cannot run away through prediction and deskew.
            self._body_velocity_x = 0.0
            self._body_velocity_y = 0.0
            self._body_yaw_rate = 0.0
            if self._failure_count == 1 or self._failure_count % 10 == 0:
                self.get_logger().warning(
                    'Scan matching rejected; holding the last pose and '
                    'resetting prediction velocity. '
                    f'{self._quality_text(quality)}, '
                    f'failures={self._failure_count}'
                )

        self._last_scan_time = scan_time
        self._last_scan_imu_yaw = self._imu_yaw
        if accepted and self._keyframe_needed():
            self._add_keyframe(points)
        self._publish(
            message.header.stamp,
            quality if accepted else None,
            delta_time,
            old_pose,
        )
        self._last_quality = quality
        self._record_processing_time(processing_start)

    def _record_processing_time(self, processing_start):
        elapsed_ms = (time.perf_counter() - processing_start) * 1000.0
        self._processing_total_ms += elapsed_ms
        self._processing_max_ms = max(self._processing_max_ms, elapsed_ms)

    @staticmethod
    def _diagnostic_value(key, value):
        return KeyValue(key=key, value=str(value))

    def _publish_diagnostics(self):
        """Publish scan-matching rates, quality, and callback CPU load."""
        now_wall = time.monotonic()
        elapsed = max(now_wall - self._statistics_start, 1.0e-6)
        match_count = self._accepted_count + self._rejected_count
        rejection_ratio = (
            self._rejected_count / match_count if match_count else 0.0
        )
        average_processing_ms = (
            self._processing_total_ms / self._scan_count
            if self._scan_count else 0.0
        )
        callback_load_percent = (
            self._processing_total_ms / (elapsed * 1000.0) * 100.0
        )

        quality = self._last_quality or {}
        x, y, yaw = matrix_pose(self._pose)
        status = DiagnosticStatus()
        status.name = 'kmu/lidar_odometry'
        status.hardware_id = 'xycar_lidar+xycar_imu'
        status.level = DiagnosticStatus.OK
        status.message = 'ICP healthy'
        if self._scan_count == 0:
            status.level = DiagnosticStatus.ERROR
            status.message = 'No LiDAR scans received'
        elif self.use_imu and self._imu_count == 0:
            status.level = DiagnosticStatus.ERROR
            status.message = 'No IMU messages received'
        elif self.use_imu and not self._gyro_calibrated:
            status.level = DiagnosticStatus.WARN
            status.message = 'Gyro calibration in progress'
        elif rejection_ratio > 0.20:
            status.level = DiagnosticStatus.WARN
            status.message = 'High ICP rejection ratio'
        elif callback_load_percent > 70.0:
            status.level = DiagnosticStatus.WARN
            status.message = 'High LiDAR callback load'

        values = {
            'scan_rate_hz': f'{self._scan_count / elapsed:.2f}',
            'imu_rate_hz': f'{self._imu_count / elapsed:.2f}',
            'accepted': self._accepted_count,
            'rejected': self._rejected_count,
            'rejection_ratio': f'{rejection_ratio:.3f}',
            'stationary_deadband': self._stationary_count,
            'dropped': self._dropped_count,
            'usable_points': self._last_point_count,
            'processing_avg_ms': f'{average_processing_ms:.2f}',
            'processing_max_ms': f'{self._processing_max_ms:.2f}',
            'callback_load_percent': f'{callback_load_percent:.1f}',
            'matcher': quality.get('method', 'not initialized'),
            'inlier_ratio': f"{quality.get('inlier_ratio', 0.0):.3f}",
            'rmse_m': f"{quality.get('rmse', math.inf):.4f}",
            'gyro_calibrated': self._gyro_calibrated,
            'imu_yaw_sign': f'{self.imu_yaw_sign:.1f}',
            'gyro_bias_z': f'{self.gyro_bias_z:.6f}',
            'imu_yaw_fused': self._imu_fused_count,
            'imu_yaw_fusion_weight': (
                f'{self._last_imu_fusion_weight:.3f}'
            ),
            'rejected_lateral_m': f'{self._last_lateral_rejection:.4f}',
            'pose_x_m': f'{x:.3f}',
            'pose_y_m': f'{y:.3f}',
            'pose_yaw_rad': f'{yaw:.3f}',
        }
        status.values = [
            self._diagnostic_value(key, value)
            for key, value in values.items()
        ]
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status.append(status)
        self._diagnostics_publisher.publish(array)

        scan_rate = values['scan_rate_hz']
        imu_rate = values['imu_rate_hz']
        rmse = values['rmse_m']
        processing_average = values['processing_avg_ms']
        processing_maximum = values['processing_max_ms']
        callback_load = values['callback_load_percent']
        imu_yaw_fused = values['imu_yaw_fused']
        imu_yaw_weight = values['imu_yaw_fusion_weight']
        self.get_logger().info(
            'ICP stats: '
            f'scan={scan_rate} Hz, '
            f'imu={imu_rate} Hz, '
            f'imu_yaw={imu_yaw_fused}@{imu_yaw_weight}, '
            f'accepted={self._accepted_count}, '
            f'rejected={self._rejected_count}, '
            f'stationary={self._stationary_count}, '
            f'points={self._last_point_count}, '
            f'rmse={rmse} m, '
            f'callback={processing_average}/{processing_maximum} ms avg/max '
            f'({callback_load}%)'
        )

        self._statistics_start = now_wall
        self._scan_count = 0
        self._accepted_count = 0
        self._rejected_count = 0
        self._stationary_count = 0
        self._imu_fused_count = 0
        self._dropped_count = 0
        self._imu_count = 0
        self._processing_total_ms = 0.0
        self._processing_max_ms = 0.0

    def _keyframe_needed(self):
        relative = np.linalg.inv(self._last_keyframe_pose) @ self._pose
        x, y, yaw = matrix_pose(relative)
        return (
            math.hypot(x, y) >= self.keyframe_translation
            or abs(yaw) >= self.keyframe_rotation
        )

    def _add_keyframe(self, laser_points):
        laser_pose = self._pose @ self._base_to_laser
        odom_points = transform_points(laser_points, laser_pose)
        self._keyframes.append(odom_points)
        combined = np.concatenate(tuple(self._keyframes), axis=0)
        self._submap = voxel_downsample(
            combined, self.voxel_size, self.max_submap_points
        )
        self._submap_index = SubmapIndex(
            self._submap,
            normal_neighbors=self.normal_neighbors,
            normal_radius=self.normal_search_radius,
            normal_min_neighbors=self.normal_min_neighbors,
            normal_min_linearity=self.normal_min_linearity,
            compute_normals=(
                self.use_hybrid_icp or self.use_point_to_line
            ),
        )
        self._last_keyframe_pose = self._pose.copy()

    def _publish(self, stamp, quality, delta_time, old_pose=None):
        x, y, yaw = matrix_pose(self._pose)
        quaternion = _yaw_quaternion(yaw)
        odometry = Odometry()
        odometry.header.stamp = stamp
        odometry.header.frame_id = self.odom_frame_id
        odometry.child_frame_id = self.base_frame_id
        odometry.pose.pose.position.x = x
        odometry.pose.pose.position.y = y
        odometry.pose.pose.orientation.z = quaternion[2]
        odometry.pose.pose.orientation.w = quaternion[3]

        if old_pose is not None and delta_time > 0.0:
            relative = np.linalg.inv(old_pose) @ self._pose
            dx, dy, dyaw = matrix_pose(relative)
            odometry.twist.twist.linear.x = dx / delta_time
            odometry.twist.twist.linear.y = dy / delta_time
            odometry.twist.twist.angular.z = dyaw / delta_time

        if quality is None:
            position_variance = 0.25
            yaw_variance = math.radians(15.0) ** 2
        else:
            position_variance = max(
                0.0025,
                quality['rmse'] ** 2
                / max(quality['inlier_ratio'], 0.10),
            )
            yaw_variance = max(
                math.radians(1.0) ** 2,
                position_variance / 4.0,
            )
        odometry.pose.covariance[0] = position_variance
        odometry.pose.covariance[7] = position_variance
        odometry.pose.covariance[14] = 1.0e6
        odometry.pose.covariance[21] = 1.0e6
        odometry.pose.covariance[28] = 1.0e6
        odometry.pose.covariance[35] = yaw_variance
        odometry.twist.covariance = list(odometry.pose.covariance)
        self._odom_publisher.publish(odometry)

        if self.publish_tf:
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self.odom_frame_id
            transform.child_frame_id = self.base_frame_id
            transform.transform.translation.x = x
            transform.transform.translation.y = y
            transform.transform.rotation.z = quaternion[2]
            transform.transform.rotation.w = quaternion[3]
            self._tf_broadcaster.sendTransform(transform)


def main(args=None):
    """Run the LiDAR odometry node."""
    rclpy.init(args=args)
    node = LidarOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
