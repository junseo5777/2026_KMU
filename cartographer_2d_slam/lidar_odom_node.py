#!/usr/bin/env python3
"""Publish odom->base_link using IMU-assisted scan-to-submap ICP."""

from collections import deque
import math

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from tf2_ros import TransformBroadcaster

from cartographer_2d_slam.lidar_odometry import deskew_points
from cartographer_2d_slam.lidar_odometry import icp_match
from cartographer_2d_slam.lidar_odometry import matrix_pose
from cartographer_2d_slam.lidar_odometry import normalize_angle
from cartographer_2d_slam.lidar_odometry import pose_matrix
from cartographer_2d_slam.lidar_odometry import transform_points
from cartographer_2d_slam.lidar_odometry import voxel_downsample


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

        self._base_to_laser = pose_matrix(
            self.lidar_x, self.lidar_y, self.lidar_yaw
        )
        self._laser_to_base = np.linalg.inv(self._base_to_laser)
        self._pose = np.eye(3, dtype=np.float64)
        self._previous_pose = self._pose.copy()
        self._last_keyframe_pose = self._pose.copy()
        self._keyframes = deque(maxlen=self.submap_keyframes)
        self._submap = np.empty((0, 2), dtype=np.float64)
        self._initialized = False
        self._last_scan_time = None
        self._last_scan_imu_yaw = 0.0
        self._failure_count = 0
        self._wrong_frame_logged = False
        self._body_velocity_x = 0.0
        self._body_velocity_y = 0.0
        self._body_yaw_rate = 0.0

        self._imu_last_time = None
        self._imu_yaw = 0.0
        self._imu_yaw_rate = 0.0
        self._gyro_samples = []
        self._gyro_calibration_start = None
        self._gyro_calibrated = self.gyro_calibration_duration <= 0.0

        imu_text = self.imu_topic if self.use_imu else 'disabled'
        self.get_logger().info(
            f'LiDAR odometry ready: scan={self.scan_topic}, imu={imu_text}, '
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
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('use_imu', True)
        self.declare_parameter('gyro_bias_z', 0.0)
        self.declare_parameter('gyro_calibration_duration', 2.0)
        self.declare_parameter('imu_yaw_weight', 1.0)
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
        self.declare_parameter('max_translation_per_scan', 0.80)
        self.declare_parameter('max_rotation_per_scan', 0.50)

    def _read_parameters(self):
        names = [
            'scan_topic', 'imu_topic', 'odom_topic', 'odom_frame_id',
            'base_frame_id', 'laser_frame_id', 'publish_tf', 'use_imu',
            'gyro_bias_z',
            'gyro_calibration_duration', 'imu_yaw_weight', 'lidar_x',
            'lidar_y', 'lidar_yaw', 'deskew_scan', 'min_range', 'max_range',
            'voxel_size', 'max_scan_points', 'max_submap_points',
            'submap_keyframes',
            'keyframe_translation', 'keyframe_rotation',
            'icp_max_iterations', 'max_correspondence_distance',
            'trim_fraction', 'min_correspondences', 'min_inlier_ratio',
            'max_rmse', 'max_translation_per_scan',
            'max_rotation_per_scan',
        ]
        for name in names:
            setattr(self, name, self.get_parameter(name).value)

        if not 0.0 < self.trim_fraction <= 1.0:
            raise ValueError('trim_fraction must be in (0, 1]')
        if self.submap_keyframes < 1:
            raise ValueError('submap_keyframes must be positive')
        if self.min_correspondences < 2:
            raise ValueError('min_correspondences must be at least two')

    def _imu_callback(self, message):
        sample_time = _stamp_seconds(message.header.stamp)
        if self._imu_last_time is None:
            self._imu_last_time = sample_time
            self._gyro_calibration_start = sample_time
            return

        delta_time = sample_time - self._imu_last_time
        self._imu_last_time = sample_time
        if delta_time <= 0.0 or delta_time > 0.5:
            return

        if not self._gyro_calibrated:
            self._gyro_samples.append(message.angular_velocity.z)
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

        yaw_rate = message.angular_velocity.z - self.gyro_bias_z
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

    def _predict_pose(self):
        relative_motion = np.linalg.inv(self._previous_pose) @ self._pose
        prediction = self._pose @ relative_motion
        if self.use_imu and self._gyro_calibrated:
            imu_delta = self._imu_yaw - self._last_scan_imu_yaw
            _, _, predicted_yaw = matrix_pose(prediction)
            _, _, current_yaw = matrix_pose(self._pose)
            imu_yaw = current_yaw + imu_delta
            yaw = normalize_angle(
                predicted_yaw
                + self.imu_yaw_weight
                * normalize_angle(imu_yaw - predicted_yaw)
            )
            prediction[:2, :2] = pose_matrix(0.0, 0.0, yaw)[:2, :2]
        return prediction

    def _scan_callback(self, message):
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
            return

        scan_time = _stamp_seconds(message.header.stamp)
        points = self._scan_points(message)
        if len(points) < self.min_correspondences:
            self.get_logger().warning(
                f'Ignoring scan with only {len(points)} usable points.'
            )
            return

        if not self._initialized:
            self._initialized = True
            self._last_scan_time = scan_time
            self._last_scan_imu_yaw = self._imu_yaw
            self._add_keyframe(points)
            self._publish(message.header.stamp, None, 0.0)
            self.get_logger().info(
                f'LiDAR odometry initialized with {len(points)} points.'
            )
            return

        delta_time = scan_time - self._last_scan_time
        if delta_time <= 0.0 or delta_time > 1.0:
            self._last_scan_time = scan_time
            self._last_scan_imu_yaw = self._imu_yaw
            return

        prediction = self._predict_pose()
        initial_laser_pose = prediction @ self._base_to_laser
        matched_laser_pose, quality = icp_match(
            points,
            self._submap,
            initial_laser_pose,
            max_iterations=self.icp_max_iterations,
            max_correspondence_distance=(
                self.max_correspondence_distance
            ),
            trim_fraction=self.trim_fraction,
            min_correspondences=self.min_correspondences,
        )
        matched_base_pose = matched_laser_pose @ self._laser_to_base
        motion = np.linalg.inv(self._pose) @ matched_base_pose
        dx, dy, dyaw = matrix_pose(motion)
        plausible_motion = (
            math.hypot(dx, dy) <= self.max_translation_per_scan
            and abs(dyaw) <= self.max_rotation_per_scan
        )
        accepted = (
            quality['correspondences'] >= self.min_correspondences
            and quality['inlier_ratio'] >= self.min_inlier_ratio
            and quality['rmse'] <= self.max_rmse
            and plausible_motion
        )

        old_pose = self._pose.copy()
        self._previous_pose = old_pose
        if accepted:
            self._pose = matched_base_pose
            self._failure_count = 0
        else:
            self._pose = prediction
            self._failure_count += 1
            if self._failure_count == 1 or self._failure_count % 10 == 0:
                self.get_logger().warning(
                    'ICP rejected; using motion prediction. '
                    f"inliers={quality['inlier_ratio']:.2f}, "
                    f"rmse={quality['rmse']:.3f} m, "
                    f'failures={self._failure_count}'
                )

        self._last_scan_time = scan_time
        self._last_scan_imu_yaw = self._imu_yaw
        relative_output = np.linalg.inv(old_pose) @ self._pose
        output_dx, output_dy, output_dyaw = matrix_pose(relative_output)
        self._body_velocity_x = output_dx / delta_time
        self._body_velocity_y = output_dy / delta_time
        self._body_yaw_rate = output_dyaw / delta_time
        if accepted and self._keyframe_needed():
            self._add_keyframe(points)
        self._publish(
            message.header.stamp,
            quality if accepted else None,
            delta_time,
            old_pose,
        )

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
