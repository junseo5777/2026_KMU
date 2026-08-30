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

"""Follow a Hybrid A* path with LiDAR-aware local steering rollouts."""

import math

from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist

from cartographer_2d_slam.hybrid_astar_planner_node import yaw_from_quaternion
from cartographer_2d_slam.path_tracking_dwa import DWAResult
from cartographer_2d_slam.path_tracking_dwa import PathTrackingDWA
from cartographer_2d_slam.path_tracking_dwa import direction_change_index
from cartographer_2d_slam.path_tracking_dwa import filter_vehicle_self_points
from cartographer_2d_slam.path_tracking_dwa import lookahead_index
from cartographer_2d_slam.path_tracking_dwa import nearest_path_index
from cartographer_2d_slam.path_tracking_dwa import normalize_angle
from cartographer_2d_slam.path_tracking_dwa import path_distance
from cartographer_2d_slam.path_tracking_dwa import path_start_reacquire_metrics
from cartographer_2d_slam.path_tracking_dwa import poses_to_local
from cartographer_2d_slam.path_tracking_dwa import poses_to_map
from cartographer_2d_slam.reverse_ssp import ReverseSSP
from cartographer_2d_slam.reverse_ssp import direction_segment_start

from nav_msgs.msg import Path

import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import LaserScan

from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Int8MultiArray
from std_msgs.msg import String

from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


PATH_TOPIC = '/global_path'
DIRECTIONS_TOPIC = '/global_path_directions'
AMCL_POSE_TOPIC = '/amcl_pose'
STATUS_TOPIC = '/path_follower_status'
DEBUG_COMMAND_TOPIC = '/hybrid_dwa_cmd'
LOCAL_PATH_TOPIC = '/local_dwa_path'
MARKERS_TOPIC = '/hybrid_dwa_markers'
AUTO_MODE_LINEAR = -500.0
AUTO_MODE_ANGULAR = -1000.0


def should_check_final_goal(directions, progress_index, current_direction):
    """Allow forward goal checks only when no reverse segment remains."""
    directions = np.asarray(directions, dtype=np.int8).reshape(-1)
    if current_direction < 0:
        return True
    progress_index = int(np.clip(
        progress_index, 0, max(0, len(directions) - 1)
    ))
    return not bool(np.any(directions[progress_index:] < 0))


def transform_freshness_failure(
        stamp, now, timeout, future_tolerance):
    """Return why a dynamic localization TF timestamp is unsafe."""
    stamp_seconds = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
    if stamp_seconds <= 0.0:
        return 'localization TF has a zero timestamp'
    age = float(now) - stamp_seconds
    if age > float(timeout):
        return f'localization TF stale ({age:.2f} s)'
    if age < -float(future_tolerance):
        return f'localization TF is {-age:.2f} s in the future'
    return ''


def _yaw_quaternion(yaw):
    """Return planar quaternion z and w components."""
    return math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def _new_marker(namespace, marker_id, marker_type, frame_id, stamp):
    """Create one initialized visualization marker."""
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.lifetime.sec = 0
    marker.lifetime.nanosec = 300_000_000
    return marker


def create_local_path_message(trajectory, frame_id, stamp):
    """Convert a local ``(x, y, yaw)`` trajectory into nav_msgs/Path."""
    message = Path()
    message.header.frame_id = frame_id
    message.header.stamp = stamp
    for x, y, yaw in np.asarray(trajectory).reshape(-1, 3):
        pose = PoseStamped()
        pose.header = message.header
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z, pose.pose.orientation.w = _yaw_quaternion(
            float(yaw)
        )
        message.poses.append(pose)
    return message


def create_dwa_markers(result, target, frame_id, stamp):
    """Show all candidates, the selected rollout, and lookahead target."""
    marker_array = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    marker_array.markers.append(clear)

    for index, trajectory in enumerate(result.candidates):
        valid = bool(result.valid_candidates[index])
        marker = _new_marker(
            'hybrid_dwa_candidates', index, Marker.LINE_STRIP, frame_id, stamp
        )
        marker.scale.x = 0.01
        marker.color.r = 0.15 if valid else 1.0
        marker.color.g = 0.75 if valid else 0.10
        marker.color.b = 1.0 if valid else 0.05
        marker.color.a = 0.28
        for x, y in trajectory[:, :2]:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.025
            marker.points.append(point)
        marker_array.markers.append(marker)

    if result.success:
        selected = _new_marker(
            'hybrid_dwa_selected', 1000, Marker.LINE_STRIP, frame_id, stamp
        )
        selected.scale.x = 0.02
        selected.color.r = 0.10
        selected.color.g = 1.0
        selected.color.b = 0.15
        selected.color.a = 1.0
        for x, y in result.trajectory[:, :2]:
            point = Point()
            point.x = float(x)
            point.y = float(y)
            point.z = 0.045
            selected.points.append(point)
        marker_array.markers.append(selected)

    target_marker = _new_marker(
        'hybrid_dwa_target', 2000, Marker.SPHERE, frame_id, stamp
    )
    target_marker.pose.position.x = float(target[0])
    target_marker.pose.position.y = float(target[1])
    target_marker.pose.position.z = 0.06
    target_marker.pose.orientation.z, target_marker.pose.orientation.w = (
        _yaw_quaternion(float(target[2]))
    )
    target_marker.scale.x = 0.10
    target_marker.scale.y = 0.10
    target_marker.scale.z = 0.10
    target_marker.color.r = 1.0
    target_marker.color.g = 0.85
    target_marker.color.b = 0.05
    target_marker.color.a = 1.0
    marker_array.markers.append(target_marker)

    return marker_array


def create_reverse_ssp_markers(
        reference, preview, virtual_front, front_projection, frame_id, stamp):
    """Show the reverse reference, SSP prediction, and error geometry."""
    marker_array = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    marker_array.markers.append(clear)

    reference_marker = _new_marker(
        'reverse_ssp_reference', 0, Marker.LINE_STRIP, frame_id, stamp
    )
    reference_marker.scale.x = 0.018
    reference_marker.color.r = 0.10
    reference_marker.color.g = 0.65
    reference_marker.color.b = 1.00
    reference_marker.color.a = 0.90
    for x, y in np.asarray(reference)[:, :2]:
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = 0.025
        reference_marker.points.append(point)
    marker_array.markers.append(reference_marker)

    preview_marker = _new_marker(
        'reverse_ssp_prediction', 1, Marker.LINE_STRIP, frame_id, stamp
    )
    preview_marker.scale.x = 0.025
    preview_marker.color.r = 0.10
    preview_marker.color.g = 1.00
    preview_marker.color.b = 0.20
    preview_marker.color.a = 1.00
    for x, y in np.asarray(preview)[:, :2]:
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = 0.045
        preview_marker.points.append(point)
    marker_array.markers.append(preview_marker)

    for marker_id, namespace, position, color in (
            (2, 'reverse_ssp_virtual_front', virtual_front, (1.0, 0.8, 0.0)),
            (3, 'reverse_ssp_projection', front_projection, (1.0, 0.1, 0.8))):
        marker = _new_marker(
            namespace, marker_id, Marker.SPHERE, frame_id, stamp
        )
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = 0.06
        marker.scale.x = 0.08
        marker.scale.y = 0.08
        marker.scale.z = 0.08
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker_array.markers.append(marker)

    error_marker = _new_marker(
        'reverse_ssp_cross_track', 4, Marker.LINE_STRIP, frame_id, stamp
    )
    error_marker.scale.x = 0.012
    error_marker.color.r = 1.0
    error_marker.color.g = 0.35
    error_marker.color.b = 0.10
    error_marker.color.a = 0.90
    for position in (virtual_front, front_projection):
        point = Point()
        point.x = float(position[0])
        point.y = float(position[1])
        point.z = 0.04
        error_marker.points.append(point)
    marker_array.markers.append(error_marker)
    return marker_array


class HybridDWAFollowerNode(Node):
    """Track forward by DWA and reverse by low-speed SSP."""

    def __init__(self):
        """Configure the controller, safety gates, and ROS interfaces."""
        super().__init__('hybrid_dwa_follower')
        self._declare_parameters()
        self._read_parameters()

        self._dwa = PathTrackingDWA()
        self._reverse_ssp = ReverseSSP()
        self._validate_parameters()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._sensor_transform = None
        self._sensor_frame = None

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Path, self._path_topic, self._path_callback, latched_qos
        )
        self.create_subscription(
            Int8MultiArray,
            self._directions_topic,
            self._directions_callback,
            latched_qos,
        )
        self.create_subscription(
            LaserScan,
            self._scan_topic,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Int32MultiArray,
            self._ultrasonic_topic,
            self._ultrasonic_callback,
            10,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            AMCL_POSE_TOPIC,
            self._amcl_callback,
            10,
        )
        self.create_subscription(
            Twist, self._mode_topic, self._mode_callback, 10
        )

        self._command_publisher = self.create_publisher(
            Twist, self._command_topic, 10
        )
        self._debug_command_publisher = self.create_publisher(
            Twist, DEBUG_COMMAND_TOPIC, 10
        )
        self._local_path_publisher = self.create_publisher(
            Path, LOCAL_PATH_TOPIC, 10
        )
        self._markers_publisher = self.create_publisher(
            MarkerArray, MARKERS_TOPIC, 10
        )
        self._status_publisher = self.create_publisher(
            String, STATUS_TOPIC, latched_qos
        )

        self._path = None
        self._path_receive_time = None
        self._pending_directions = None
        self._directions_receive_time = None
        self._directions = None
        self._navigation_goal = None
        self._forward_extension_start_index = None
        self._path_ready = False
        self._progress_index = 0
        self._reverse_retry_count = 0
        self._reverse_recovery_active = False
        self._reverse_retry_reference = None
        self._reverse_recovery_dwell_started = None
        self._reverse_retry_dwell_started = None
        self._forward_goal_retry_count = 0
        self._forward_goal_reverse_active = False
        self._forward_goal_retry_tail = None
        self._forward_goal_retry_failure_index = None
        self._forward_goal_reverse_start_pose = None
        self._forward_goal_reverse_started = None
        self._forward_goal_reverse_path = None
        self._forward_goal_reverse_dwell_started = None
        self._forward_goal_forward_dwell_started = None
        self._gear_direction = 1
        self._gear_stop_started = None
        self._current_steer = 0.0
        self._goal_reached = False
        self._blocked_recovery_started = None
        self._blocked_recovery_reverse_started = None
        self._blocked_recovery_settle_started = None
        self._blocked_recovery_start_pose = None
        self._blocked_recovery_attempted = False

        self._scan_message = None
        self._scan_receive_time = None
        self._ultrasonic_values = None
        self._ultrasonic_receive_time = None
        self._reverse_ultrasonic_close_count = 0
        self._amcl_pose = None
        self._amcl_pose_stamp = None
        self._amcl_pose_frame = None
        self._amcl_pose_received = False
        self._amcl_position_stddev = math.inf
        self._amcl_yaw_stddev = math.inf
        self._auto_mode_active = False
        self._mode_receive_time = None
        self._last_status_state = None
        self._last_status_time = -math.inf

        self.create_timer(1.0 / self._control_rate, self._control_tick)
        mode = 'DRIVE ENABLED' if self._enable_drive else 'SHADOW MODE'
        self.get_logger().warning(
            f'Hybrid path follower ready: forward=DWA, reverse=SSP, {mode}; '
            f'command={self._command_topic}, scan={self._scan_topic}'
        )
        self._publish_status(
            'WAITING_FOR_PATH',
            f'{mode}; waiting for path and direction topics',
            force=True,
        )

    def _declare_parameters(self):
        """Declare follower topics, path tracking, output, and safety gates."""
        self.declare_parameter('enable_drive', True)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('path_topic', PATH_TOPIC)
        self.declare_parameter('directions_topic', DIRECTIONS_TOPIC)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('ultrasonic_topic', '/xycar_ultrasonic')
        self.declare_parameter('command_topic', '/erp_writes')
        self.declare_parameter('mode_topic', '/cmd_vel')
        self.declare_parameter('control_rate', 10.0)

        self.declare_parameter('lookahead_distance', 0.65)
        self.declare_parameter(
            'forward_to_reverse_extension_distance', 0.70
        )
        self.declare_parameter('nearest_search_back', 5)
        self.declare_parameter('nearest_search_ahead', 80)
        self.declare_parameter('max_path_error', 0.80)
        self.declare_parameter('reverse_retry_extension_distance', 0.90)
        self.declare_parameter('forward_path_extension_distance', 0.70)
        self.declare_parameter(
            'forward_goal_retry_reverse_distance', 0.25
        )
        self.declare_parameter(
            'forward_goal_retry_reverse_timeout_sec', 2.0
        )
        self.declare_parameter('enable_start_reacquire', True)
        self.declare_parameter('start_reacquire_distance', 0.75)
        self.declare_parameter('start_reacquire_yaw_tolerance_deg', 30.0)
        self.declare_parameter('start_reacquire_min_travel', 1.00)
        self.declare_parameter('goal_position_tolerance', 0.12)
        self.declare_parameter('reverse_goal_position_tolerance', 0.07)
        self.declare_parameter('goal_yaw_tolerance_deg', 8.0)
        self.declare_parameter('gear_stop_distance', 0.12)
        self.declare_parameter('gear_change_dwell_sec', 1.0)

        self.declare_parameter('forward_speed_cmd', 5.0)
        self.declare_parameter('reverse_speed_cmd', 4.0)
        self.declare_parameter('minimum_speed_cmd', 4.0)
        self.declare_parameter('slowdown_distance', 0.70)
        self.declare_parameter('obstacle_slow_distance', 0.55)
        self.declare_parameter('steer_speed_reduction', 0.55)
        self.declare_parameter('steer_command_limit', 55.0)
        self.declare_parameter('steer_command_sign', -1.0)
        self.declare_parameter('steer_command_gain', 1.2)
        self.declare_parameter('allow_blind_reverse', True)
        self.declare_parameter('require_reverse_ultrasonic', True)
        self.declare_parameter('reverse_ultrasonic_stop_value', 23.0)
        self.declare_parameter('reverse_ultrasonic_timeout_sec', 0.50)
        self.declare_parameter('reverse_ultrasonic_indices', [5, 6, 7])
        self.declare_parameter('reverse_ultrasonic_confirm_count', 5)
        self.declare_parameter('enable_blocked_reverse_recovery', True)
        self.declare_parameter('require_auto_for_blocked_recovery', True)
        self.declare_parameter('mode_timeout_sec', 0.50)
        self.declare_parameter('blocked_reverse_speed_cmd', 5.0)
        self.declare_parameter('blocked_reverse_distance', 0.20)
        self.declare_parameter('blocked_reverse_dwell_sec', 0.50)
        self.declare_parameter('blocked_reverse_timeout_sec', 1.20)
        self.declare_parameter('blocked_reverse_settle_sec', 0.80)

        self.declare_parameter('scan_timeout_sec', 0.50)
        self.declare_parameter('localization_tf_timeout_sec', 0.50)
        self.declare_parameter(
            'localization_tf_future_tolerance_sec', 1.50
        )
        self.declare_parameter('tf_lookup_timeout_sec', 0.05)
        self.declare_parameter('path_pair_timeout_sec', 0.75)
        self.declare_parameter('obstacle_max_range', 3.0)
        self.declare_parameter('scan_stride', 2)
        self.declare_parameter('self_filter_padding', 0.04)
        self.declare_parameter('require_amcl_quality', True)
        self.declare_parameter('max_position_stddev', 0.50)
        self.declare_parameter('max_yaw_stddev', math.radians(15.0))

    def _read_parameters(self):
        """Cache non-DWA ROS and control parameters."""
        def value(name):
            return self.get_parameter(name).value

        self._enable_drive = bool(value('enable_drive'))
        self._map_frame = str(value('map_frame'))
        self._base_frame = str(value('base_frame'))
        self._path_topic = str(value('path_topic'))
        self._directions_topic = str(value('directions_topic'))
        self._scan_topic = str(value('scan_topic'))
        self._ultrasonic_topic = str(value('ultrasonic_topic'))
        self._command_topic = str(value('command_topic'))
        self._mode_topic = str(value('mode_topic'))
        self._control_rate = float(value('control_rate'))
        self._lookahead = float(value('lookahead_distance'))
        self._forward_to_reverse_extension_distance = float(value(
            'forward_to_reverse_extension_distance'
        ))
        self._search_back = int(value('nearest_search_back'))
        self._search_ahead = int(value('nearest_search_ahead'))
        self._max_path_error = float(value('max_path_error'))
        self._reverse_retry_extension_distance = float(
            value('reverse_retry_extension_distance')
        )
        self._forward_path_extension_distance = float(
            value('forward_path_extension_distance')
        )
        self._forward_goal_retry_reverse_distance = float(
            value('forward_goal_retry_reverse_distance')
        )
        self._forward_goal_retry_reverse_timeout = float(
            value('forward_goal_retry_reverse_timeout_sec')
        )
        self._enable_start_reacquire = bool(value(
            'enable_start_reacquire'
        ))
        self._start_reacquire_distance = float(value(
            'start_reacquire_distance'
        ))
        self._start_reacquire_yaw_tolerance = math.radians(float(value(
            'start_reacquire_yaw_tolerance_deg'
        )))
        self._start_reacquire_min_travel = float(value(
            'start_reacquire_min_travel'
        ))
        self._goal_position_tolerance = float(
            value('goal_position_tolerance')
        )
        self._reverse_goal_position_tolerance = float(
            value('reverse_goal_position_tolerance')
        )
        self._goal_yaw_tolerance = math.radians(float(
            value('goal_yaw_tolerance_deg')
        ))
        self._gear_stop_distance = float(value('gear_stop_distance'))
        self._gear_change_dwell = float(value('gear_change_dwell_sec'))
        self._forward_speed = float(value('forward_speed_cmd'))
        self._reverse_speed = float(value('reverse_speed_cmd'))
        self._minimum_speed = float(value('minimum_speed_cmd'))
        self._slowdown_distance = float(value('slowdown_distance'))
        self._obstacle_slow_distance = float(
            value('obstacle_slow_distance')
        )
        self._steer_speed_reduction = float(
            value('steer_speed_reduction')
        )
        self._steer_command_limit = float(value('steer_command_limit'))
        self._steer_command_sign = float(value('steer_command_sign'))
        self._steer_command_gain = float(value('steer_command_gain'))
        self._allow_blind_reverse = bool(value('allow_blind_reverse'))
        self._require_reverse_ultrasonic = bool(
            value('require_reverse_ultrasonic')
        )
        self._reverse_ultrasonic_stop_value = float(
            value('reverse_ultrasonic_stop_value')
        )
        self._reverse_ultrasonic_timeout = float(
            value('reverse_ultrasonic_timeout_sec')
        )
        self._reverse_ultrasonic_indices = tuple(
            int(index) for index in value('reverse_ultrasonic_indices')
        )
        self._reverse_ultrasonic_confirm_count = int(
            value('reverse_ultrasonic_confirm_count')
        )
        self._enable_blocked_reverse_recovery = bool(value(
            'enable_blocked_reverse_recovery'
        ))
        self._require_auto_for_blocked_recovery = bool(value(
            'require_auto_for_blocked_recovery'
        ))
        self._mode_timeout = float(value('mode_timeout_sec'))
        self._blocked_reverse_speed = float(value(
            'blocked_reverse_speed_cmd'
        ))
        self._blocked_reverse_distance = float(value(
            'blocked_reverse_distance'
        ))
        self._blocked_reverse_dwell = float(value(
            'blocked_reverse_dwell_sec'
        ))
        self._blocked_reverse_timeout = float(value(
            'blocked_reverse_timeout_sec'
        ))
        self._blocked_reverse_settle = float(value(
            'blocked_reverse_settle_sec'
        ))
        self._scan_timeout = float(value('scan_timeout_sec'))
        self._localization_tf_timeout = float(
            value('localization_tf_timeout_sec')
        )
        self._localization_tf_future_tolerance = float(
            value('localization_tf_future_tolerance_sec')
        )
        self._tf_lookup_timeout = float(value('tf_lookup_timeout_sec'))
        self._path_pair_timeout = float(value('path_pair_timeout_sec'))
        self._obstacle_max_range = float(value('obstacle_max_range'))
        self._scan_stride = int(value('scan_stride'))
        self._self_filter_padding = float(value('self_filter_padding'))
        self._require_amcl_quality = bool(value('require_amcl_quality'))
        self._max_position_stddev = float(value('max_position_stddev'))
        self._max_yaw_stddev = float(value('max_yaw_stddev'))

    def _validate_parameters(self):
        """Reject unsafe follower parameters before creating the timer."""
        self._dwa.validate()
        self._reverse_ssp.validate()
        positive = {
            'control_rate': self._control_rate,
            'lookahead_distance': self._lookahead,
            'forward_to_reverse_extension_distance': (
                self._forward_to_reverse_extension_distance
            ),
            'max_path_error': self._max_path_error,
            'start_reacquire_distance': self._start_reacquire_distance,
            'start_reacquire_yaw_tolerance': (
                self._start_reacquire_yaw_tolerance
            ),
            'start_reacquire_min_travel': (
                self._start_reacquire_min_travel
            ),
            'goal_position_tolerance': self._goal_position_tolerance,
            'reverse_goal_position_tolerance': (
                self._reverse_goal_position_tolerance
            ),
            'goal_yaw_tolerance': self._goal_yaw_tolerance,
            'gear_stop_distance': self._gear_stop_distance,
            'forward_speed_cmd': self._forward_speed,
            'reverse_speed_cmd': self._reverse_speed,
            'minimum_speed_cmd': self._minimum_speed,
            'reverse_ultrasonic_stop_value': (
                self._reverse_ultrasonic_stop_value
            ),
            'reverse_ultrasonic_timeout_sec': (
                self._reverse_ultrasonic_timeout
            ),
            'slowdown_distance': self._slowdown_distance,
            'obstacle_slow_distance': self._obstacle_slow_distance,
            'steer_command_limit': self._steer_command_limit,
            'steer_command_gain': self._steer_command_gain,
            'mode_timeout_sec': self._mode_timeout,
            'blocked_reverse_speed_cmd': self._blocked_reverse_speed,
            'blocked_reverse_distance': self._blocked_reverse_distance,
            'blocked_reverse_dwell_sec': self._blocked_reverse_dwell,
            'blocked_reverse_timeout_sec': self._blocked_reverse_timeout,
            'blocked_reverse_settle_sec': self._blocked_reverse_settle,
            'scan_timeout_sec': self._scan_timeout,
            'localization_tf_timeout_sec': self._localization_tf_timeout,
            'localization_tf_future_tolerance_sec': (
                self._localization_tf_future_tolerance
            ),
            'tf_lookup_timeout_sec': self._tf_lookup_timeout,
            'path_pair_timeout_sec': self._path_pair_timeout,
            'obstacle_max_range': self._obstacle_max_range,
        }
        invalid = [name for name, item in positive.items() if item <= 0.0]
        if invalid:
            raise ValueError(
                f'follower parameters must be positive: {", ".join(invalid)}'
            )
        if self._minimum_speed > self._forward_speed:
            raise ValueError('minimum_speed_cmd exceeds forward_speed_cmd')
        if self._reverse_retry_extension_distance <= 0.0:
            raise ValueError(
                'reverse_retry_extension_distance must be positive'
            )
        if self._forward_path_extension_distance <= 0.0:
            raise ValueError(
                'forward_path_extension_distance must be positive'
            )
        if self._forward_goal_retry_reverse_distance <= 0.0:
            raise ValueError(
                'forward_goal_retry_reverse_distance must be positive'
            )
        if self._forward_goal_retry_reverse_timeout <= 0.0:
            raise ValueError(
                'forward_goal_retry_reverse_timeout_sec must be positive'
            )
        if self._reverse_ultrasonic_stop_value < 0.0:
            raise ValueError(
                'reverse_ultrasonic_stop_value cannot be negative'
            )
        if (
                not self._reverse_ultrasonic_indices
                or any(index < 0 for index in self._reverse_ultrasonic_indices)
        ):
            raise ValueError(
                'reverse_ultrasonic_indices must contain non-negative indices'
            )
        if self._reverse_ultrasonic_confirm_count < 1:
            raise ValueError(
                'reverse_ultrasonic_confirm_count must be positive'
            )
        if self._search_back < 0 or self._search_ahead < 1:
            raise ValueError('nearest path search window is invalid')
        if self._scan_stride < 1:
            raise ValueError('scan_stride must be positive')
        if not 0.0 <= self._steer_speed_reduction < 1.0:
            raise ValueError('steer_speed_reduction must be in [0, 1)')
        if self._steer_command_sign not in {-1.0, 1.0}:
            raise ValueError('steer_command_sign must be +1 or -1')

    def _now_seconds(self):
        """Return the node clock as floating-point seconds."""
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _path_callback(self, message):
        """Cache a new latched global path and wait for matching directions."""
        if message.header.frame_id.lstrip('/') != self._map_frame.lstrip('/'):
            self._path = None
            self._path_ready = False
            self._goal_reached = False
            self._stop(
                'INVALID_PATH_FRAME',
                f'path frame={message.header.frame_id!r}',
            )
            return
        if not message.poses:
            self._path = None
            self._path_ready = False
            self._goal_reached = False
            self._stop('WAITING_FOR_PATH', 'planner published an empty path')
            return
        self._path = np.asarray([
            (
                pose.pose.position.x,
                pose.pose.position.y,
                yaw_from_quaternion(pose.pose.orientation),
            )
            for pose in message.poses
        ], dtype=np.float64)
        self._navigation_goal = self._path[-1].copy()
        self._forward_extension_start_index = None
        self._path_receive_time = self._now_seconds()
        self._path_ready = False
        self._goal_reached = False
        self._progress_index = 0
        self._reverse_retry_count = 0
        self._reverse_recovery_active = False
        self._reverse_retry_reference = None
        self._reverse_recovery_dwell_started = None
        self._reverse_retry_dwell_started = None
        self._forward_goal_retry_count = 0
        self._forward_goal_reverse_active = False
        self._forward_goal_retry_tail = None
        self._forward_goal_retry_failure_index = None
        self._forward_goal_reverse_start_pose = None
        self._forward_goal_reverse_started = None
        self._forward_goal_reverse_path = None
        self._forward_goal_reverse_dwell_started = None
        self._forward_goal_forward_dwell_started = None
        self._gear_stop_started = None
        self._current_steer = 0.0
        self._reset_blocked_reverse_recovery()
        self._try_pair_path()

    def _directions_callback(self, message):
        """Cache the gear sequence paired with the global path."""
        self._pending_directions = np.asarray(
            message.data, dtype=np.int8
        ).reshape(-1)
        self._directions_receive_time = self._now_seconds()
        self._try_pair_path()

    def _try_pair_path(self):
        """Arm tracking only when path and direction messages match."""
        if self._path_ready:
            return
        if (
                self._path is None
                or self._pending_directions is None
                or self._path_receive_time is None
                or self._directions_receive_time is None):
            return
        if abs(
                self._path_receive_time - self._directions_receive_time
                ) > self._path_pair_timeout:
            return
        if len(self._pending_directions) != len(self._path):
            self._stop(
                'INVALID_DIRECTIONS',
                f'{len(self._pending_directions)} directions for '
                f'{len(self._path)} poses',
            )
            return
        if np.any(self._pending_directions == 0):
            self._stop('INVALID_DIRECTIONS', 'direction cannot be zero')
            return
        self._directions = np.where(
            self._pending_directions > 0, 1, -1
        ).astype(np.int8)
        self._append_forward_only_extension()
        self._gear_direction = int(self._directions[0])
        self._path_ready = True
        self.get_logger().info(
            f'Accepted Hybrid A* path: {len(self._path)} poses, '
            f'initial gear={self._gear_direction:+d}'
        )
        self._publish_status(
            'PATH_READY', f'{len(self._path)} poses', force=True
        )

    def _append_forward_only_extension(self):
        """Append a short sampled tangent to a forward-only global path."""
        if len(self._path) == 0 or np.any(self._directions < 0):
            self._forward_extension_start_index = None
            return
        endpoint = self._path[-1].copy()
        distance = self._forward_path_extension_distance
        spacing = self._reverse_ssp.preview_spacing
        count = max(1, int(math.ceil(distance / spacing)))
        samples = np.linspace(distance / count, distance, count)
        yaw = float(endpoint[2])
        extension = np.column_stack((
            endpoint[0] + samples * math.cos(yaw),
            endpoint[1] + samples * math.sin(yaw),
            np.full(count, yaw),
        ))
        self._forward_extension_start_index = len(self._path)
        self._path = np.vstack((self._path, extension))
        self._directions = np.concatenate((
            self._directions,
            np.ones(count, dtype=np.int8),
        ))

    def _forward_to_reverse_lookahead(
            self, vehicle_pose, segment_end, progress_index=None):
        """Target a sampled forward tangent beyond a forward/reverse cusp."""
        if progress_index is None:
            progress_index = self._progress_index
        progress_index = int(np.clip(
            progress_index, 0, int(segment_end)
        ))
        endpoint = self._path[int(segment_end)].copy()
        distance = self._forward_to_reverse_extension_distance
        spacing = self._reverse_ssp.preview_spacing
        count = max(1, int(math.ceil(distance / spacing)))
        samples = np.linspace(distance / count, distance, count)
        yaw = float(endpoint[2])
        extension = np.column_stack((
            endpoint[0] + samples * math.cos(yaw),
            endpoint[1] + samples * math.sin(yaw),
            np.full(count, yaw),
        ))
        reference = np.vstack((
            self._path[progress_index:int(segment_end) + 1],
            extension,
        ))
        target_index = lookahead_index(
            reference, 0, len(reference) - 1, self._lookahead
        )
        target = poses_to_local(
            reference[target_index:target_index + 1], vehicle_pose
        )[0]
        return target, target_index == len(reference) - 1, reference

    def _gear_change_stop_required(
            self,
            has_gear_change,
            next_direction,
            endpoint_distance,
            remaining,
            forward_extension_reached=False):
        """Return whether the current gear segment must stop now."""
        if not has_gear_change:
            return False
        if self._gear_stop_started is not None:
            return True
        if self._gear_direction > 0 and int(next_direction) < 0:
            return bool(forward_extension_reached)
        return (
            endpoint_distance <= self._gear_stop_distance
            or remaining <= self._gear_stop_distance
        )

    def _scan_callback(self, message):
        """Cache the newest scan for conversion in the control tick."""
        self._scan_message = message
        self._scan_receive_time = self._now_seconds()

    def _ultrasonic_callback(self, message):
        """Cache values and debounce the configured rear sensors."""
        self._ultrasonic_values = np.asarray(
            message.data, dtype=np.float64
        ).reshape(-1)
        self._ultrasonic_receive_time = self._now_seconds()
        usable_indices = [
            index for index in self._reverse_ultrasonic_indices
            if index < len(self._ultrasonic_values)
        ]
        rear_minimum = (
            float(np.min(self._ultrasonic_values[usable_indices]))
            if usable_indices else math.inf
        )
        if rear_minimum <= self._reverse_ultrasonic_stop_value:
            self._reverse_ultrasonic_close_count = min(
                self._reverse_ultrasonic_close_count + 1,
                self._reverse_ultrasonic_confirm_count,
            )
        else:
            self._reverse_ultrasonic_close_count = 0

    def _reverse_ultrasonic_stop_reason(self, now):
        """Return why reverse motion must stop from ultrasonic input."""
        if not getattr(self, '_require_reverse_ultrasonic', False):
            return ''
        values = getattr(self, '_ultrasonic_values', None)
        received = getattr(self, '_ultrasonic_receive_time', None)
        if values is None or received is None:
            return 'no reverse ultrasonic message received'
        age = float(now) - float(received)
        if age < -1.0 or age > self._reverse_ultrasonic_timeout:
            return f'reverse ultrasonic stale ({age:.2f} s)'
        if len(values) == 0:
            return 'reverse ultrasonic array is empty'
        usable_indices = [index for index in self._reverse_ultrasonic_indices
                          if index < len(values)]
        if not usable_indices:
            return 'reverse ultrasonic array has no configured values'
        minimum = float(np.min(values[usable_indices]))
        close_count = getattr(self, '_reverse_ultrasonic_close_count', 0)
        if (
                minimum <= self._reverse_ultrasonic_stop_value
                and close_count >= self._reverse_ultrasonic_confirm_count
        ):
            return (
                f'reverse ultrasonic minimum={minimum:.1f} '
                f'<= {self._reverse_ultrasonic_stop_value:.1f} '
                f'for {close_count} consecutive readings'
            )
        return ''

    def _amcl_callback(self, message):
        """Cache AMCL pose for RViz preview and covariance for drive safety."""
        pose = message.pose.pose
        self._amcl_pose = (
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
        )
        self._amcl_pose_stamp = message.header.stamp
        self._amcl_pose_frame = message.header.frame_id.lstrip('/')
        covariance = message.pose.covariance
        position_variance = max(
            0.0, float(covariance[0]), float(covariance[7])
        )
        self._amcl_position_stddev = math.sqrt(position_variance)
        self._amcl_yaw_stddev = math.sqrt(max(0.0, float(covariance[35])))
        self._amcl_pose_received = True

    def _mode_callback(self, message):
        """Track the repeated AUTO sentinel published on ``/cmd_vel``."""
        self._auto_mode_active = (
            math.isclose(message.linear.x, AUTO_MODE_LINEAR, abs_tol=1.0e-6)
            and math.isclose(
                message.angular.z, AUTO_MODE_ANGULAR, abs_tol=1.0e-6
            )
        )
        self._mode_receive_time = self._now_seconds()

    def _localization_failure(self):
        """
        Return an AMCL quality failure or an empty string.

        AMCL publishes ``/amcl_pose`` after filter updates, not as a fixed-rate
        heartbeat. A stationary, correctly localized vehicle can therefore go
        many seconds without a new pose message. Dynamic map TF freshness is
        checked separately in :meth:`_vehicle_pose`.
        """
        if not self._require_amcl_quality:
            return ''
        if not self._amcl_pose_received:
            return 'no /amcl_pose received'
        if self._amcl_position_stddev > self._max_position_stddev:
            return (
                f'AMCL position stddev={self._amcl_position_stddev:.2f} m'
            )
        if self._amcl_yaw_stddev > self._max_yaw_stddev:
            return (
                f'AMCL yaw stddev='
                f'{math.degrees(self._amcl_yaw_stddev):.1f} deg'
            )
        return ''

    def _lookup_planar_transform(self, target_frame, source_frame):
        """Return planar target-from-source transform as x, y, yaw."""
        transform = self._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
            timeout=Duration(seconds=self._tf_lookup_timeout),
        )
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            float(translation.x),
            float(translation.y),
            yaw_from_quaternion(rotation),
        )

    def _vehicle_pose(self, now):
        """Return a live map-frame pose and its exact TF timestamp."""
        transform = self._tf_buffer.lookup_transform(
            self._map_frame,
            self._base_frame,
            Time(),
            timeout=Duration(seconds=self._tf_lookup_timeout),
        )
        failure = transform_freshness_failure(
            transform.header.stamp,
            now,
            self._localization_tf_timeout,
            self._localization_tf_future_tolerance,
        )
        if failure:
            raise RuntimeError(failure)
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        return (
            (
                float(translation.x),
                float(translation.y),
                yaw_from_quaternion(rotation),
            ),
            transform.header.stamp,
        )

    def _scan_obstacles(self, now):
        """Transform valid raw LaserScan returns into base_link points."""
        if self._scan_message is None or self._scan_receive_time is None:
            raise RuntimeError('no scan received')
        age = now - self._scan_receive_time
        if age < -1.0 or age > self._scan_timeout:
            raise RuntimeError(f'scan stale ({age:.2f} s)')
        message = self._scan_message
        frame = message.header.frame_id.lstrip('/')
        if not frame:
            raise RuntimeError('scan has an empty frame_id')
        if frame != self._base_frame.lstrip('/'):
            if self._sensor_frame != frame or self._sensor_transform is None:
                self._sensor_transform = self._lookup_planar_transform(
                    self._base_frame, frame
                )
                self._sensor_frame = frame
            transform_x, transform_y, transform_yaw = self._sensor_transform
        else:
            transform_x, transform_y, transform_yaw = (0.0, 0.0, 0.0)

        ranges = np.asarray(message.ranges, dtype=np.float64)
        indices = np.arange(0, len(ranges), self._scan_stride)
        ranges = ranges[indices]
        angles = message.angle_min + indices * message.angle_increment
        valid = (
            np.isfinite(ranges)
            & (ranges >= message.range_min)
            & (ranges <= min(message.range_max, self._obstacle_max_range))
        )
        ranges = ranges[valid]
        angles = angles[valid]
        sensor_x = ranges * np.cos(angles)
        sensor_y = ranges * np.sin(angles)
        cosine = math.cos(transform_yaw)
        sine = math.sin(transform_yaw)
        points = np.column_stack((
            transform_x + cosine * sensor_x - sine * sensor_y,
            transform_y + sine * sensor_x + cosine * sensor_y,
        ))
        return filter_vehicle_self_points(
            points,
            self._dwa.vehicle_length,
            self._dwa.vehicle_width,
            self._dwa.rear_overhang,
            self._self_filter_padding,
        )

    def _calculate_dwa_preview(self, vehicle_pose, obstacles):
        """Calculate DWA without changing follower progress or gear."""
        segment_end = direction_change_index(
            self._directions,
            self._progress_index,
            self._gear_direction,
        )
        nearest, _ = nearest_path_index(
            self._path,
            vehicle_pose[:2],
            self._progress_index,
            self._search_back,
            self._search_ahead,
            end_index=segment_end,
        )
        preview_index = max(self._progress_index, nearest)
        has_gear_change = (
            int(self._directions[segment_end]) != self._gear_direction
        )
        forward_to_reverse = (
            has_gear_change
            and self._gear_direction > 0
            and self._directions[segment_end] < 0
        )
        if forward_to_reverse:
            target, _, _ = self._forward_to_reverse_lookahead(
                vehicle_pose, segment_end, progress_index=preview_index
            )
        else:
            target_index = lookahead_index(
                self._path,
                preview_index,
                segment_end,
                self._lookahead,
            )
            target = poses_to_local(
                self._path[target_index:target_index + 1], vehicle_pose
            )[0]
        remaining = path_distance(
            self._path, preview_index, segment_end
        )
        endpoint_distance = math.hypot(
            self._path[segment_end, 0] - vehicle_pose[0],
            self._path[segment_end, 1] - vehicle_pose[1],
        )
        prediction_distance = min(
            self._dwa.prediction_distance,
            max(remaining, endpoint_distance, self._dwa.step),
        )
        result = self._dwa.dwa(
            target,
            obstacles,
            1,
            steer=0.0,
            prediction_distance=prediction_distance,
        )
        return result, target

    @staticmethod
    def _dwa_result_to_map(result, vehicle_pose):
        """Transform one local DWA result into map coordinates for RViz."""
        return DWAResult(
            result.success,
            result.reason,
            poses_to_map(result.trajectory, vehicle_pose),
            result.steer,
            result.score,
            result.clearance,
            tuple(
                poses_to_map(trajectory, vehicle_pose)
                for trajectory in result.candidates
            ),
            result.valid_candidates,
        )

    def _calculate_reverse_ssp(self, vehicle_pose, segment_end):
        """Calculate SSP from the current reverse segment in map frame."""
        segment_start = direction_segment_start(
            self._directions, self._progress_index
        )
        first = max(
            segment_start, self._progress_index - self._search_back
        )
        reference = self._path[first:segment_end + 1]
        return self._reverse_ssp.track(reference, vehicle_pose)

    def _publish_reverse_ssp_visualization(
            self, result, vehicle_pose, stamp, frame_id):
        """Publish the SSP reference and prediction in map or base_link."""
        preview = self._reverse_ssp.predict(result.steer)
        if frame_id.lstrip('/') == self._map_frame.lstrip('/'):
            reference = result.reference
            displayed_preview = poses_to_map(preview, vehicle_pose)
            virtual_front = result.virtual_front
            front_projection = result.front_projection
        else:
            reference = poses_to_local(result.reference, vehicle_pose)
            points = np.array([
                [
                    result.virtual_front[0],
                    result.virtual_front[1],
                    vehicle_pose[2],
                ],
                [
                    result.front_projection[0],
                    result.front_projection[1],
                    vehicle_pose[2],
                ],
            ])
            local_points = poses_to_local(points, vehicle_pose)
            displayed_preview = preview
            virtual_front = local_points[0, :2]
            front_projection = local_points[1, :2]

        self._markers_publisher.publish(create_reverse_ssp_markers(
            reference,
            displayed_preview,
            virtual_front,
            front_projection,
            frame_id,
            stamp,
        ))
        self._local_path_publisher.publish(create_local_path_message(
            displayed_preview, frame_id, stamp
        ))

    def _publish_unsafe_localization_preview(self, now):
        """Publish map-frame DWA or SSP markers without allowing drive."""
        if self._amcl_pose is None or self._amcl_pose_stamp is None:
            return False
        map_frame = self._map_frame.lstrip('/')
        if self._amcl_pose_frame and self._amcl_pose_frame != map_frame:
            return False
        if self._gear_direction < 0:
            segment_end = direction_change_index(
                self._directions,
                self._progress_index,
                self._gear_direction,
            )
            result = self._calculate_reverse_ssp(
                self._amcl_pose, segment_end
            )
            self._publish_reverse_ssp_visualization(
                result,
                self._amcl_pose,
                self._amcl_pose_stamp,
                self._map_frame,
            )
            return True
        try:
            obstacles = self._scan_obstacles(now)
        except (TransformException, RuntimeError):
            # Localization debugging should still show the geometric candidates
            # if the scan is temporarily unavailable.  These results never
            # reach the actuator because the caller immediately stops.
            obstacles = np.empty((0, 2), dtype=np.float64)
        result, target = self._calculate_dwa_preview(
            self._amcl_pose, obstacles
        )
        mapped_result = self._dwa_result_to_map(
            result, self._amcl_pose
        )
        mapped_target = poses_to_map(
            target[None, :], self._amcl_pose
        )[0]
        self._markers_publisher.publish(create_dwa_markers(
            mapped_result,
            mapped_target,
            self._map_frame,
            self._amcl_pose_stamp,
        ))
        self._local_path_publisher.publish(create_local_path_message(
            mapped_result.trajectory,
            self._map_frame,
            self._amcl_pose_stamp,
        ))
        return True

    def _control_tick(self):
        """Run one localization, path-progress, DWA, and command cycle."""
        now = self._now_seconds()
        if not self._path_ready:
            self._stop('WAITING_FOR_PATH', 'path/directions pair not ready')
            return
        if self._goal_reached:
            self._stop('GOAL_REACHED', 'holding final stop')
            return
        localization_failure = self._localization_failure()
        if localization_failure:
            self._stop('LOCALIZATION_UNSAFE', localization_failure)
            self._publish_unsafe_localization_preview(now)
            return
        try:
            vehicle_pose, vehicle_pose_stamp = self._vehicle_pose(now)
        except (TransformException, RuntimeError) as error:
            self._stop('LOCALIZATION_UNSAFE', str(error))
            self._publish_unsafe_localization_preview(now)
            return
        if self._hold_reverse_cycle_transition(now):
            return
        if self._forward_goal_reverse_active:
            self._run_forward_goal_reverse(
                now, vehicle_pose, vehicle_pose_stamp
            )
            return
        obstacles = np.empty((0, 2), dtype=np.float64)
        if self._gear_direction > 0:
            try:
                obstacles = self._scan_obstacles(now)
            except (TransformException, RuntimeError) as error:
                self._stop('SENSOR_UNAVAILABLE', str(error))
                return

        segment_end = direction_change_index(
            self._directions,
            self._progress_index,
            self._gear_direction,
        )
        nearest, path_error = nearest_path_index(
            self._path,
            vehicle_pose[:2],
            self._progress_index,
            self._search_back,
            self._search_ahead,
            end_index=segment_end,
        )
        self._progress_index = max(self._progress_index, nearest)
        if (
                path_error > self._max_path_error
                and self._restart_from_path_start_if_safe(vehicle_pose)):
            return

        tracking_endpoint = self._path[-1]
        endpoint_distance = math.hypot(
            tracking_endpoint[0] - vehicle_pose[0],
            tracking_endpoint[1] - vehicle_pose[1],
        )
        endpoint_yaw_error = abs(normalize_angle(
            tracking_endpoint[2] - vehicle_pose[2]
        ))
        goal = (
            self._navigation_goal
            if self._navigation_goal is not None
            else tracking_endpoint
        )
        goal_distance = math.hypot(
            goal[0] - vehicle_pose[0], goal[1] - vehicle_pose[1]
        )
        goal_yaw_error = abs(normalize_angle(goal[2] - vehicle_pose[2]))
        if (
                self._forward_goal_reverse_active
                and endpoint_distance
                <= self._reverse_goal_position_tolerance
                and endpoint_yaw_error <= self._goal_yaw_tolerance):
            self._finish_forward_goal_reverse(
                vehicle_pose, now, 'reverse setup endpoint reached'
            )
            return
        goal_check_allowed = (
            not self._forward_goal_reverse_active
            and not self._reverse_recovery_active
            and should_check_final_goal(
                self._directions,
                self._progress_index,
                self._gear_direction,
            )
        )
        goal_position_tolerance = (
            self._reverse_goal_position_tolerance
            if self._gear_direction < 0
            else self._goal_position_tolerance
        )
        if (
                goal_check_allowed
                and goal_distance <= goal_position_tolerance
                and goal_yaw_error <= self._goal_yaw_tolerance):
            self._goal_reached = True
            self._stop(
                'GOAL_REACHED',
                f'position={goal_distance:.2f} m, '
                f'yaw={math.degrees(goal_yaw_error):.1f} deg',
            )
            return
        if (
                self._enable_drive
                and self._gear_direction < 0
                and not self._allow_blind_reverse):
            self._stop(
                'REVERSE_SENSOR_REQUIRED',
                'rear scan is vehicle-occluded; set allow_blind_reverse '
                'only after providing another safety check',
            )
            return
        if self._gear_direction < 0:
            ultrasonic_reason = self._reverse_ultrasonic_stop_reason(now)
            if ultrasonic_reason:
                if self._forward_goal_reverse_active:
                    self._finish_forward_goal_reverse(
                        vehicle_pose, now, ultrasonic_reason
                    )
                    return
                if self._begin_reverse_retry(
                        vehicle_pose, ultrasonic_reason, now):
                    return
                self._stop('REVERSE_ULTRASONIC_STOP', ultrasonic_reason)
                return

        remaining = path_distance(
            self._path, self._progress_index, segment_end
        )
        endpoint_distance = math.hypot(
            self._path[segment_end, 0] - vehicle_pose[0],
            self._path[segment_end, 1] - vehicle_pose[1],
        )
        has_gear_change = (
            int(self._directions[segment_end]) != self._gear_direction
        )
        forward_to_reverse = (
            has_gear_change
            and self._gear_direction > 0
            and self._directions[segment_end] < 0
        )
        forward_transition_target = None
        forward_extension_reached = False
        if forward_to_reverse:
            (
                forward_transition_target,
                forward_extension_reached,
                _,
            ) = self._forward_to_reverse_lookahead(
                vehicle_pose, segment_end
            )
        gear_stop_required = self._gear_change_stop_required(
            has_gear_change,
            self._directions[segment_end],
            endpoint_distance,
            remaining,
            forward_extension_reached=forward_extension_reached,
        )
        if gear_stop_required:
            if self._gear_stop_started is None:
                self._gear_stop_started = now
                trigger = (
                    'forward lookahead reached '
                    f'{self._forward_to_reverse_extension_distance:.2f} m '
                    'cusp extension endpoint'
                    if forward_to_reverse
                    else f'near gear-change point, '
                         f'distance={endpoint_distance:.2f} m'
                )
                self._stop(
                    'GEAR_CHANGE_STOP',
                    f'{trigger}; stopping before gear '
                    f'{self._directions[segment_end]:+d}',
                )
                return
            dwell = now - self._gear_stop_started
            if dwell < self._gear_change_dwell:
                self._stop(
                    'GEAR_CHANGE_STOP',
                    f'dwell={dwell:.2f}/{self._gear_change_dwell:.2f} s',
                )
                return
            self._gear_direction = int(self._directions[segment_end])
            self._progress_index = segment_end
            self._gear_stop_started = None
            self._current_steer = 0.0
            self._reset_blocked_reverse_recovery()
            self._stop(
                'GEAR_CHANGED', f'new gear={self._gear_direction:+d}'
            )
            return
        self._gear_stop_started = None

        if self._gear_direction < 0:
            try:
                result = self._calculate_reverse_ssp(
                    vehicle_pose, segment_end
                )
            except ValueError as error:
                if self._forward_goal_reverse_active:
                    self._finish_forward_goal_reverse(
                        vehicle_pose, now, str(error)
                    )
                    return
                if self._begin_reverse_retry(
                        vehicle_pose, str(error), now):
                    return
                self._stop('REVERSE_SSP_INVALID', str(error))
                return
            self._publish_reverse_ssp_visualization(
                result,
                vehicle_pose,
                vehicle_pose_stamp,
                self._base_frame,
            )
            if result.tracking_point_at_endpoint:
                reason = 'SSP tracking point reached extension endpoint'
                if self._begin_reverse_retry(vehicle_pose, reason, now):
                    return
                self._stop('REVERSE_EXTENSION_ENDPOINT', reason)
                return
            self._current_steer = result.steer
            speed = -self._reverse_speed
            steer_command = self._steer_command(
                result.steer, self._reverse_ssp.max_steer
            )
            self._publish_command(speed, steer_command)
            state = (
                'REVERSE_SSP_TRACKING'
                if self._enable_drive
                else 'SHADOW_REVERSE_SSP'
            )
            self._publish_status(
                state,
                f'index={self._progress_index}/{len(self._path) - 1}, '
                f'speed={speed:.1f}, steer={steer_command:.1f}, '
                f'cross_track={result.cross_track_error:.3f} m, '
                f'heading={math.degrees(result.heading_error):.1f} deg',
            )
            return

        if forward_transition_target is not None:
            target_index = None
            target = forward_transition_target
        else:
            target_index = lookahead_index(
                self._path,
                self._progress_index,
                segment_end,
                self._lookahead,
            )
            target = poses_to_local(
                self._path[target_index:target_index + 1], vehicle_pose
            )[0]
        if self._reverse_recovery_active and target_index == segment_end:
            self._finish_reverse_recovery(
                vehicle_pose,
                now,
                'forward lookahead reached extension endpoint',
            )
            return
        goal_not_reached = (
            goal_distance > self._goal_position_tolerance
            or goal_yaw_error > self._goal_yaw_tolerance
        )
        lookahead_at_extension_end = (
            self._forward_extension_start_index is not None
            and target_index is not None
            and target_index == len(self._path) - 1
        )
        if lookahead_at_extension_end and goal_not_reached:
            if self._forward_goal_retry_count < 1:
                self._begin_forward_goal_retry(
                    vehicle_pose,
                    now,
                    'lookahead reached forward extension endpoint',
                )
            else:
                self._stop(
                    'FORWARD_GOAL_RETRY_FAILED',
                    f'goal position={goal_distance:.2f} m, '
                    f'yaw={math.degrees(goal_yaw_error):.1f} deg',
                )
            return
        prediction_distance = min(
            self._dwa.prediction_distance,
            max(remaining, endpoint_distance, self._dwa.step),
        )
        result = self._dwa.dwa(
            target,
            obstacles,
            1,
            # SLAM DWA는 이전 조향을 중심으로 창을 줄이지 않고 매 주기
            # -max_steer ~ +max_steer 전체 후보를 다시 평가한다.
            steer=0.0,
            prediction_distance=prediction_distance,
        )
        # Use the exact transform time that produced vehicle_pose.  Publishing
        # with wall-clock "now" races ahead of the LiDAR-stamped odom TF and
        # makes RViz reject base_link markers with future extrapolation.
        stamp = vehicle_pose_stamp
        self._markers_publisher.publish(create_dwa_markers(
            result, target, self._base_frame, stamp
        ))
        self._local_path_publisher.publish(create_local_path_message(
            result.trajectory, self._base_frame, stamp
        ))
        if (
                not self._reverse_recovery_active
                and self._blocked_recovery_started is not None):
            if self._run_blocked_reverse_recovery(
                    now, vehicle_pose, obstacles):
                return
        if not result.success:
            if self._reverse_recovery_active:
                self._finish_reverse_recovery(
                    vehicle_pose,
                    now,
                    f'forward recovery DWA blocked: {result.reason}',
                )
                return
            on_forward_extension = (
                self._forward_extension_start_index is not None
                and self._progress_index
                >= self._forward_extension_start_index
            )
            if on_forward_extension:
                if self._forward_goal_retry_count < 1:
                    self._begin_forward_goal_retry(
                        vehicle_pose,
                        now,
                        f'forward extension DWA blocked: {result.reason}',
                    )
                else:
                    self._stop(
                        'FORWARD_GOAL_RETRY_FAILED',
                        f'forward retry blocked: {result.reason}',
                    )
                return
            if self._run_blocked_reverse_recovery(
                    now, vehicle_pose, obstacles):
                return
            self._stop('BLOCKED', result.reason)
            return

        self._blocked_recovery_attempted = False
        self._current_steer = result.steer
        speed = self._speed_command(
            result.steer,
            result.clearance,
            remaining if has_gear_change else max(remaining, goal_distance),
        )
        steer_command = self._steer_command(result.steer)
        self._publish_command(speed, steer_command)
        state = 'TRACKING' if self._enable_drive else 'SHADOW_TRACKING'
        clearance_text = (
            'inf' if math.isinf(result.clearance)
            else f'{result.clearance:.2f}'
        )
        self._publish_status(
            state,
            f'index={self._progress_index}/{len(self._path) - 1}, '
            f'gear={self._gear_direction:+d}, speed={speed:.1f}, '
            f'steer={steer_command:.1f}, clearance={clearance_text} m',
        )

    def _begin_forward_goal_retry(self, vehicle_pose, now, reason):
        """Arm one goal-heading reverse before the final forward retry."""
        if self._forward_goal_retry_count >= 1:
            return False
        vehicle = np.asarray(vehicle_pose, dtype=np.float64).reshape(3)
        self._forward_goal_retry_tail = self._path.copy()
        self._forward_goal_retry_failure_index = int(np.clip(
            self._progress_index, 0, len(self._path) - 1
        ))
        goal_yaw = (
            float(self._navigation_goal[2])
            if self._navigation_goal is not None
            else float(vehicle[2])
        )
        distance = self._forward_goal_retry_reverse_distance
        spacing = self._reverse_ssp.preview_spacing
        count = max(1, int(math.ceil(distance / spacing)))
        samples = np.linspace(0.0, distance, count + 1)
        self._forward_goal_reverse_path = np.column_stack((
            vehicle[0] - samples * math.cos(goal_yaw),
            vehicle[1] - samples * math.sin(goal_yaw),
            np.full(count + 1, goal_yaw),
        ))
        self._gear_direction = -1
        self._gear_stop_started = None
        self._forward_goal_reverse_active = True
        self._forward_goal_reverse_start_pose = vehicle.copy()
        self._forward_goal_reverse_started = None
        self._forward_goal_reverse_dwell_started = float(now)
        self._forward_goal_forward_dwell_started = None
        self._forward_goal_retry_count += 1
        self._reset_blocked_reverse_recovery()
        self._stop(
            'FORWARD_GOAL_REVERSE_GEAR_STOP',
            f'{reason}; dwell=0.00/{self._gear_change_dwell:.2f} s',
        )
        return True

    def _run_forward_goal_reverse(
            self, now, vehicle_pose, vehicle_pose_stamp=None):
        """Use SSP to follow the goal-heading reverse setup path."""
        ultrasonic_reason = self._reverse_ultrasonic_stop_reason(now)
        if ultrasonic_reason:
            self._finish_forward_goal_reverse(
                vehicle_pose, now, ultrasonic_reason
            )
            return
        vehicle = np.asarray(vehicle_pose, dtype=np.float64).reshape(3)
        if self._forward_goal_reverse_start_pose is None:
            self._forward_goal_reverse_start_pose = vehicle.copy()
        if self._forward_goal_reverse_started is None:
            self._forward_goal_reverse_started = float(now)
        traveled = float(np.hypot(*(
            vehicle[:2] - self._forward_goal_reverse_start_pose[:2]
        )))
        elapsed = float(now) - self._forward_goal_reverse_started
        if self._forward_goal_reverse_path is None:
            self._finish_forward_goal_reverse(
                vehicle_pose, now, 'goal-heading reverse path is missing'
            )
            return
        reverse_endpoint = self._forward_goal_reverse_path[-1]
        endpoint_distance = float(np.hypot(
            reverse_endpoint[0] - vehicle[0],
            reverse_endpoint[1] - vehicle[1],
        ))
        if (
                endpoint_distance <= self._reverse_ssp.preview_spacing
                or elapsed >= self._forward_goal_retry_reverse_timeout):
            limit = (
                'path endpoint'
                if endpoint_distance <= self._reverse_ssp.preview_spacing
                else 'timeout'
            )
            self._finish_forward_goal_reverse(
                vehicle_pose,
                now,
                f'{limit}; SSP reverse traveled={traveled:.2f} m',
            )
            return
        try:
            result = self._reverse_ssp.track(
                self._forward_goal_reverse_path, vehicle_pose
            )
        except ValueError as error:
            self._finish_forward_goal_reverse(
                vehicle_pose, now, f'goal-heading SSP invalid: {error}'
            )
            return
        if vehicle_pose_stamp is not None:
            self._publish_reverse_ssp_visualization(
                result,
                vehicle_pose,
                vehicle_pose_stamp,
                self._base_frame,
            )
        self._current_steer = result.steer
        steer_command = self._steer_command(
            result.steer, self._reverse_ssp.max_steer
        )
        self._publish_command(-self._reverse_speed, steer_command)
        self._publish_status(
            'FORWARD_GOAL_SSP_REVERSING',
            f'traveled={traveled:.2f}/'
            f'{self._forward_goal_retry_reverse_distance:.2f} m, '
            f'endpoint={endpoint_distance:.2f} m, '
            f'speed={-self._reverse_speed:.1f}, '
            f'steer={steer_command:.1f}',
        )

    def _finish_forward_goal_reverse(self, vehicle_pose, now, reason):
        """Prepare the final forward retry from the reverse setup pose."""
        if self._forward_goal_retry_tail is None:
            self._forward_goal_reverse_active = False
            return
        vehicle = np.asarray(vehicle_pose, dtype=np.float64).reshape(3)
        source_path = self._forward_goal_retry_tail
        failure_index = self._forward_goal_retry_failure_index
        if failure_index is None:
            failure_index = len(source_path) - 1
        search_points = max(
            self._search_back,
            int(math.ceil(
                self._forward_goal_retry_reverse_distance
                / self._reverse_ssp.preview_spacing
            )) + 2,
        )
        first = max(0, int(failure_index) - search_points)
        last = min(len(source_path) - 1, int(failure_index))
        search_path = source_path[first:last + 1]
        distances = np.hypot(
            search_path[:, 0] - vehicle[0],
            search_path[:, 1] - vehicle[1],
        )
        local_index = int(np.argmin(distances))
        rejoin_index = first + local_index
        self._path = source_path
        self._directions = np.ones(len(source_path), dtype=np.int8)
        self._progress_index = rejoin_index
        self._gear_direction = 1
        self._gear_stop_started = None
        self._forward_goal_reverse_active = False
        self._forward_goal_retry_tail = None
        self._forward_goal_retry_failure_index = None
        self._forward_goal_reverse_start_pose = None
        self._forward_goal_reverse_started = None
        self._forward_goal_reverse_path = None
        self._forward_goal_reverse_dwell_started = None
        self._forward_goal_forward_dwell_started = float(now)
        self._reset_blocked_reverse_recovery()
        self._stop(
            'FORWARD_GOAL_FORWARD_GEAR_STOP',
            f'{reason}; dwell=0.00/{self._gear_change_dwell:.2f} s',
        )

    def _begin_reverse_retry(self, vehicle_pose, reason, now):
        """Arm one goal-yaw forward line after a failed planned reverse."""
        if self._reverse_retry_count >= 1 or self._reverse_recovery_active:
            return False
        segment_start = direction_segment_start(
            self._directions, self._progress_index
        )
        segment_end = direction_change_index(
            self._directions,
            self._progress_index,
            self._gear_direction,
        )
        if self._gear_direction >= 0 or segment_end <= segment_start:
            return False
        self._reverse_retry_reference = self._path[
            segment_start:segment_end + 1
        ].copy()
        goal = (
            self._navigation_goal.copy()
            if self._navigation_goal is not None
            else self._path[segment_end].copy()
        )
        tangent = float(goal[2])
        extension = self._reverse_retry_extension_distance
        extension_count = max(
            1,
            int(math.ceil(extension / self._reverse_ssp.preview_spacing)),
        )
        distances = np.linspace(
            0.0,
            extension,
            extension_count + 1,
        )
        recovery_path = np.column_stack((
            goal[0] + distances * math.cos(tangent),
            goal[1] + distances * math.sin(tangent),
            np.full(extension_count + 1, tangent),
        ))
        self._path = recovery_path
        self._directions = np.ones(len(recovery_path), dtype=np.int8)
        self._progress_index = 0
        self._gear_direction = 1
        self._gear_stop_started = None
        self._reverse_recovery_active = True
        self._reverse_recovery_dwell_started = float(now)
        self._reverse_retry_dwell_started = None
        self._reverse_retry_count += 1
        self._reset_blocked_reverse_recovery()
        self._stop(
            'REVERSE_RECOVERY_GEAR_STOP',
            f'{reason}; dwell=0.00/{self._gear_change_dwell:.2f} s',
        )
        return True

    def _finish_reverse_recovery(self, vehicle_pose, now, reason):
        """Restore the planned reverse reference for the second attempt."""
        if (
                not self._reverse_recovery_active
                or self._reverse_retry_reference is None):
            return
        vehicle = np.asarray(vehicle_pose, dtype=np.float64).reshape(3)
        retry_path = self._reverse_retry_reference
        distances = np.hypot(
            retry_path[:, 0] - vehicle[0],
            retry_path[:, 1] - vehicle[1],
        )
        self._path = retry_path
        self._directions = -np.ones(len(retry_path), dtype=np.int8)
        self._progress_index = int(np.argmin(distances))
        self._gear_direction = -1
        self._gear_stop_started = None
        self._reverse_recovery_active = False
        self._reverse_retry_reference = None
        self._reverse_recovery_dwell_started = None
        self._reverse_retry_dwell_started = float(now)
        self._reset_blocked_reverse_recovery()
        self._stop(
            'REVERSE_RETRY_GEAR_STOP',
            f'{reason}; dwell=0.00/{self._gear_change_dwell:.2f} s',
        )

    def _hold_reverse_cycle_transition(self, now):
        """Hold zero command for each recovery direction change."""
        if self._forward_goal_reverse_dwell_started is not None:
            dwell = float(now) - self._forward_goal_reverse_dwell_started
            if dwell < self._gear_change_dwell:
                self._stop(
                    'FORWARD_GOAL_REVERSE_GEAR_STOP',
                    f'dwell={dwell:.2f}/{self._gear_change_dwell:.2f} s',
                )
                return True
            self._forward_goal_reverse_dwell_started = None
            self._stop(
                'FORWARD_GOAL_REVERSING',
                'gear dwell complete; starting reverse setup',
            )
            return True
        if self._forward_goal_forward_dwell_started is not None:
            dwell = float(now) - self._forward_goal_forward_dwell_started
            if dwell < self._gear_change_dwell:
                self._stop(
                    'FORWARD_GOAL_FORWARD_GEAR_STOP',
                    f'dwell={dwell:.2f}/{self._gear_change_dwell:.2f} s',
                )
                return True
            self._forward_goal_forward_dwell_started = None
            self._stop(
                'FORWARD_GOAL_RETRY_STARTED',
                'gear dwell complete; starting final forward retry',
            )
            return True
        if self._reverse_recovery_dwell_started is not None:
            dwell = float(now) - self._reverse_recovery_dwell_started
            if dwell < self._gear_change_dwell:
                self._stop(
                    'REVERSE_RECOVERY_GEAR_STOP',
                    f'dwell={dwell:.2f}/{self._gear_change_dwell:.2f} s',
                )
                return True
            self._reverse_recovery_dwell_started = None
            self._stop(
                'REVERSE_RECOVERY_FORWARD',
                'gear dwell complete; starting forward recovery',
            )
            return True
        if self._reverse_retry_dwell_started is not None:
            dwell = float(now) - self._reverse_retry_dwell_started
            if dwell < self._gear_change_dwell:
                self._stop(
                    'REVERSE_RETRY_GEAR_STOP',
                    f'dwell={dwell:.2f}/{self._gear_change_dwell:.2f} s',
                )
                return True
            self._reverse_retry_dwell_started = None
            self._stop(
                'REVERSE_RETRY_STARTED',
                'gear dwell complete; starting second reverse',
            )
            return True
        return False

    def _restart_from_path_start_if_safe(self, vehicle_pose):
        """Rewind a stale path only near its start with matching body yaw."""
        if not self._enable_start_reacquire:
            return False
        traveled = path_distance(self._path, 0, self._progress_index)
        if traveled < self._start_reacquire_min_travel:
            return False
        position_error, yaw_error = path_start_reacquire_metrics(
            self._path, vehicle_pose
        )
        if (
                position_error > self._start_reacquire_distance
                or yaw_error > self._start_reacquire_yaw_tolerance):
            return False

        self._progress_index = 0
        self._gear_direction = int(self._directions[0])
        self._gear_stop_started = None
        self._current_steer = 0.0
        self._goal_reached = False
        self._reset_blocked_reverse_recovery()
        self._stop(
            'PATH_RESTARTED',
            f'start error={position_error:.2f} m, '
            f'yaw={math.degrees(yaw_error):.1f} deg',
        )
        return True

    def _reset_blocked_reverse_recovery(self):
        """Clear a blocked-recovery episode and allow one future attempt."""
        self._blocked_recovery_started = None
        self._blocked_recovery_reverse_started = None
        self._blocked_recovery_settle_started = None
        self._blocked_recovery_start_pose = None
        self._blocked_recovery_attempted = False

    def _auto_mode_is_fresh(self, now):
        """Return whether ``total.py`` should forward AUTO commands."""
        if not self._require_auto_for_blocked_recovery:
            return True
        return (
            self._auto_mode_active
            and self._mode_receive_time is not None
            and now - self._mode_receive_time <= self._mode_timeout
        )

    def _finish_blocked_reverse_recovery(self):
        """End the current maneuver without allowing repeated blind reverse."""
        self._blocked_recovery_started = None
        self._blocked_recovery_reverse_started = None
        self._blocked_recovery_settle_started = None
        self._blocked_recovery_start_pose = None
        self._blocked_recovery_attempted = True

    def _run_blocked_reverse_recovery(self, now, vehicle_pose, obstacles):
        """Back straight once after all forward DWA candidates are rejected."""
        eligible = (
            self._enable_blocked_reverse_recovery
            and self._gear_direction > 0
            and self._allow_blind_reverse
            and not self._blocked_recovery_attempted
            and self._auto_mode_is_fresh(now)
        )
        if not eligible:
            if self._blocked_recovery_started is not None:
                self._reset_blocked_reverse_recovery()
            return False

        ultrasonic_checker = getattr(
            self,
            '_reverse_ultrasonic_stop_reason',
            lambda _now: '',
        )
        ultrasonic_reason = ultrasonic_checker(now)
        if ultrasonic_reason:
            self._finish_blocked_reverse_recovery()
            self._stop('REVERSE_ULTRASONIC_STOP', ultrasonic_reason)
            return True

        if self._blocked_recovery_started is None:
            self._blocked_recovery_started = now
            self._blocked_recovery_start_pose = (
                float(vehicle_pose[0]), float(vehicle_pose[1])
            )
            self._current_steer = 0.0
            self._stop(
                'BLOCKED_REVERSE_STOP',
                f'dwell=0.00/{self._blocked_reverse_dwell:.2f} s',
            )
            return True

        if self._blocked_recovery_settle_started is not None:
            settled = now - self._blocked_recovery_settle_started
            if settled < self._blocked_reverse_settle:
                self._stop(
                    'BLOCKED_REVERSE_SETTLE',
                    f'dwell={settled:.2f}/'
                    f'{self._blocked_reverse_settle:.2f} s',
                )
                return True
            self._finish_blocked_reverse_recovery()
            self._stop(
                'BLOCKED_REVERSE_DONE',
                'one straight reverse recovery completed',
            )
            return True

        if self._blocked_recovery_reverse_started is None:
            dwell = now - self._blocked_recovery_started
            if dwell < self._blocked_reverse_dwell:
                self._stop(
                    'BLOCKED_REVERSE_STOP',
                    f'dwell={dwell:.2f}/'
                    f'{self._blocked_reverse_dwell:.2f} s',
                )
                return True
            self._blocked_recovery_reverse_started = now

        traveled = math.hypot(
            vehicle_pose[0] - self._blocked_recovery_start_pose[0],
            vehicle_pose[1] - self._blocked_recovery_start_pose[1],
        )
        reverse_elapsed = now - self._blocked_recovery_reverse_started
        if (
                traveled >= self._blocked_reverse_distance
                or reverse_elapsed >= self._blocked_reverse_timeout):
            self._blocked_recovery_settle_started = now
            limit = (
                'distance' if traveled >= self._blocked_reverse_distance
                else 'timeout'
            )
            self._stop(
                'BLOCKED_REVERSE_SETTLE',
                f'{limit}; traveled={traveled:.2f} m',
            )
            return True

        check_distance = max(
            self._dwa.step,
            self._blocked_reverse_distance - traveled,
        )
        reverse_trajectory = self._dwa.rollout(
            0.0, direction=-1, distance=check_distance
        )
        clearance = self._dwa.trajectory_clearance(
            reverse_trajectory, obstacles
        )
        if clearance < 0.0:
            self._finish_blocked_reverse_recovery()
            self._stop(
                'BLOCKED_REVERSE_UNSAFE',
                f'straight rear clearance={clearance:.2f} m',
            )
            return True

        self._current_steer = 0.0
        self._publish_command(-self._blocked_reverse_speed, 0.0)
        clearance_text = (
            'inf' if math.isinf(clearance) else f'{clearance:.2f}'
        )
        self._publish_status(
            'BLOCKED_REVERSING',
            f'traveled={traveled:.2f}/{self._blocked_reverse_distance:.2f} m, '
            f'speed={-self._blocked_reverse_speed:.1f}, steer=0.0, '
            f'clearance={clearance_text} m',
        )
        return True

    def _speed_command(self, steer, clearance, remaining):
        """Apply forward curvature, obstacle, and endpoint slowdowns."""
        base = self._forward_speed
        minimum = self._minimum_speed
        minimum_scale = min(1.0, minimum / base)
        steer_ratio = min(1.0, abs(steer) / self._dwa.max_steer)
        scale = 1.0 - self._steer_speed_reduction * steer_ratio
        if (
                math.isfinite(clearance)
                and clearance < self._obstacle_slow_distance):
            obstacle_scale = max(
                minimum_scale,
                clearance / self._obstacle_slow_distance,
            )
            scale = min(scale, obstacle_scale)
        if remaining < self._slowdown_distance:
            remaining_scale = max(
                minimum_scale,
                remaining / self._slowdown_distance,
            )
            scale = min(scale, remaining_scale)
        magnitude = max(minimum, base * scale)
        return magnitude

    def _steer_command(self, steer, max_steer=None):
        """Convert positive-left wheel radians to configurable motor units."""
        if max_steer is None:
            max_steer = self._dwa.max_steer
        normalized = float(steer) / float(max_steer)
        base_command = float(np.clip(
            self._steer_command_sign
            * normalized
            * self._steer_command_limit,
            -self._steer_command_limit,
            self._steer_command_limit,
        ))
        return base_command * self._steer_command_gain

    def _publish_command(self, speed, steer):
        """Publish debug intent and publish motor intent only when armed."""
        command = Twist()
        command.linear.x = float(speed)
        command.angular.z = float(steer)
        self._debug_command_publisher.publish(command)
        if self._enable_drive:
            self._command_publisher.publish(command)

    def _stop(self, state, detail):
        """Publish a zero command when armed and report why control stopped."""
        self._publish_command(0.0, 0.0)
        self._publish_status(state, detail)

    def _publish_status(self, state, detail, force=False):
        """Publish state changes and refresh them once per second."""
        now = self._now_seconds()
        state_changed = state != self._last_status_state
        if (
                not force
                and not state_changed
                and now - self._last_status_time < 1.0):
            return
        message = String()
        message.data = f'{state}: {detail}'
        self._status_publisher.publish(message)
        if state_changed:
            self.get_logger().info(message.data)
        self._last_status_state = state
        self._last_status_time = now

    def publish_final_stop(self):
        """Send a final stop at shutdown when this node owns the actuator."""
        if self._enable_drive:
            command = Twist()
            self._command_publisher.publish(command)


def main(args=None):
    """Run the Hybrid A* path follower."""
    rclpy.init(args=args)
    node = None
    try:
        node = HybridDWAFollowerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.publish_final_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
