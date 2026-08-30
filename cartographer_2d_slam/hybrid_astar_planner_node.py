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

"""Expose the Hybrid A* planner through simple ROS 2 topics."""

import math
import os
import threading

from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from cartographer_2d_slam.hs import HybridAStar
from cartographer_2d_slam.hs import OccupancyMap

from nav_msgs.msg import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from std_msgs.msg import ColorRGBA
from std_msgs.msg import Int8MultiArray
from std_msgs.msg import String

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


MAP_FRAME = 'map'
INITIAL_POSE_TOPIC = '/initialpose'
AMCL_POSE_TOPIC = '/amcl_pose'
GOAL_TOPIC = '/goal_pose'
PATH_TOPIC = '/global_path'
DIRECTIONS_TOPIC = '/global_path_directions'
MARKERS_TOPIC = '/global_path_markers'
STATUS_TOPIC = '/planning_status'
FOLLOWER_STATUS_TOPIC = '/path_follower_status'
NEAR_GOAL_RETURN_STATES = frozenset({
    'FORWARD_GOAL_RETRY_FAILED',
    'REVERSE_ULTRASONIC_STOP',
    'REVERSE_EXTENSION_ENDPOINT',
    'REVERSE_SSP_INVALID',
    'BLOCKED',
})
COMPETITION_START_POSE = (1.8, 0.9, 3.14)
COMPETITION_PARKING_A_POSE = (0.0, 4.2, 0.0)
COMPETITION_PARKING_B_POSE = (2.1, 3.3, -1.57)
COMPETITION_LEG_NAMES = ('START_TO_A', 'A_TO_B', 'B_TO_START')


def yaw_from_quaternion(orientation):
    """Return planar yaw from a geometry quaternion."""
    sin_yaw = 2.0 * (
        orientation.w * orientation.z
        + orientation.x * orientation.y
    )
    cos_yaw = 1.0 - 2.0 * (
        orientation.y * orientation.y
        + orientation.z * orientation.z
    )
    return math.atan2(sin_yaw, cos_yaw)


def pose_tuple(message):
    """Convert PoseStamped or PoseWithCovarianceStamped to (x, y, yaw)."""
    pose = message.pose.pose if hasattr(message.pose, 'pose') else message.pose
    return (
        float(pose.position.x),
        float(pose.position.y),
        yaw_from_quaternion(pose.orientation),
    )


def _yaw_quaternion(yaw):
    """Return the planar z and w quaternion components for yaw."""
    return math.sin(0.5 * yaw), math.cos(0.5 * yaw)


def create_path_message(poses, frame_id, stamp):
    """Build a nav_msgs/Path from an (N, 3) map-pose array."""
    path = Path()
    path.header.frame_id = frame_id
    path.header.stamp = stamp
    for x, y, yaw in np.asarray(poses, dtype=np.float64).reshape(-1, 3):
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z, pose.pose.orientation.w = _yaw_quaternion(
            float(yaw)
        )
        path.poses.append(pose)
    return path


def _new_marker(namespace, marker_id, marker_type, frame_id, stamp):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    return marker


def _color(red, green, blue, alpha=1.0):
    color = ColorRGBA()
    color.r = red
    color.g = green
    color.b = blue
    color.a = alpha
    return color


def create_path_markers(poses, directions, frame_id, stamp):
    """Color forward path segments red and reverse segments blue in RViz."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1)
    marker_array = MarkerArray()

    segments = _new_marker(
        'hybrid_astar_path', 0, Marker.LINE_LIST, frame_id, stamp
    )
    segments.scale.x = 0.045
    forward_color = _color(1.0, 0.15, 0.05)
    reverse_color = _color(0.10, 0.35, 1.0)
    for index in range(max(0, len(poses) - 1)):
        direction = directions[min(index, len(directions) - 1)]
        color = forward_color if direction >= 0.0 else reverse_color
        for pose in (poses[index], poses[index + 1]):
            point = Point()
            point.x = float(pose[0])
            point.y = float(pose[1])
            point.z = 0.035
            segments.points.append(point)
            segments.colors.append(color)
    marker_array.markers.append(segments)

    endpoint_specs = (
        ('hybrid_astar_start', 1, poses[0], _color(0.1, 1.0, 0.1)),
        ('hybrid_astar_goal', 2, poses[-1], _color(1.0, 0.0, 1.0)),
    )
    for namespace, marker_id, pose, color in endpoint_specs:
        endpoint = _new_marker(
            namespace, marker_id, Marker.ARROW, frame_id, stamp
        )
        endpoint.pose.position.x = float(pose[0])
        endpoint.pose.position.y = float(pose[1])
        endpoint.pose.position.z = 0.06
        endpoint.pose.orientation.z, endpoint.pose.orientation.w = (
            _yaw_quaternion(float(pose[2]))
        )
        endpoint.scale.x = 0.30
        endpoint.scale.y = 0.07
        endpoint.scale.z = 0.10
        endpoint.color = color
        marker_array.markers.append(endpoint)
    return marker_array


def create_mission_waypoint_markers(waypoints, frame_id, stamp):
    """Create fixed START/A/B arrows without changing the active path."""
    marker_array = MarkerArray()
    colors = (
        _color(0.1, 1.0, 0.1),
        _color(1.0, 0.65, 0.0),
        _color(0.65, 0.2, 1.0),
    )
    for index, pose in enumerate(waypoints):
        arrow = _new_marker(
            'competition_waypoints', 10 + index, Marker.ARROW,
            frame_id, stamp
        )
        arrow.pose.position.x = float(pose[0])
        arrow.pose.position.y = float(pose[1])
        arrow.pose.position.z = 0.08
        arrow.pose.orientation.z, arrow.pose.orientation.w = _yaw_quaternion(
            float(pose[2])
        )
        arrow.scale.x = 0.36
        arrow.scale.y = 0.09
        arrow.scale.z = 0.12
        arrow.color = colors[index]
        marker_array.markers.append(arrow)
    return marker_array


class HybridAStarPlannerNode(Node):
    """Plan between RViz poses or from the latest AMCL localization pose."""

    def __init__(self):
        """Load the map, configure Hybrid A*, and connect ROS interfaces."""
        super().__init__('hybrid_astar_planner')
        self._declare_parameters()

        self._map_frame = str(self.get_parameter('map_frame').value)
        self._start_mode = str(self.get_parameter('start_mode').value).lower()
        if self._start_mode not in {
                'preview', 'localization', 'competition'}:
            raise ValueError(
                'start_mode must be preview, localization, or competition'
            )
        self._planning_behavior = str(
            self.get_parameter('planning_behavior').value
        ).lower()
        if self._planning_behavior not in {
                'legacy', 'forward_then_final_reverse'}:
            raise ValueError(
                'planning_behavior must be legacy or '
                'forward_then_final_reverse'
            )
        self._reverse_activation_radius = float(
            self.get_parameter('reverse_activation_radius').value
        )
        self._max_reverse_length = float(
            self.get_parameter('max_reverse_length').value
        )
        self._return_to_start = bool(
            self.get_parameter('return_to_start').value
        )
        self._return_near_goal_distance = float(
            self.get_parameter('return_near_goal_distance').value
        )
        self._mission_start = self._pose_parameter('mission_start_pose')
        self._mission_parking_a = self._pose_parameter(
            'mission_parking_a_pose'
        )
        self._mission_parking_b = self._pose_parameter(
            'mission_parking_b_pose'
        )
        self._mission_goals = (
            self._mission_parking_a,
            self._mission_parking_b,
            self._mission_start,
        )
        self._mission_waypoints = (
            self._mission_start,
            self._mission_parking_a,
            self._mission_parking_b,
        )
        self._mission_dwell_sec = float(
            self.get_parameter('mission_dwell_sec').value
        )
        self._mission_near_goal_distance = float(
            self.get_parameter('mission_near_goal_distance').value
        )
        self._mission_start_position_tolerance = float(
            self.get_parameter('mission_start_position_tolerance').value
        )
        self._mission_start_yaw_tolerance = math.radians(float(
            self.get_parameter('mission_start_yaw_tolerance_deg').value
        ))
        if self._start_mode == 'competition':
            # Competition already contains the final return as its third leg.
            self._return_to_start = False

        map_yaml = os.path.abspath(os.path.expanduser(
            str(self.get_parameter('map_yaml').value)
        ))
        if not os.path.isfile(map_yaml):
            raise FileNotFoundError(f'map_yaml does not exist: {map_yaml}')

        free_threshold = float(
            self.get_parameter('free_thresh_override').value
        )
        occupied_threshold = float(
            self.get_parameter('occupied_thresh_override').value
        )
        self._occupancy_map = OccupancyMap.from_nav2_yaml(
            map_yaml,
            free_thresh=(free_threshold if free_threshold >= 0.0 else None),
            occupied_thresh=(
                occupied_threshold if occupied_threshold >= 0.0 else None
            ),
            unknown_is_blocked=bool(
                self.get_parameter('unknown_is_blocked').value
            ),
        )
        self._planner = HybridAStar(self._occupancy_map)
        self._configure_planner()
        self._validate_planner_parameters()

        latched_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._path_publisher = self.create_publisher(
            Path, PATH_TOPIC, latched_qos
        )
        self._directions_publisher = self.create_publisher(
            Int8MultiArray, DIRECTIONS_TOPIC, latched_qos
        )
        self._markers_publisher = self.create_publisher(
            MarkerArray, MARKERS_TOPIC, latched_qos
        )
        self._status_publisher = self.create_publisher(
            String, STATUS_TOPIC, latched_qos
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            INITIAL_POSE_TOPIC,
            self._initial_pose_callback,
            10,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            AMCL_POSE_TOPIC,
            self._amcl_pose_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            GOAL_TOPIC,
            self._goal_callback,
            10,
        )
        self.create_subscription(
            String,
            FOLLOWER_STATUS_TOPIC,
            self._follower_status_callback,
            10,
        )

        self._preview_start = None
        self._amcl_start = None
        self._amcl_stamp = None
        self._amcl_position_stddev = math.inf
        self._amcl_yaw_stddev = math.inf
        self._goal = None
        self._request_generation = 0
        self._planning = False
        self._pending_plan_request = None
        self._round_trip_origin = None
        self._round_trip_outbound_goal = None
        self._round_trip_phase = 'idle'
        self._pending_near_goal_stop_state = None
        self._mission_phase = (
            'waiting_for_start'
            if self._start_mode == 'competition' else 'disabled'
        )
        self._mission_leg_index = -1
        self._mission_active_goal = None
        self._mission_dwell_started = None
        self._pending_mission_terminal_state = None
        self._mission_last_wait_detail = None
        self._lock = threading.Lock()

        if self._start_mode == 'competition':
            self._mission_timer = self.create_timer(
                0.10, self._competition_mission_tick
            )
            self._publish_status(
                'MISSION_WAITING_FOR_START',
                'set AMCL near the configured START pose',
            )
        else:
            self._mission_timer = None
            self._publish_status(
                'WAITING_FOR_START',
                f'mode={self._start_mode}; set 2D Pose Estimate, then '
                '2D Goal Pose'
                if self._start_mode == 'preview'
                else 'waiting for a valid /amcl_pose, then 2D Goal Pose',
            )
        self.get_logger().info(
            f'Hybrid A* ready in {self._start_mode} mode, '
            f'behavior={self._planning_behavior}; map={map_yaml}, '
            f'{self._occupancy_map.width}x{self._occupancy_map.height} @ '
            f'{self._occupancy_map.resolution:.3f} m'
        )

    def _declare_parameters(self):
        """Declare map, localization-gate, and planner parameters."""
        self.declare_parameter('map_yaml', '')
        self.declare_parameter('map_frame', MAP_FRAME)
        self.declare_parameter('start_mode', 'preview')
        self.declare_parameter(
            'planning_behavior', 'forward_then_final_reverse'
        )
        self.declare_parameter('reverse_activation_radius', 1.2)
        self.declare_parameter('max_reverse_length', 0.8)
        self.declare_parameter('return_to_start', False)
        self.declare_parameter('return_near_goal_distance', 0.30)
        self.declare_parameter('free_thresh_override', 0.19)
        self.declare_parameter('occupied_thresh_override', -1.0)
        self.declare_parameter('unknown_is_blocked', True)
        self.declare_parameter('max_position_stddev', 0.30)
        self.declare_parameter('max_yaw_stddev', math.radians(15.0))
        self.declare_parameter('pose_timeout_sec', 2.0)
        self.declare_parameter(
            'mission_start_pose', list(COMPETITION_START_POSE)
        )
        self.declare_parameter(
            'mission_parking_a_pose', list(COMPETITION_PARKING_A_POSE)
        )
        self.declare_parameter(
            'mission_parking_b_pose', list(COMPETITION_PARKING_B_POSE)
        )
        self.declare_parameter('mission_dwell_sec', 1.0)
        self.declare_parameter('mission_near_goal_distance', 0.30)
        self.declare_parameter('mission_start_position_tolerance', 0.50)
        self.declare_parameter('mission_start_yaw_tolerance_deg', 30.0)

        self.declare_parameter('wheel_base', 0.345)
        self.declare_parameter('max_steer_deg', 25.0)
        self.declare_parameter('steer_count', 9)
        self.declare_parameter('allow_reverse', True)
        self.declare_parameter('step', 0.25)
        self.declare_parameter('substeps', 3)
        self.declare_parameter('xy_resolution', 0.15)
        self.declare_parameter('yaw_resolution_deg', 5.0)
        self.declare_parameter('collision_margin', 0.03)
        self.declare_parameter('safe_dist', 0.35)
        self.declare_parameter('w_length', 1.0)
        self.declare_parameter('w_steer', 0.35)
        self.declare_parameter('w_steer_change', 0.5)
        self.declare_parameter('w_obstacle', 1.5)
        self.declare_parameter('w_reverse', 2.0)
        self.declare_parameter('w_gear_change', 2.0)
        self.declare_parameter('heuristic_weight', 1.4)
        self.declare_parameter('heuristic_coarse_factor', 5)
        self.declare_parameter('max_expansions', 40000)
        self.declare_parameter('time_limit', 10.0)
        self.declare_parameter('connect_interval', 5)
        self.declare_parameter('connect_radius', 2.0)
        self.declare_parameter('connect_max_length', 5.0)
        self.declare_parameter('output_spacing', 0.05)

    def _pose_parameter(self, name):
        """Read and validate one fixed ``[x, y, yaw]`` parameter."""
        values = np.asarray(
            self.get_parameter(name).value, dtype=np.float64
        ).reshape(-1)
        if len(values) != 3 or not np.isfinite(values).all():
            raise ValueError(f'{name} must contain three finite values')
        return tuple(float(value) for value in values)

    def _configure_planner(self):
        """Copy ROS parameters into the pure-Python planner instance."""
        planner = self._planner
        planner.wheel_base = float(self.get_parameter('wheel_base').value)
        planner.max_steer = math.radians(float(
            self.get_parameter('max_steer_deg').value
        ))
        planner.steer_count = int(self.get_parameter('steer_count').value)
        planner.allow_reverse = bool(
            self.get_parameter('allow_reverse').value
        )
        planner.step = float(self.get_parameter('step').value)
        planner.substeps = int(self.get_parameter('substeps').value)
        planner.xy_resolution = float(
            self.get_parameter('xy_resolution').value
        )
        planner.yaw_resolution = math.radians(float(
            self.get_parameter('yaw_resolution_deg').value
        ))
        planner.collision_margin = float(
            self.get_parameter('collision_margin').value
        )
        planner.safe_dist = float(self.get_parameter('safe_dist').value)
        planner.w_length = float(self.get_parameter('w_length').value)
        planner.w_steer = float(self.get_parameter('w_steer').value)
        planner.w_steer_change = float(
            self.get_parameter('w_steer_change').value
        )
        planner.w_obstacle = float(
            self.get_parameter('w_obstacle').value
        )
        planner.w_reverse = float(
            self.get_parameter('w_reverse').value
        )
        planner.w_gear_change = float(
            self.get_parameter('w_gear_change').value
        )
        planner.heuristic_weight = float(
            self.get_parameter('heuristic_weight').value
        )
        planner.heuristic_coarse_factor = int(
            self.get_parameter('heuristic_coarse_factor').value
        )
        planner.max_expansions = int(
            self.get_parameter('max_expansions').value
        )
        planner.time_limit = float(self.get_parameter('time_limit').value)
        planner.connect_interval = int(
            self.get_parameter('connect_interval').value
        )
        planner.connect_radius = float(
            self.get_parameter('connect_radius').value
        )
        planner.connect_max_length = float(
            self.get_parameter('connect_max_length').value
        )
        planner.output_spacing = float(
            self.get_parameter('output_spacing').value
        )

    def _validate_planner_parameters(self):
        """Reject values that would make the search invalid or unsafe."""
        planner = self._planner
        positive = {
            'wheel_base': planner.wheel_base,
            'max_steer_deg': math.degrees(planner.max_steer),
            'step': planner.step,
            'xy_resolution': planner.xy_resolution,
            'yaw_resolution_deg': math.degrees(planner.yaw_resolution),
            'time_limit': planner.time_limit,
            'connect_radius': planner.connect_radius,
            'connect_max_length': planner.connect_max_length,
            'output_spacing': planner.output_spacing,
            'return_near_goal_distance': self._return_near_goal_distance,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(
                f'planner parameters must be positive: {", ".join(invalid)}'
            )
        if planner.max_steer >= math.pi / 2.0:
            raise ValueError('max_steer_deg must be less than 90 degrees')
        if planner.steer_count < 2:
            raise ValueError('steer_count must be at least 2')
        if planner.substeps < 1:
            raise ValueError('substeps must be positive')
        if planner.collision_margin < 0.0 or planner.safe_dist < 0.0:
            raise ValueError(
                'collision_margin and safe_dist cannot be negative'
            )
        costs = {
            'w_length': planner.w_length,
            'w_steer': planner.w_steer,
            'w_steer_change': planner.w_steer_change,
            'w_obstacle': planner.w_obstacle,
            'w_reverse': planner.w_reverse,
            'w_gear_change': planner.w_gear_change,
        }
        invalid_costs = [
            name for name, value in costs.items() if value < 0.0
        ]
        if invalid_costs:
            raise ValueError(
                f'planner costs cannot be negative: '
                f'{", ".join(invalid_costs)}'
            )
        if self._reverse_activation_radius <= 0.0:
            raise ValueError('reverse_activation_radius must be positive')
        if self._max_reverse_length <= 0.0:
            raise ValueError('max_reverse_length must be positive')
        if self._mission_dwell_sec < 0.0:
            raise ValueError('mission_dwell_sec cannot be negative')
        mission_positive = {
            'mission_near_goal_distance': self._mission_near_goal_distance,
            'mission_start_position_tolerance': (
                self._mission_start_position_tolerance
            ),
            'mission_start_yaw_tolerance_deg': math.degrees(
                self._mission_start_yaw_tolerance
            ),
        }
        invalid_mission = [
            name for name, value in mission_positive.items()
            if value <= 0.0
        ]
        if invalid_mission:
            raise ValueError(
                f'mission parameters must be positive: '
                f'{", ".join(invalid_mission)}'
            )
        if planner.heuristic_weight <= 0.0:
            raise ValueError('heuristic_weight must be positive')
        if planner.heuristic_coarse_factor < 1:
            raise ValueError('heuristic_coarse_factor must be positive')
        if planner.max_expansions < 1 or planner.connect_interval < 1:
            raise ValueError(
                'max_expansions and connect_interval must be positive'
            )

    def _pose_in_map_frame(self, message, label):
        """Validate a pose frame and return its planar tuple."""
        frame = message.header.frame_id.lstrip('/')
        expected = self._map_frame.lstrip('/')
        if frame != expected:
            self._publish_status(
                'INVALID_FRAME',
                f'{label} frame={message.header.frame_id!r}; '
                f'expected {self._map_frame!r}',
            )
            return None
        return pose_tuple(message)

    def _initial_pose_callback(self, message):
        """Use RViz 2D Pose Estimate as the preview-only virtual start."""
        if self._start_mode != 'preview':
            return
        start = self._pose_in_map_frame(message, 'start')
        if start is None:
            return
        with self._lock:
            self._preview_start = start
        self.get_logger().info(
            f'Preview start set to x={start[0]:.2f}, y={start[1]:.2f}, '
            f'yaw={math.degrees(start[2]):.1f} deg'
        )
        self._request_plan('outbound', capture_round_trip_origin=True)

    def _amcl_pose_callback(self, message):
        """Cache the latest map pose and covariance for localization mode."""
        start = self._pose_in_map_frame(message, 'AMCL pose')
        if start is None:
            return
        covariance = message.pose.covariance
        position_variance = max(
            0.0, float(covariance[0]), float(covariance[7])
        )
        yaw_variance = max(0.0, float(covariance[35]))
        with self._lock:
            self._amcl_start = start
            self._amcl_stamp = message.header.stamp
            self._amcl_position_stddev = math.sqrt(position_variance)
            self._amcl_yaw_stddev = math.sqrt(yaw_variance)
            pending_stop = self._pending_near_goal_stop_state
            pending_mission_stop = self._pending_mission_terminal_state
        if self._start_mode == 'competition':
            if pending_mission_stop is not None:
                self._try_accept_mission_terminal_stop(
                    pending_mission_stop
                )
            self._competition_mission_tick()
            return
        if pending_stop is not None:
            self._try_arm_near_goal_return(pending_stop)
        self._try_start_return_plan()

    def _goal_callback(self, message):
        """Plan whenever RViz publishes a new 2D Goal Pose."""
        if self._start_mode == 'competition':
            self._publish_status(
                'MISSION_INPUT_IGNORED',
                'RViz goals are disabled in competition mode',
            )
            return
        goal = self._pose_in_map_frame(message, 'goal')
        if goal is None:
            return
        with self._lock:
            self._goal = goal
            self._round_trip_origin = None
            self._round_trip_outbound_goal = goal
            self._round_trip_phase = (
                'outbound_waiting_for_plan'
                if self._return_to_start else 'disabled'
            )
            self._pending_near_goal_stop_state = None
        self.get_logger().info(
            f'Goal set to x={goal[0]:.2f}, y={goal[1]:.2f}, '
            f'yaw={math.degrees(goal[2]):.1f} deg'
        )
        self._request_plan('outbound', capture_round_trip_origin=True)

    def _follower_status_callback(self, message):
        """Start the return leg only after the follower finishes outbound."""
        state = message.data.partition(':')[0].strip()
        if getattr(self, '_start_mode', None) == 'competition':
            self._competition_follower_status(state)
            return
        if not self._return_to_start:
            return
        start_return = False
        round_trip_complete = False
        check_near_goal_stop = False
        with self._lock:
            if state == 'PATH_READY':
                if self._round_trip_phase == 'outbound_path_published':
                    self._round_trip_phase = 'outbound_tracking'
                elif self._round_trip_phase == 'return_path_published':
                    self._round_trip_phase = 'return_tracking'
            elif (
                    state == 'GOAL_REACHED'
                    and self._round_trip_phase == 'outbound_tracking'):
                self._round_trip_phase = 'return_waiting_for_start'
                start_return = True
            elif (
                    state == 'GOAL_REACHED'
                    and self._round_trip_phase == 'return_tracking'):
                self._round_trip_phase = 'complete'
                round_trip_complete = True
            elif (
                    state in NEAR_GOAL_RETURN_STATES
                    and self._round_trip_phase == 'outbound_tracking'):
                check_near_goal_stop = True

        if check_near_goal_stop:
            start_return = self._try_arm_near_goal_return(state)

        if start_return:
            if state == 'GOAL_REACHED':
                self._publish_status(
                    'OUTBOUND_GOAL_REACHED',
                    'planning a return path to the saved start pose',
                )
            self._try_start_return_plan()
        elif round_trip_complete:
            origin = self._round_trip_origin
            self._publish_status(
                'ROUND_TRIP_COMPLETE',
                f'returned to ({origin[0]:.2f}, {origin[1]:.2f})',
            )
            self.get_logger().info('Round trip complete at the saved start')

    def _competition_follower_status(self, state):
        """Advance only after the follower accepts and finishes one leg."""
        if state == 'PATH_READY':
            with self._lock:
                if self._mission_phase != 'path_published':
                    return
                self._mission_phase = 'tracking'
                leg_name = COMPETITION_LEG_NAMES[self._mission_leg_index]
            self._publish_status(
                'MISSION_LEG_TRACKING', f'{leg_name}; follower accepted path'
            )
            return

        with self._lock:
            tracking = self._mission_phase == 'tracking'
        if not tracking:
            return
        if state == 'GOAL_REACHED':
            self._begin_mission_dwell('GOAL_REACHED')
        elif state in NEAR_GOAL_RETURN_STATES:
            self._try_accept_mission_terminal_stop(state)

    def _try_accept_mission_terminal_stop(self, stop_state):
        """Treat an unrecoverable stop near this leg's goal as arrival."""
        if stop_state not in NEAR_GOAL_RETURN_STATES:
            return False
        with self._lock:
            if self._mission_phase != 'tracking':
                self._pending_mission_terminal_state = None
                return False
            goal = self._mission_active_goal
            leg_index = self._mission_leg_index
        if goal is None or leg_index < 0:
            return False

        start, failure = self._localization_start()
        if start is None:
            with self._lock:
                if self._mission_phase == 'tracking':
                    self._pending_mission_terminal_state = stop_state
            self._publish_status(
                'MISSION_WAITING_FOR_NEAR_GOAL_CHECK',
                f'{COMPETITION_LEG_NAMES[leg_index]}; '
                f'{stop_state}; {failure}',
            )
            return False

        distance = math.hypot(start[0] - goal[0], start[1] - goal[1])
        with self._lock:
            self._pending_mission_terminal_state = None
            if self._mission_phase != 'tracking':
                return False
            if distance > self._mission_near_goal_distance:
                self._mission_phase = 'failed'
                failed = True
            else:
                failed = False
        leg_name = COMPETITION_LEG_NAMES[leg_index]
        if failed:
            self._publish_status(
                'MISSION_STOPPED_SHORT',
                f'{leg_name}; {stop_state}; goal distance={distance:.2f} m '
                f'> {self._mission_near_goal_distance:.2f} m',
            )
            return False

        return self._begin_mission_dwell(
            f'{stop_state}; goal distance={distance:.2f} m'
        )

    def _begin_mission_dwell(self, reason):
        """Hold the completed parking pose before starting the next leg."""
        with self._lock:
            if self._mission_phase != 'tracking':
                return False
            self._mission_phase = 'dwell'
            self._mission_dwell_started = (
                self.get_clock().now().nanoseconds * 1.0e-9
            )
            self._pending_mission_terminal_state = None
            leg_name = COMPETITION_LEG_NAMES[self._mission_leg_index]
        self._publish_status(
            'MISSION_WAYPOINT_REACHED',
            f'{leg_name}; {reason}; dwell={self._mission_dwell_sec:.2f} s',
        )
        return True

    def _competition_mission_tick(self):
        """Start and sequence START-A-B-START from quality-gated AMCL."""
        if self._start_mode != 'competition':
            return
        with self._lock:
            phase = self._mission_phase
            leg_index = self._mission_leg_index
            dwell_started = self._mission_dwell_started

        if phase == 'waiting_for_start':
            start, failure = self._localization_start()
            if start is None:
                self._publish_mission_wait(
                    'MISSION_WAITING_FOR_START', failure
                )
                return
            position_error = math.hypot(
                start[0] - self._mission_start[0],
                start[1] - self._mission_start[1],
            )
            yaw_error = abs(math.atan2(
                math.sin(start[2] - self._mission_start[2]),
                math.cos(start[2] - self._mission_start[2]),
            ))
            if (
                    position_error > self._mission_start_position_tolerance
                    or yaw_error > self._mission_start_yaw_tolerance):
                self._publish_mission_wait(
                    'MISSION_WAITING_FOR_START',
                    f'AMCL offset position={position_error:.2f}/'
                    f'{self._mission_start_position_tolerance:.2f} m, '
                    f'yaw={math.degrees(yaw_error):.1f}/'
                    f'{math.degrees(self._mission_start_yaw_tolerance):.1f} '
                    'deg',
                )
                return
            self._start_competition_leg(0, start)
            return

        if phase == 'dwell':
            now = self.get_clock().now().nanoseconds * 1.0e-9
            if dwell_started is None:
                return
            if now - dwell_started < self._mission_dwell_sec:
                return
            if leg_index >= len(self._mission_goals) - 1:
                with self._lock:
                    if self._mission_phase != 'dwell':
                        return
                    self._mission_phase = 'complete'
                    self._mission_dwell_started = None
                self._publish_status(
                    'MISSION_COMPLETE',
                    'START -> PARKING_A -> PARKING_B -> START complete',
                )
                self.get_logger().info('Competition parking mission complete')
                return
            with self._lock:
                if self._mission_phase != 'dwell':
                    return
                self._mission_phase = 'waiting_for_next_start'
                self._mission_dwell_started = None
                self._mission_last_wait_detail = None
            phase = 'waiting_for_next_start'

        if phase == 'waiting_for_next_start':
            start, failure = self._localization_start()
            if start is None:
                self._publish_mission_wait(
                    'MISSION_WAITING_FOR_NEXT_START', failure
                )
                return
            self._start_competition_leg(leg_index + 1, start)

    def _publish_mission_wait(self, state, detail):
        """Publish changing wait reasons without flooding the status topic."""
        key = (state, detail)
        with self._lock:
            if self._mission_last_wait_detail == key:
                return
            self._mission_last_wait_detail = key
        self._publish_status(state, detail)

    def _start_competition_leg(self, leg_index, start):
        """Plan one leg from the car's current AMCL pose."""
        goal = self._mission_goals[leg_index]
        leg_name = COMPETITION_LEG_NAMES[leg_index]
        with self._lock:
            if self._mission_phase not in {
                    'waiting_for_start', 'waiting_for_next_start'}:
                return False
            self._mission_phase = 'planning'
            self._mission_leg_index = leg_index
            self._mission_active_goal = goal
            self._pending_mission_terminal_state = None
            self._mission_last_wait_detail = None
        self._publish_status(
            'MISSION_LEG_PLANNING',
            f'{leg_name}; start=({start[0]:.2f}, {start[1]:.2f}), '
            f'goal=({goal[0]:.2f}, {goal[1]:.2f})',
        )
        self._request_plan(
            f'competition_{leg_name}',
            start_override=start,
            goal_override=goal,
        )
        return True

    def _try_arm_near_goal_return(self, stop_state):
        """Accept a terminal follower stop within the near-goal radius."""
        if stop_state not in NEAR_GOAL_RETURN_STATES:
            return False
        with self._lock:
            if self._round_trip_phase != 'outbound_tracking':
                self._pending_near_goal_stop_state = None
                return False
            goal = self._round_trip_outbound_goal
        if goal is None:
            return False

        start, failure = self._localization_start()
        if start is None:
            with self._lock:
                self._pending_near_goal_stop_state = stop_state
            self._publish_status(
                'WAITING_FOR_NEAR_GOAL_CHECK',
                f'{stop_state}; {failure}',
            )
            return False

        distance = math.hypot(start[0] - goal[0], start[1] - goal[1])
        with self._lock:
            self._pending_near_goal_stop_state = None
            if (
                    self._round_trip_phase != 'outbound_tracking'
                    or distance > self._return_near_goal_distance):
                return False
            self._round_trip_phase = 'return_waiting_for_start'
        self._publish_status(
            'OUTBOUND_NEAR_GOAL_STOP',
            f'{stop_state}; goal distance={distance:.2f} m <= '
            f'{self._return_near_goal_distance:.2f} m; planning return',
        )
        return True

    def _try_start_return_plan(self):
        """Plan from the latest valid AMCL pose back to the saved origin."""
        if not self._return_to_start:
            return False
        with self._lock:
            if self._round_trip_phase != 'return_waiting_for_start':
                return False
            origin = self._round_trip_origin
        if origin is None:
            self._publish_status(
                'RETURN_ABORTED', 'the outbound start pose was not saved'
            )
            with self._lock:
                self._round_trip_phase = 'return_failed'
            return False

        start, failure = self._localization_start()
        if start is None:
            self._publish_status('WAITING_FOR_RETURN_START', failure)
            return False

        with self._lock:
            if self._round_trip_phase != 'return_waiting_for_start':
                return False
            self._round_trip_phase = 'return_planning'
        self._request_plan(
            'return',
            start_override=start,
            goal_override=origin,
        )
        return True

    def _localization_start(self):
        """Return a quality-gated AMCL start pose or a failure description."""
        if self._amcl_start is None or self._amcl_stamp is None:
            return None, 'no /amcl_pose received'
        stamp_seconds = (
            float(self._amcl_stamp.sec)
            + float(self._amcl_stamp.nanosec) * 1.0e-9
        )
        now_seconds = self.get_clock().now().nanoseconds * 1.0e-9
        age = now_seconds - stamp_seconds
        timeout = float(self.get_parameter('pose_timeout_sec').value)
        if age < -1.0 or age > timeout:
            return None, f'/amcl_pose is stale ({age:.2f} s old)'

        max_position = float(
            self.get_parameter('max_position_stddev').value
        )
        max_yaw = float(self.get_parameter('max_yaw_stddev').value)
        if self._amcl_position_stddev > max_position:
            return None, (
                f'position stddev {self._amcl_position_stddev:.2f} m '
                f'exceeds {max_position:.2f} m'
            )
        if self._amcl_yaw_stddev > max_yaw:
            return None, (
                f'yaw stddev {math.degrees(self._amcl_yaw_stddev):.1f} deg '
                f'exceeds {math.degrees(max_yaw):.1f} deg'
            )
        return self._amcl_start, ''

    def _request_plan(
            self,
            purpose='outbound',
            start_override=None,
            goal_override=None,
            capture_round_trip_origin=False,
            expected_generation=None):
        """Snapshot one outbound or return request and plan in background."""
        with self._lock:
            if (
                    expected_generation is not None
                    and expected_generation != self._request_generation):
                return
            goal = self._goal if goal_override is None else goal_override
            if goal is None:
                self._publish_status(
                    'WAITING_FOR_GOAL', 'set a 2D Goal Pose in RViz'
                )
                return
            if start_override is not None:
                start = tuple(start_override)
                failure = ''
            elif self._start_mode == 'preview':
                start = self._preview_start
                failure = 'set a 2D Pose Estimate in RViz'
            else:
                start, failure = self._localization_start()
            if start is None:
                self._publish_status('WAITING_FOR_START', failure)
                return

            start = tuple(start)
            goal = tuple(goal)
            if capture_round_trip_origin and self._return_to_start:
                self._round_trip_origin = start
                self._round_trip_outbound_goal = goal
                self._round_trip_phase = 'outbound_planning'

            self._request_generation += 1
            generation = self._request_generation
            if self._planning:
                self._pending_plan_request = (
                    generation,
                    purpose,
                    start_override,
                    goal_override,
                    capture_round_trip_origin,
                )
                self._publish_status(
                    'QUEUED', 'newest start/goal will run after current plan'
                )
                return
            self._planning = True

        self._publish_empty_path()
        planning_state = (
            'MISSION_LEG_PLANNING'
            if purpose.startswith('competition_') else 'PLANNING'
        )
        self._publish_status(
            planning_state,
            f'{purpose}; start=({start[0]:.2f}, {start[1]:.2f}), '
            f'goal=({goal[0]:.2f}, {goal[1]:.2f})',
        )
        worker = threading.Thread(
            target=self._run_plan,
            args=(generation, start, goal, purpose),
            daemon=True,
        )
        worker.start()

    def _plan_between(self, start, goal):
        """Run the configured behavior for one independently tracked leg."""
        if self._planning_behavior == 'legacy':
            return self._planner.plan(start, goal)
        return self._planner.plan_forward_then_final_reverse(
            start,
            goal,
            reverse_activation_radius=self._reverse_activation_radius,
            max_reverse_length=self._max_reverse_length,
        )

    def _run_plan(self, generation, start, goal, purpose='outbound'):
        """Run Hybrid A* off the executor thread and publish its result."""
        try:
            result = self._plan_between(start, goal)
            with self._lock:
                current = generation == self._request_generation
            if current:
                if result.success:
                    self._publish_result(result, purpose)
                else:
                    if purpose.startswith('competition_'):
                        state = 'MISSION_NO_PATH'
                        with self._lock:
                            self._mission_phase = 'failed'
                    else:
                        state = (
                            'RETURN_NO_PATH'
                            if purpose == 'return' else 'NO_PATH'
                        )
                    if purpose == 'return':
                        with self._lock:
                            self._round_trip_phase = 'return_failed'
                    self._publish_status(
                        state,
                        f'{result.reason}; expansions={result.expansions}, '
                        f'elapsed={result.elapsed:.2f} s',
                    )
                    self.get_logger().warning(
                        f'Hybrid A* failed: {result.reason}'
                    )
        except Exception as error:  # noqa: B902 - keep ROS alive on bad input.
            self.get_logger().error(f'Hybrid A* planning error: {error}')
            with self._lock:
                current = generation == self._request_generation
            if current:
                if purpose.startswith('competition_'):
                    state = 'MISSION_ERROR'
                    with self._lock:
                        self._mission_phase = 'failed'
                else:
                    state = (
                        'RETURN_ERROR' if purpose == 'return' else 'ERROR'
                    )
                if purpose == 'return':
                    with self._lock:
                        self._round_trip_phase = 'return_failed'
                self._publish_status(state, str(error))
        finally:
            with self._lock:
                self._planning = False
                pending = self._pending_plan_request
                self._pending_plan_request = None
            if pending is not None:
                pending_generation, *request = pending
                with self._lock:
                    still_newest = (
                        not self._planning
                        and pending_generation == self._request_generation
                    )
                if still_newest:
                    self._request_plan(
                        *request,
                        expected_generation=pending_generation,
                    )

    def _publish_result(self, result, purpose='outbound'):
        """Publish the geometric path, gear direction, and RViz markers."""
        competition_leg = purpose.startswith('competition_')
        if competition_leg:
            with self._lock:
                if self._mission_phase != 'planning':
                    return
                self._mission_phase = 'path_published'
        if self._return_to_start:
            with self._lock:
                self._round_trip_phase = (
                    'return_path_published'
                    if purpose == 'return'
                    else 'outbound_path_published'
                )
        stamp = self.get_clock().now().to_msg()
        self._path_publisher.publish(create_path_message(
            result.poses, self._map_frame, stamp
        ))
        direction_message = Int8MultiArray()
        direction_message.data = [
            1 if value >= 0.0 else -1 for value in result.directions
        ]
        self._directions_publisher.publish(direction_message)
        markers = create_path_markers(
            result.poses, result.directions, self._map_frame, stamp
        )
        if self._start_mode == 'competition':
            mission_markers = create_mission_waypoint_markers(
                self._mission_waypoints, self._map_frame, stamp
            )
            markers.markers.extend(mission_markers.markers)
        self._markers_publisher.publish(markers)
        success_state = (
            'MISSION_LEG_PATH_PUBLISHED' if competition_leg else 'SUCCEEDED'
        )
        self._publish_status(
            success_state,
            f'{purpose}; {len(result.poses)} poses, {result.length:.2f} m, '
            f'gear_changes={result.gear_changes}, '
            f'finish={result.reason}, '
            f'expansions={result.expansions}, elapsed={result.elapsed:.2f} s',
        )
        self.get_logger().info(
            f'Hybrid A* path ready: {len(result.poses)} poses, '
            f'{result.length:.2f} m, {result.gear_changes} gear changes'
        )

    def _publish_empty_path(self):
        """Remove a stale path from RViz when a new request begins."""
        stamp = self.get_clock().now().to_msg()
        self._path_publisher.publish(create_path_message(
            np.empty((0, 3)), self._map_frame, stamp
        ))
        delete_all = MarkerArray()
        marker = Marker()
        marker.action = Marker.DELETEALL
        delete_all.markers.append(marker)
        self._markers_publisher.publish(delete_all)

    def _publish_status(self, state, detail):
        """Publish a latched human-readable planning state."""
        message = String()
        message.data = f'{state}: {detail}'
        self._status_publisher.publish(message)


def main(args=None):
    """Run the Hybrid A* ROS adapter."""
    rclpy.init(args=args)
    node = None
    try:
        node = HybridAStarPlannerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
