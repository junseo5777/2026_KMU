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

"""Publish the vehicle position and forward direction for RViz."""

import math

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


BASE_FRAME = 'base_link'
MAP_FRAME = 'map'
MARKER_TOPIC = '/vehicle_marker_array'
PUBLISH_PERIOD_SEC = 0.5

# Measured KMU vehicle dimensions.
VEHICLE_LENGTH_M = 0.585
VEHICLE_WIDTH_M = 0.290
WHEELBASE_M = 0.345

# The front and rear overhangs have not been measured separately. Centering the
# body between the axles is only a visual approximation; the base_link origin
# and forward arrow remain exact.
BODY_CENTER_X_M = WHEELBASE_M / 2.0


def _new_marker(namespace, marker_id, marker_type, frame_id, stamp=None):
    marker = Marker()
    marker.header.frame_id = frame_id
    if stamp is not None:
        marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = marker_id
    marker.type = marker_type
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.frame_locked = True
    return marker


def _yaw_from_pose(pose):
    """Return planar yaw from a geometry_msgs Pose quaternion."""
    orientation = pose.orientation
    sin_yaw = 2.0 * (
        orientation.w * orientation.z
        + orientation.x * orientation.y
    )
    cos_yaw = 1.0 - 2.0 * (
        orientation.y * orientation.y
        + orientation.z * orientation.z
    )
    return math.atan2(sin_yaw, cos_yaw)


def _place_marker(marker, vehicle_pose, local_x, local_y, local_z):
    """Compose a planar vehicle pose with a marker-local offset."""
    yaw = _yaw_from_pose(vehicle_pose)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    marker.pose.position.x = (
        vehicle_pose.position.x + cos_yaw * local_x - sin_yaw * local_y
    )
    marker.pose.position.y = (
        vehicle_pose.position.y + sin_yaw * local_x + cos_yaw * local_y
    )
    marker.pose.position.z = vehicle_pose.position.z + local_z

    marker.pose.orientation.x = vehicle_pose.orientation.x
    marker.pose.orientation.y = vehicle_pose.orientation.y
    marker.pose.orientation.z = vehicle_pose.orientation.z
    marker.pose.orientation.w = vehicle_pose.orientation.w


def create_vehicle_marker_array(
    vehicle_pose=None,
    frame_id=BASE_FRAME,
    stamp=None,
):
    """Return body, direction, and origin markers in the requested frame."""
    if vehicle_pose is None:
        vehicle_pose = Pose()
        vehicle_pose.orientation.w = 1.0

    marker_array = MarkerArray()

    body = _new_marker(
        'vehicle_body',
        0,
        Marker.CUBE,
        frame_id,
        stamp,
    )
    _place_marker(body, vehicle_pose, BODY_CENTER_X_M, 0.0, 0.05)
    body.scale.x = VEHICLE_LENGTH_M
    body.scale.y = VEHICLE_WIDTH_M
    body.scale.z = 0.10
    body.color.r = 0.10
    body.color.g = 0.45
    body.color.b = 1.00
    body.color.a = 0.40
    marker_array.markers.append(body)

    direction = _new_marker(
        'vehicle_direction',
        0,
        Marker.ARROW,
        frame_id,
        stamp,
    )
    _place_marker(direction, vehicle_pose, 0.0, 0.0, 0.13)
    direction.scale.x = VEHICLE_LENGTH_M + 0.12
    direction.scale.y = 0.055
    direction.scale.z = 0.11
    direction.color.r = 0.10
    direction.color.g = 1.00
    direction.color.b = 0.20
    direction.color.a = 1.00
    marker_array.markers.append(direction)

    origin = _new_marker(
        'base_link_origin',
        0,
        Marker.CYLINDER,
        frame_id,
        stamp,
    )
    _place_marker(origin, vehicle_pose, 0.0, 0.0, 0.08)
    origin.scale.x = 0.09
    origin.scale.y = 0.09
    origin.scale.z = 0.16
    origin.color.r = 1.00
    origin.color.g = 0.85
    origin.color.b = 0.00
    origin.color.a = 1.00
    marker_array.markers.append(origin)

    return marker_array


class VehicleMarkerPublisher(Node):
    """Publish a frame-locked vehicle marker array for RViz."""

    def __init__(self):
        """Configure either base-frame or map-pose marker publishing."""
        super().__init__('vehicle_marker_publisher')
        self.declare_parameter('pose_topic', '')
        self.declare_parameter('output_frame', BASE_FRAME)

        self._pose_topic = str(
            self.get_parameter('pose_topic').value
        )
        self._output_frame = str(
            self.get_parameter('output_frame').value
        )

        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._publisher = self.create_publisher(
            MarkerArray,
            MARKER_TOPIC,
            marker_qos,
        )
        self._markers = None

        if self._pose_topic:
            self._pose_subscription = self.create_subscription(
                PoseWithCovarianceStamped,
                self._pose_topic,
                self._pose_callback,
                10,
            )
            self.get_logger().info(
                f'Waiting for vehicle poses on {self._pose_topic}; '
                f'markers will use frame {self._output_frame}'
            )
        else:
            self._pose_subscription = None
            self._markers = create_vehicle_marker_array(
                frame_id=self._output_frame,
            )
            self._publish_markers()
            self.get_logger().info(
                f'Publishing vehicle markers on {MARKER_TOPIC} '
                f'in frame {self._output_frame}'
            )

        self.create_timer(PUBLISH_PERIOD_SEC, self._publish_markers)

    def _pose_callback(self, message):
        """Publish the AMCL vehicle pose directly in the map frame."""
        source_frame = message.header.frame_id.lstrip('/')
        output_frame = self._output_frame.lstrip('/')
        if source_frame != output_frame:
            self.get_logger().warning(
                f'Ignoring pose in frame {message.header.frame_id}; '
                f'expected {self._output_frame}'
            )
            return

        self._markers = create_vehicle_marker_array(
            vehicle_pose=message.pose.pose,
            frame_id=self._output_frame,
            stamp=message.header.stamp,
        )
        self._publish_markers()

    def _publish_markers(self):
        if self._markers is not None:
            self._publisher.publish(self._markers)


def main(args=None):
    """Run the vehicle marker publisher."""
    rclpy.init(args=args)
    node = VehicleMarkerPublisher()
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
