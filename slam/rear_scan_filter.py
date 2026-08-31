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

"""Crop vehicle-occluded rear LiDAR measurements for SLAM."""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


INPUT_SCAN_TOPIC = '/scan'          # LiDAR 원본 topic
OUTPUT_SCAN_TOPIC = '/scan_slam'    # mapping/localization용 topic

# ROS laser frame: 0 degrees forward, positive angles counter-clockwise.
FRONT_MIN_ANGLE_DEG = -90.0         # 사용할 우측 한계 각도
FRONT_MAX_ANGLE_DEG = 90.0          # 사용할 좌측 한계 각도
FRONT_MIN_ANGLE_RAD = math.radians(FRONT_MIN_ANGLE_DEG)  # rad 변환값
FRONT_MAX_ANGLE_RAD = math.radians(FRONT_MAX_ANGLE_DEG)  # rad 변환값


def front_sector_bounds(angle_min, angle_increment, beam_count):
    """Return the half-open beam interval inside the forward 180 degrees."""
    if beam_count <= 0:
        raise ValueError('beam_count must be positive')
    if angle_increment <= 0.0:
        raise ValueError('angle_increment must be positive')

    first = math.ceil(
        (FRONT_MIN_ANGLE_RAD - angle_min) / angle_increment
    )
    last = math.floor(
        (FRONT_MAX_ANGLE_RAD - angle_min) / angle_increment
    )
    first = max(0, min(beam_count, first))
    stop = max(first, min(beam_count, last + 1))
    if first == stop:
        raise ValueError('LaserScan does not contain the front sector')
    return first, stop


def shifted_stamp(stamp, offset_sec):
    """Return normalized sec/nanosec values after a positive offset."""
    total_nanoseconds = (
        int(stamp.sec) * 1_000_000_000
        + int(stamp.nanosec)
        + round(offset_sec * 1_000_000_000)
    )
    return divmod(total_nanoseconds, 1_000_000_000)


class RearScanFilter(Node):
    """Publish a SLAM scan containing only the unobstructed front sector."""

    def __init__(self):
        """Configure the raw-scan subscriber and cropped-scan publisher."""
        super().__init__('rear_scan_filter')
        self._logged_first_scan = False
        # The raw YDLIDAR publisher is Best Effort. Offer the filtered scan as
        # Reliable so Cartographer, RViz, and Best Effort odometry consumers
        # can all connect without a reliability mismatch.
        output_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._publisher = self.create_publisher(
            LaserScan,
            OUTPUT_SCAN_TOPIC,
            output_qos,
        )
        self.create_subscription(
            LaserScan,
            INPUT_SCAN_TOPIC,
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f'Filtering {INPUT_SCAN_TOPIC} to {OUTPUT_SCAN_TOPIC}; '
            f'keeping {FRONT_MIN_ANGLE_DEG:.1f} to '
            f'{FRONT_MAX_ANGLE_DEG:.1f} degrees'
        )

    def _scan_callback(self, message):
        filtered_message = LaserScan()
        first, stop = front_sector_bounds(
            message.angle_min,
            message.angle_increment,
            len(message.ranges),
        )
        filtered_message.header.frame_id = message.header.frame_id
        stamp_sec, stamp_nanosec = shifted_stamp(
            message.header.stamp,
            first * message.time_increment,
        )
        filtered_message.header.stamp.sec = stamp_sec
        filtered_message.header.stamp.nanosec = stamp_nanosec
        filtered_message.angle_min = (
            message.angle_min + first * message.angle_increment
        )
        filtered_message.angle_max = (
            message.angle_min + (stop - 1) * message.angle_increment
        )
        filtered_message.angle_increment = message.angle_increment
        filtered_message.time_increment = message.time_increment
        filtered_message.scan_time = message.scan_time
        filtered_message.range_min = message.range_min
        filtered_message.range_max = message.range_max
        filtered_message.ranges = message.ranges[first:stop]
        if len(message.intensities) == len(message.ranges):
            filtered_message.intensities = message.intensities[first:stop]
        self._publisher.publish(filtered_message)

        if not self._logged_first_scan:
            invalid_count = sum(
                not math.isfinite(distance)
                or distance < message.range_min
                or distance > message.range_max
                for distance in filtered_message.ranges
            )
            self.get_logger().info(
                f'First filtered scan: {len(message.ranges)} -> '
                f'{len(filtered_message.ranges)} front beams, '
                f'{invalid_count} invalid'
            )
            self._logged_first_scan = True


def main(args=None):
    """Run the rear scan filter node."""
    rclpy.init(args=args)
    node = RearScanFilter()
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
