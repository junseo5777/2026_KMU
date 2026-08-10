"""Mask vehicle-occluded rear LiDAR measurements for SLAM."""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


INPUT_SCAN_TOPIC = '/scan'          # LiDAR 원본 topic
OUTPUT_SCAN_TOPIC = '/scan_slam'    # mapping/localization용 topic

# ROS laser frame: 0 degrees forward, positive angles counter-clockwise.
FRONT_MIN_ANGLE_DEG = -90.0         # 사용할 우측 한계 각도
FRONT_MAX_ANGLE_DEG = 90.0          # 사용할 좌측 한계 각도
FRONT_MIN_ANGLE_RAD = math.radians(FRONT_MIN_ANGLE_DEG)  # rad 변환값
FRONT_MAX_ANGLE_RAD = math.radians(FRONT_MAX_ANGLE_DEG)  # rad 변환값
MASKED_RANGE = math.inf             # 제거할 beam에 넣는 무효 거리


def is_front_angle(angle):
    """Return whether an angle lies in the vehicle's unobstructed front half."""
    normalized_angle = math.atan2(math.sin(angle), math.cos(angle))
    return FRONT_MIN_ANGLE_RAD <= normalized_angle <= FRONT_MAX_ANGLE_RAD


def mask_rear_ranges(angle_min, angle_increment, ranges):
    """Return scan ranges with the rear half-plane replaced by infinity."""
    filtered_ranges = list(ranges)
    for index in range(len(filtered_ranges)):
        angle = angle_min + index * angle_increment
        if not is_front_angle(angle):
            filtered_ranges[index] = MASKED_RANGE
    return filtered_ranges


class RearScanFilter(Node):
    """Publish a SLAM scan containing only the unobstructed front sector."""

    def __init__(self):
        super().__init__('rear_scan_filter')
        self._logged_first_scan = False
        self._publisher = self.create_publisher(
            LaserScan,
            OUTPUT_SCAN_TOPIC,
            qos_profile_sensor_data,
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
        filtered_message.header = message.header
        filtered_message.angle_min = message.angle_min
        filtered_message.angle_max = message.angle_max
        filtered_message.angle_increment = message.angle_increment
        filtered_message.time_increment = message.time_increment
        filtered_message.scan_time = message.scan_time
        filtered_message.range_min = message.range_min
        filtered_message.range_max = message.range_max
        filtered_message.ranges = mask_rear_ranges(
            message.angle_min,
            message.angle_increment,
            message.ranges,
        )
        filtered_message.intensities = message.intensities
        self._publisher.publish(filtered_message)

        if not self._logged_first_scan:
            masked_count = sum(
                math.isinf(distance)
                for distance in filtered_message.ranges
            )
            self.get_logger().info(
                f'First filtered scan: {len(message.ranges)} beams, '
                f'{masked_count} rear/invalid beams are infinite'
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
