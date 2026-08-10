"""Periodically save the latest SLAM occupancy grid."""

import os
import threading

from cartographer_2d_slam.map_io import save_occupancy_grid
from nav_msgs.msg import OccupancyGrid
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy


DEFAULT_MAP_OUTPUT_FILE = os.path.join(
    os.path.expanduser('~'),
    '.ros',
    'cartographer_2d_slam',
)


class AutoMapSaver(Node):
    """Save the first, periodic, and final snapshots of the latest map."""

    def __init__(self):
        super().__init__('auto_map_saver')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('output_file', DEFAULT_MAP_OUTPUT_FILE)
        self.declare_parameter('save_interval_sec', 30.0)
        self.declare_parameter('save_on_first_map', True)
        self.declare_parameter('save_on_shutdown', True)
        self.declare_parameter('free_thresh', 0.25)
        self.declare_parameter('occupied_thresh', 0.65)

        self.map_topic = self.get_parameter('map_topic').value
        self.output_file = self.get_parameter('output_file').value
        self.save_interval_sec = self.get_parameter(
            'save_interval_sec'
        ).value
        self.save_on_first_map = self.get_parameter(
            'save_on_first_map'
        ).value
        self.save_on_shutdown = self.get_parameter(
            'save_on_shutdown'
        ).value
        self.free_thresh = self.get_parameter('free_thresh').value
        self.occupied_thresh = self.get_parameter(
            'occupied_thresh'
        ).value

        if self.save_interval_sec < 0.0:
            raise ValueError('save_interval_sec must be zero or positive')

        self._lock = threading.Lock()
        self._latest_map = None
        self._map_revision = 0
        self._saved_revision = 0

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self._map_callback,
            map_qos,
        )

        if self.save_interval_sec > 0.0:
            self.create_timer(
                self.save_interval_sec,
                self._periodic_save,
            )

        self.get_logger().info(
            f'Waiting for {self.map_topic}; maps will be saved to '
            f'{self.output_file}.yaml/.pgm'
        )

    def _map_callback(self, message):
        with self._lock:
            first_map = self._latest_map is None
            self._latest_map = message
            self._map_revision += 1

        if first_map and self.save_on_first_map:
            self.save_latest(reason='first map')

    def _periodic_save(self):
        self.save_latest(reason='periodic')

    def save_latest(self, reason, force=False):
        """Write the latest received map if it has changed."""
        with self._lock:
            map_message = self._latest_map
            map_revision = self._map_revision
            already_saved = map_revision == self._saved_revision

        if map_message is None:
            if force:
                self.get_logger().warning(
                    'No /map message was received; nothing was saved'
                )
            return False
        if already_saved and not force:
            return False

        try:
            yaml_path, image_path = save_occupancy_grid(
                map_message,
                self.output_file,
                self.free_thresh,
                self.occupied_thresh,
            )
        except (OSError, ValueError) as error:
            self.get_logger().error(f'Failed to save map: {error}')
            return False

        with self._lock:
            self._saved_revision = max(
                self._saved_revision,
                map_revision,
            )
        self.get_logger().info(
            f'Saved map ({reason}): {yaml_path} and {image_path}'
        )
        return True


def main(args=None):
    """Run the automatic map saver node."""
    rclpy.init(args=args)
    node = None
    try:
        node = AutoMapSaver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            if node.save_on_shutdown:
                node.save_latest(reason='shutdown', force=True)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
