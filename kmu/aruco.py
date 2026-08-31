#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool


CAMERA_TOPIC = '/camera/image_raw/compressed'
OUTPUT_TOPIC = '/aruco_detected'

TARGET_ID = 9

DETECT_FRAMES = 3
LOST_FRAMES = 3


class ArucoDetectorNode(Node):

    def __init__(self):
        super().__init__('aruco_detector_node')

        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36H11
        )

        parameters = cv2.aruco.DetectorParameters()

        self.detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters
        )

        self.subscription = self.create_subscription(
            CompressedImage,
            CAMERA_TOPIC,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.publisher = self.create_publisher(
            Bool,
            OUTPUT_TOPIC,
            10
        )

        self.detected = False

        self.detect_count = 0
        self.lost_count = 0

        self.get_logger().info(
            f'AprilTag detector started | target ID = {TARGET_ID}'
        )


    def image_callback(self, msg):

        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        corners, ids, rejected = self.detector.detectMarkers(frame)

        target_found = False

        if ids is not None:
            ids = ids.flatten()

            if TARGET_ID in ids:
                target_found = True


        # ==========================================
        # 마커 발견
        # ==========================================

        if target_found:

            self.detect_count += 1
            self.lost_count = 0

            if not self.detected and self.detect_count >= DETECT_FRAMES:

                self.detected = True
                self.detect_count = 0

                self.get_logger().info(
                    f'TARGET DETECTED | ID={TARGET_ID}'
                )


        # ==========================================
        # 마커 없음
        # ==========================================

        else:

            self.lost_count += 1
            self.detect_count = 0

            if self.detected and self.lost_count >= LOST_FRAMES:

                self.detected = False
                self.lost_count = 0

                self.get_logger().info(
                    f'TARGET LOST | ID={TARGET_ID}'
                )


        # ==========================================
        # 현재 상태 퍼블리시
        # ==========================================

        output = ArucoDetected()
        output.detected = self.detected

        self.publisher.publish(output)


def main(args=None):

    rclpy.init(args=args)

    node = ArucoDetectorNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()