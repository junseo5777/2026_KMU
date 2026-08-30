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

"""
순수 IMU 적분만으로 odom->base_link TF를 발행하는 dead-reckoning 노드.

GPS 등 외부 절대 위치 보정을 전혀 쓰지 않으므로 시간이 지나면 위치가 드리프트되는
것이 정상 동작이다. 그 드리프트는 gmcl_py가 라이다-지도 매칭으로 만든 map->odom
보정치로 상쇄해준다 (odom 프레임은 "로컬에서는 매끄럽지만 전역적으로는 틀어질 수
있는" dead-reckoning 프레임이면 충분하다).
"""
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


def yaw_to_quaternion(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ImuOdomNode(Node):
    def __init__(self):
        super().__init__("imu_odom_node")

        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("odom_frame_id", "odom")
        self.declare_parameter("base_frame_id", "base_link")
        self.declare_parameter("calib_duration", 2.0)  # 이 시간 동안 정지시켜두고 바이어스 추정

        self.odom_frame_id = self.get_parameter("odom_frame_id").value
        self.base_frame_id = self.get_parameter("base_frame_id").value
        self.calib_duration = self.get_parameter("calib_duration").value

        self.tf_broadcaster = TransformBroadcaster(self)

        # dead-reckoning 상태 (odom 프레임 기준)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.last_stamp: float | None = None

        # 정지 상태에서 추정하는 센서 바이어스
        self.calibrating = True
        self.calib_start_stamp: float | None = None
        self.calib_samples: list[tuple[float, float, float]] = []
        self.gyro_bias_z = 0.0
        self.accel_bias_x = 0.0
        self.accel_bias_y = 0.0

        imu_topic = self.get_parameter("imu_topic").value
        self.create_subscription(Imu, imu_topic, self._on_imu, qos_profile_sensor_data)

        self.get_logger().info(
            f"IMU dead-reckoning 시작 ({imu_topic}). "
            f"바이어스 추정을 위해 {self.calib_duration:.1f}초간 차량을 정지 상태로 두세요."
        )

    def _on_imu(self, msg: Imu):
        stamp = msg.header.stamp
        t = stamp.sec + stamp.nanosec * 1e-9

        if self.calibrating:
            self._accumulate_calibration(msg, t)
            return

        if self.last_stamp is None:
            self.last_stamp = t
            return

        dt = t - self.last_stamp
        self.last_stamp = t
        if dt <= 0.0 or dt > 0.5:
            # 타임스탬프 역행/큰 갭(재연결 등) -> 이번 샘플은 버리고 다음부터 재개
            return

        gyro_z = msg.angular_velocity.z - self.gyro_bias_z
        self.yaw = normalize_angle(self.yaw + gyro_z * dt)

        ax_body = msg.linear_acceleration.x - self.accel_bias_x
        ay_body = msg.linear_acceleration.y - self.accel_bias_y

        c, s = math.cos(self.yaw), math.sin(self.yaw)
        ax_odom = c * ax_body - s * ay_body
        ay_odom = s * ax_body + c * ay_body

        self.vx += ax_odom * dt
        self.vy += ay_odom * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

        self._broadcast_tf(stamp)

    def _accumulate_calibration(self, msg: Imu, t: float):
        if self.calib_start_stamp is None:
            self.calib_start_stamp = t
        self.calib_samples.append((
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
        ))

        if t - self.calib_start_stamp >= self.calib_duration:
            n = len(self.calib_samples)
            self.gyro_bias_z = sum(s[0] for s in self.calib_samples) / n
            self.accel_bias_x = sum(s[1] for s in self.calib_samples) / n
            self.accel_bias_y = sum(s[2] for s in self.calib_samples) / n
            self.calibrating = False
            self.last_stamp = t
            self.get_logger().info(
                f"바이어스 보정 완료: gyro_z={self.gyro_bias_z:.5f} rad/s, "
                f"accel_x={self.accel_bias_x:.4f}, accel_y={self.accel_bias_y:.4f} m/s^2"
            )

    def _broadcast_tf(self, stamp):
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame_id
        t.child_frame_id = self.base_frame_id
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(self.yaw)
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = ImuOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
