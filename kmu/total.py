#!/usr/bin/env python3
#-*-coding:utf-8-*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray

from collections import deque

import time

class PubSubErp(Node):
    def __init__(self):
        super().__init__('PubSubErp_node')
        qos_profile = QoSProfile(depth=10)
        qos_profile2 = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.teleop_sub = self.create_subscription(Twist, "/cmd_vel", self.teleop_callback, qos_profile)
        self.state_sub = self.create_subscription(Twist, "/erp_writes", self.state_callback, qos_profile2)

        self.motor_pub = self.create_publisher(Float32MultiArray, "/xycar_motor", qos_profile2)


        self.mode = 'MANUAL'
        self.manual_speed = 0.0
        self.manual_steer = 0.0
        self.auto_speed = 0.0
        self.auto_steer = 0.0

        self.auto_speed_cmd = 0.0
        self.parameter = 0.2


        self.COMMANDS = {
            (49.5, 99.0): 'MANUAL',
            (-500.0, -1000.0): 'AUTO',
        }

    def teleop_callback(self, data: Twist):
        key = (data.linear.x, data.angular.z)
        cmd = self.COMMANDS.get(key)

        print(cmd)
        if cmd == 'MANUAL':
            self.mode = 'MANUAL'
            self.manual_speed = 0.0
            self.manual_steer = 0.0
            return
        elif cmd == 'AUTO':
            if self.mode != 'AUTO':
                self.auto_speed_cmd = 0.0
            self.mode = 'AUTO'
            self.manual_speed = 0.0
            self.manual_steer = 0.0
            return

        else: # 모드 전환용 숫자가 아니라면 
            self.manual_speed = data.linear.x
            self.manual_steer = -data.angular.z
        
        if self.mode == 'MANUAL':
            self.send_to_motor(self.manual_speed, self.manual_steer, mode="MANUAL")


    def state_callback(self, data: Twist):
        if self.mode == 'AUTO':
            target_speed = data.linear.x
            target_steer = data.angular.z

            p=self.parameter
            self.auto_speed_cmd += p * (target_speed - self.auto_speed_cmd)
            if self.auto_speed_cmd > 0.9 * target_speed:
                self.auto_speed_cmd = target_speed
            self.get_logger().info(f"[AUTO] 수신={data.linear.x}, {data.angular.z} | 전송={self.auto_speed_cmd}, {target_steer}")
            self.send_to_motor(self.auto_speed_cmd, target_steer, "AUTO")



    def send_to_motor(self, speed: float, steer: float, mode: str):
        msg = Float32MultiArray()
        msg.data = [float(steer)-8.3, float(speed)]

        self.motor_pub.publish(msg)
        self.get_logger().info(
            f"{mode} → /xycar_motor publish: steer={msg.data[0]}, speed={msg.data[1]}"
        )



def main(args=None):
    rclpy.init(args=args)
    node = PubSubErp()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("노드 종료")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()