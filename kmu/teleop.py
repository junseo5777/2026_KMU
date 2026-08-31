#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class TeleopTwistJoy(Node):
    def __init__(self):
        super().__init__('teleop_twist_joy')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        # 초기 선형 및 각속도 (매개변수로 설정 가능)
        self.speed = self.declare_parameter('speed', 20.0).value
        self.turn  = self.declare_parameter('turn', 60.0).value

        # 연속 이동 제어를 위한 축 인덱스 (조이스틱에 따라 수정)
        self.linear_axis = 1   # 보통 왼쪽 스틱의 상하축
        self.angular_axis = 3  # 보통 오른쪽 스틱의 좌우축

        # 마지막 명령을 저장할 Twist 메시지
        self.last_twist = Twist()

        # 10Hz로 last_twist를 발행하는 타이머
        self.create_timer(0.1, self.publish_command)

        # 이전 버튼 상태를 저장하여 rising edge 감지를 위한 변수
        self.last_buttons = None
        self.auto_mode_active = False 
        self.auto_twist = Twist()

        # 모드 전환 명령 (버튼 번호 : (linear.x, angular.z))
        self.modeBindings = {
            0: (-500, -1000),  # AUTO mode (버튼 A)
            3: (49.5, 99)      # MANUAL mode (버튼 Y)
        }
        msg = """
Joystick Teleoperation Node:
----------------------------
[이동 제어]
  - 왼쪽 조이스틱:
      축 {linear_axis} (상하) → 선형 이동, 축 {angular_axis} (좌우) → 회전 제어

[모드 전환 명령 (버튼)]
  - 버튼 A: AUTO mode (linear.x=-500, angular.z=-1000)
  - 버튼 Y: MANUAL mode (linear.x=49.5, angular.z=99)

CTRL-C로 종료하세요.
"""
        self.get_logger().info(msg.format(linear_axis=self.linear_axis, angular_axis=self.angular_axis))

    def publish_command(self):
        if self.auto_mode_active:
            self.publisher_.publish(self.auto_twist)
        else:
            self.publisher_.publish(self.last_twist)


    def joy_callback(self, msg: Joy):
    # 최초 콜백에서 이전 버튼 상태 초기화
        if self.last_buttons is None:
            self.last_buttons = [0] * len(msg.buttons)

        # 우선 모드 전환 명령 처리 (rising edge 감지)
        for btn_idx, mode in self.modeBindings.items():
            if btn_idx < len(msg.buttons) and msg.buttons[btn_idx] == 1 and self.last_buttons[btn_idx] == 0:
                if btn_idx == 0:  # A: AUTO 모드
                    self.auto_mode_active = True
                    self.auto_twist = Twist()
                    self.auto_twist.linear.x = float(mode[0])
                    self.auto_twist.angular.z = float(mode[1])
                    self.get_logger().info(f"AUTO 모드 진입한다옹: linear.x={self.auto_twist.linear.x}, angular.z={self.auto_twist.angular.z}")
                    self.last_twist = Twist()
                    self.last_buttons = list(msg.buttons)
                    return
                    
                elif btn_idx == 3:  # Y: MANUAL 모드
                    self.auto_mode_active = False
                    self.get_logger().info("MANUAL 모드 진입한다옹")
                    self.last_twist = Twist()
                    self.last_twist.linear.x = float(mode[0])
                    self.last_twist.angular.z = float(mode[1])
                    self.last_buttons = list(msg.buttons)
                    return

        # 연속 이동: 조이스틱 축 입력으로 Twist 업데이트
        if not self.auto_mode_active:
            twist = Twist()
            if len(msg.axes) > self.linear_axis:
                twist.linear.x = self.speed * msg.axes[self.linear_axis]
            if len(msg.axes) > self.angular_axis:
                twist.angular.z = self.turn * msg.axes[self.angular_axis]
            self.last_twist = twist

        # 버튼 상태 업데이트 (다음 콜백을 위해)
        self.last_buttons = list(msg.buttons)

def main(args=None):
    rclpy.init(args=args)
    node = TeleopTwistJoy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Teleop Joy 노드가 사용자의 인터럽트로 종료됨")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()




