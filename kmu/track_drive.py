#!/usr/bin/env python3
# -*- coding: utf-8 -*- 1
import rclpy, time, cv2, os
import numpy as np
from collections import deque
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from visualization_msgs.msg import MarkerArray
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from .dwa import DWA
from custom_msgs.msg import LidarData, CamData
from .state import StateMachine
from .visualization import RvizVisualizer
from .ridar import LidarPerception
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


LEFT        = 0   # 왼쪽 옆
RIGHT       = 4   # 오른쪽 옆
# REAR_RIGHT  = 5   # 후면 오른쪽
# REAR_CENTER = 6   # 후면 중앙
# REAR_LEFT   = 7   # 후면 왼쪽

# 센서 원본 구독용 (Unity 발행 RELIABLE 에 맞춤)
QOS_SENSOR = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

#  추월 로직 파라미터
#
# 진입 판단은 ridar.LidarPerception 이 확정한 'car' 하나로 끝난다.
# TTC / 접근속도 / 차로점유(car_lane) / 전방 클리어 / 초음파 측면거리 판단은
# 전부 제거했다. YOLO 트리거는 lidar_prepro 의 진입 조건 한 줄에 붙인다.

# 추월은 '옆 차선으로 한 번 나가는 것'이 전부다. 정상 완료 시 복귀는 없다.
LANE_CHANGE_MIN_SEC = 0.5  # 조향 개시 후 최소 유지 시간 (commit)
LANE_CHANGE_MAX_SEC = 0.6  # 타임아웃 — 차선변경 상태에 갇히지 않게

# ── 차선 이탈 abort: 유일한 복귀 경로 ────────────────────────────
LANE_LOST_FRAMES = 6      # current_lane 미확정이 이만큼 이어지면 복귀
RECOVER_MAX_SEC = 2.0     # 복귀 조향을 물고 있을 최대 시간

# 차선변경 직후 같은 차를 다시 물어 반대로 되돌아가는 핑퐁을 막는다.
OVERTAKE_COOLDOWN_SEC = 3.8

# SLAM에서 사용하는 실측 차량 기준. base_link는 뒷바퀴 축 중앙이며
# LiDAR는 그보다 0.410 m 앞에 있다. nuclear.py의 DWA 기본값은 바꾸지 않고
# track_drive에서만 rear-axle 경로 모드를 활성화한다.
TRACK_DWA_WHEEL_BASE = 0.345
TRACK_DWA_LIDAR_FORWARD_OFFSET = 0.410
# DWA가 장애물을 실제 위치보다 5 cm 가까이 보도록 당긴다.
# 객체 분류에는 원본 좌표를 쓰고, DWA 비용/마커에만 적용한다.
DWA_OBSTACLE_Y_PULL = 0.05

# ══════════════════════════════════════════════════════════════════
#  RViz 시각화 스위치
# ══════════════════════════════════════════════════════════════════
# 마커 발행은 스캔 주기마다 MarkerArray 를 새로 만들어 채운다.
#   lidar 마커 : DWA ROI 안 점 + DWA/객체분류 ROI 박스
#   dwa 마커   : 후보 경로 13개 x 6점 + 장애물 원 + DWA ROI 박스
# 주행에 쓰이지 않는 순수 진단 출력이라 연산이 아쉬우면 끈다.
# 표시 범위는 아래 DWA/객체분류 ROI 설정값과 자동으로 연동된다.
USE_RVIZ = True

CHANGE_OFF = 25.0         # 차선변경 조향 오프셋
OVERTAKE_SPEED = 16. # 추월  속도

USE_ULTRA = False
# 처리 결과/제어 명령용 (내부 노드 간)
QOS_CMD = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
class TrackDriverNode(Node, StateMachine):


    def __init__(self):

        super().__init__('driver')
        self.get_logger().info('----- Xycar self-driving node started -----')
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT)
        # ROS2 Publisher & Subscriber 설정
        if USE_RVIZ:
            self.dwa_marker_pub = self.create_publisher(MarkerArray, '/dwa_markers', 10)
            self.lidar_marker_pub = self.create_publisher(MarkerArray, '/lidar_markers', 10)
        else:
            self.dwa_marker_pub = None
            self.lidar_marker_pub = None
        if USE_ULTRA:
            self.ultra_sub = self.create_subscription(
                Int32MultiArray, '/xycar_ultrasonic', self.ultra_callback, 10)
        else:
            self.ultra_sub = None
        self.sub_front = self.create_subscription(
            CamData, '/cam_data', self.cam_callback, 10)
        self.scan_subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.state_pub = self.create_publisher(Twist, "/erp_writes",10)
        self.get_logger().info("Track Driver Node Initialized")
        # 상수값 및 초기값 설정
        self.image = None  # 카메라 토픽 데이터를 저장할 변수
        self.state_msg = Twist()  # 모터토픽 메시지
        self.lidar_ranges = None
        self.scan_msg = None   # scan_callback은 이 값만 저장, 처리는 lidar_scan_process에서
        self.dwa = DWA(
            wheel_base=TRACK_DWA_WHEEL_BASE,
            sensor_forward_offset=TRACK_DWA_LIDAR_FORWARD_OFFSET,
        )
        self.lidar = LidarPerception()
        self.s = 0.
        self.speed = 0.               # 속도
        self.steer = 0.               # 조향각
        self.lane_steer = 0.          # 레인 조향
        self.lane_one = None
        self.traffic = 'NONE'
        self.prev_traffic = 'NONE'    # 직전 프레임 신호 — LEFT->NONE 엣지 판정용
        self.left_pending = False     # LEFT->NONE 엣지 latch (LANE_DRIVE 에서 소비)
        self.left_done = False        # 하드코딩 좌회전 완료 여부
        self.left_t_trigger = 0.      # 좌회전 하드코딩 시작 시각 (time.time)
        self.obs = []                 # 장애물 좌표 리스트 [(x, y), ...]
        self.point = []
        self.regular_point = []
        self.lane_detected = False    # 차선 검출 여부
        self.stop_line_detected = False  # 정지선 검출 여부
        self.police_car = False       # 경찰차 존재 여부
        self.overtake_mode = False    # 추월 모드 진입 여부
        self.overtake_done = False    # 추월 완료 여부
        self.overtake_start_t = None  # 차선변경 조향 개시 시각 (time.time)
        self.overtake_end_t = 0.      # 마지막 차선변경 종료 시각 (쿨다운용)
        self.overtake_recover = False # 차선 이탈로 원 차선 복귀 중인지
        self.recover_start_t = None   # 복귀 진입 시각
        self.lane_lost_streak = 0     # current_lane 미확정 연속 프레임 수
        self.current_lane = None
        self.overtake_origin_lane = None  # 차선변경 시작 시 원래 차선
        self.passing_lane = None      # 추월용으로 빠져나갈 옆 차선
        self.target_speed = 0.
        self.left_turn_trigger = None
        self.tim = False
        self.cone_lidar_value = None
        self.car_lidar_value = None
        self.x  = None                # 라이다가 'car' 로 본 클러스터 중심 x [m]
        self.y = None                 # 같은 클러스터 중심 y (전방거리) [m]
        self.abs_overtake = False
        self.dwa_marker_frame = 'laser_frame'
        self.dwa_marker_lifetime = 0.2
        self.lidar_marker_angle_offset = np.deg2rad(-90.0)
        self.lidar_marker_max_range = 3.0
        self.cone_count = None
        self.return_count = None
        self.dynamic = False
        self.shortcut = False
        self.ret = False
        self.cone_detect = False
        self.left = None              # 초음파 좌측 [cm] (현재 판단에는 미사용)
        self.right = None             # 초음파 우측 [cm] (현재 판단에는 미사용)
        self.shortcut_detection_history = deque(maxlen=3)
        self.prev_static = deque(maxlen=2)
        self.shortcut_detected = False
        self.prev_shortcut_detected = False
        self.shortcut = False
        self.dwa_time = False
        self.static_detected = False
        self.shortcut_detected_on = False
        self.checkerboard_detected = False
       # roi
        self.dwa_roi_half_width = 0.7   #원래 0.7
        self.dwa_roi_min_y = 0.05       #원래 0.02
        # rear-axle 경로 끝(0.69 m) + end_safe_dist(0.33 m) + 여유(0.03 m)
        self.dwa_roi_max_y = 1.2

        self.regular_roi_half_width = 0.7
        self.regular_roi_min_y = 0.02
        self.regular_roi_max_y = 2.5

        self.cone_cam_detected = None
        self.lidar_static_value = None

        self.visualizer = RvizVisualizer(
            self,
            self.dwa_marker_pub,
            self.lidar_marker_pub,
            marker_frame=self.dwa_marker_frame,
            marker_lifetime=self.dwa_marker_lifetime,
            roi_half_width=self.dwa_roi_half_width,
            roi_min_y=self.dwa_roi_min_y,
            roi_max_y=self.dwa_roi_max_y,
            regular_roi_half_width=self.regular_roi_half_width,
            regular_roi_min_y=self.regular_roi_min_y,
            regular_roi_max_y=self.regular_roi_max_y,
        ) if USE_RVIZ else None

        StateMachine.__init__(self)
        self.create_timer(0.1,self.run)


    def cam_callback(self, msg):
        self.target_speed = msg.target_speed
        self.current_lane = msg.current_lane
        self.traffic = msg.traffic_light_color
        self.lane_detected = msg.lane_detected
        self.lane_steer = msg.steering_angle
        self.dynamic = msg.dynamic_detected
        self.cone_cam_detected = msg.cone_cam_detected
        self.shortcut_edge(msg.shortcut_detected)
        self.static_detected = msg.static_detected
        self.checkerboard_detected = msg.checkerboard_detected
        self.shortcut_detected_on = msg.shortcut_on

    def ultra_callback(self, msg):
        """초음파 콜백. USE_ULTRA=False 면 구독을 안 걸므로 호출되지 않는다."""
        d = msg.data
        if len(d) <= RIGHT: 
            return
        self.left      = d[LEFT]
        self.right     = d[RIGHT]

            
            
    def overtake(self):
        now = time.time()

        if self.overtake_start_t is None:
            self.overtake_start_t = now
        elapsed = now - self.overtake_start_t

        # 차로 미확정이면 어느 쪽으로 나갈지 정할 수 없다.
        # 확정될 때까지는 차선주행 그대로 두고, 타임아웃까지 안 오면 포기한다.
        if self.overtake_origin_lane is None:
            if self.current_lane not in (1, 2):
                self.speed = self.target_speed
                self.steer = self.lane_steer
                if elapsed > LANE_CHANGE_MAX_SEC:
                    self.overtake_done = True
                return
            self.overtake_origin_lane = self.current_lane
            self.passing_lane = 2 if self.current_lane == 1 else 1
            # 최소 유지 시간·타임아웃은 '조향을 꺾기 시작한' 이 시점부터 센다.
            self.overtake_start_t = now
            elapsed = 0.

        # 1차로는 노란선 좌측, 2차로는 우측 → 번호가 커지는 쪽이 우조향(+).
        sign = 1.0 if self.passing_lane > self.overtake_origin_lane else -1.0
        self.speed = OVERTAKE_SPEED

        if self.current_lane in (1, 2):
            self.lane_lost_streak = 0
        else:
            self.lane_lost_streak += 1

        if not self.overtake_recover and self.lane_lost_streak >= LANE_LOST_FRAMES:
            self.overtake_recover = True
            self.recover_start_t = now

        # 복귀 중에는 최소 유지 시간을 보지 않는다 — 차선을 잃은 상태다.
        if self.overtake_recover:
            self.steer = -sign * CHANGE_OFF + self.lane_steer * 0.35
            if (self.current_lane == self.passing_lane
                    or now - self.recover_start_t > RECOVER_MAX_SEC):
                self.overtake_done = True
            return

        self.steer = sign * CHANGE_OFF + self.lane_steer * 0.35
        if elapsed < LANE_CHANGE_MIN_SEC:
            return
        if self.current_lane == self.passing_lane or elapsed > LANE_CHANGE_MAX_SEC:
            self.overtake_done = True

    def reset_overtake(self):
        self.overtake_mode = False
        self.overtake_done = False
        self.overtake_origin_lane = None
        self.passing_lane = None
        self.overtake_start_t = None
        self.overtake_recover = False
        self.recover_start_t = None
        self.lane_lost_streak = 0
        # 쿨다운 기준 시각 — 방금 지나친 차를 다시 물어 반대로 되돌아가는 것을 막는다.
        self.overtake_end_t = time.time()


    def is_in_dwa_roi(self, ox, oy):
        return (
            abs(ox) < self.dwa_roi_half_width
            and self.dwa_roi_min_y < oy < self.dwa_roi_max_y
        )

    def regular_roi(self, ox, oy):
        return(
            abs(ox) < self.regular_roi_half_width
            and self.regular_roi_min_y < oy < self.regular_roi_max_y
        )

    def scan_callback(self, msg):
        self.scan_msg = msg

    def lidar_scan_process(self):
        msg = self.scan_msg
        if msg is None:
            return
        # 새로 수신한 스캔은 한 번만 처리한다. 다음 /scan 콜백이 새 메시지를
        # 저장하기 전까지 타이머가 같은 스캔을 반복 처리하지 않도록 소비한다.
        self.scan_msg = None
        self.lidar.angle_increment = msg.angle_increment
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = (
            msg.angle_min +
            np.arange(len(ranges), dtype=np.float32) * msg.angle_increment +
            self.lidar_marker_angle_offset
        )

        valid_mask = (
            np.isfinite(ranges) &
            (ranges >= msg.range_min) &
            (ranges <= min(msg.range_max, self.lidar_marker_max_range))
        )
        if not np.any(valid_mask):
            self.x = []
            self.y = []
            self.point = []
            self.regular_point = []
            if self.visualizer:
                self.visualizer.publish_lidar_markers([], msg.header.frame_id)
            return

        ranges = ranges[valid_mask]
        angles = angles[valid_mask]
        points = np.column_stack((-ranges * np.cos(angles), -ranges * np.sin(angles)))
        self.point = []
        self.regular_point = []
        m_dwa = ((np.abs(points[:, 0]) < self.dwa_roi_half_width)
                 & (points[:, 1] > self.dwa_roi_min_y)
                 & (points[:, 1] < self.dwa_roi_max_y))
        m_reg = ((np.abs(points[:, 0]) < self.regular_roi_half_width)
                 & (points[:, 1] > self.regular_roi_min_y)
                 & (points[:, 1] < self.regular_roi_max_y))

        self.point = points[m_dwa].tolist()
        clusters = self.lidar.clusterize(points[m_reg])
        self.lidar_static_value = self.lidar.detect_static(points)

        if self.visualizer:
            self.visualizer.publish_lidar_markers(self.point, msg.header.frame_id)
        # ridar.classfi 가 프레임 히스토리 투표까지 마친 결과를 그대로 받는다.
        # cone/car 확정 로직은 전부 ridar 쪽에 있고 여기서 다시 세지 않는다.
        cone_detect, car_detect, car_center_point = self.lidar.classfi(clusters)
        self.cone_lidar_value = cone_detect
        self.car_lidar_value = car_detect
        if car_center_point is not None and len(car_center_point) > 0:
            pts = np.asarray(car_center_point, dtype=float).reshape(-1, 2)
            self.x = pts[:, 0].tolist()
            self.y = pts[:, 1].tolist()
        else:
            self.x = []
            self.y = []

        self.get_logger().info(f"""
con= {self.cone_lidar_value}, car= {self.car_lidar_value}
state = {self.state}
cone_cam  = {self.cone_cam_detected}
""")

    def lidar_prepro(self):
        self.cone_detect = self.cone_lidar_value == 'rubbercone'

        # ── 추월 진입 판단 ──
        # 라이다가 'car' 를 확정했고 직전 추월의 쿨다운이 끝났으면 진입한다.
        # 차로 확정 여부는 여기서 보지 않는다 — overtake() 가 확정될 때까지
        # 기다렸다가 그 시점의 차로를 원 차선으로 잡는다. 여기서 미리 잡으면
        # 진입 순간의 한 프레임짜리 차로 오판이 그대로 굳는다.
        # YOLO 트리거를 붙일 자리도 이 조건이다 (and self.car_yolo 한 줄).
        if ((not self.overtake_mode
                and self.car_lidar_value == 'car'
                and time.time() - self.overtake_end_t > OVERTAKE_COOLDOWN_SEC
                and self.state == 'LANE_DRIVE'
                and self.dynamic)
                or 
                (not self.overtake_mode
                and self.lidar_static_value == 'static'
                and time.time() -self.overtake_end_t > OVERTAKE_COOLDOWN_SEC
                and self.state == 'LANE_DRIVE'
                and self.static_detected
                )
                or
                (not self.overtake_mode
                and self.lidar_static_value == 'static'
                and time.time() -self.overtake_end_t > OVERTAKE_COOLDOWN_SEC
                and self.state == 'STATIC'
                and self.static_detected)):
            self.overtake_mode = True

    def dwa_steer(self):
        # self.point 가 이미 lidar_scan_process 에서 같은 is_in_dwa_roi 로
        # 걸러 만든 ROI 점 목록이다. 여기서 origin_x/origin_y 를 다시 훑으면
        # 스캔 점 전체에 대한 ROI 판정을 매 프레임 두 번 하는 셈이 된다.
        obs = np.asarray(self.point, dtype=float) if self.point else None
        if obs is not None:
            # 레거시 축은 +y가 차량 전방이므로 y를 줄이면 차량 쪽으로 당겨진다.
            # publish_dwa_visualization에도 이 배열을 넘겨 보정 위치를 그대로 본다.
            obs[:, 1] -= DWA_OBSTACLE_Y_PULL
        best_path, best_steer, all_path = self.dwa.dwa(self.s, obs)
        if self.visualizer:
            self.visualizer.publish_dwa_visualization(
                best_path,
                all_path,
                obs,
                self.dwa.end_safe_dist,
            )
        self.s = best_steer
        return self.s

    def drive(self, angle, speed):
        self.state_msg.linear.x = speed
        self.state_msg.angular.z = angle
        self.state_pub.publish(self.state_msg)


   
def main(args=None):
    rclpy.init(args=args)
    node = TrackDriverNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        # 사용자 인터럽트 (Ctrl+C)가 발생하면 예외를 처리합니다.
        pass
    finally:
        # 노드를 종료하고 ROS2를 정리합니다.
        if rclpy.ok():
            node.drive(angle=0., speed=0.)
        executor.shutdown()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()