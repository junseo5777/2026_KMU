#!/usr/bin/env python3
# -*- coding: utf-8 -*- 1
import rclpy, time, cv2, os
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from visualization_msgs.msg import MarkerArray
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from .dwa import DWA
from custom_msgs.msg import LidarData, CamData
from .state import StateMachine
from .visualization import RvizVisualizer
from .ridar import LidarPerception
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        Interpreter = None


LEFT        = 0   # 왼쪽 옆
RIGHT       = 4   # 오른쪽 옆
# REAR_RIGHT  = 5   # 후면 오른쪽
# REAR_CENTER = 6   # 후면 중앙
# REAR_LEFT   = 7   # 후면 왼쪽

# 센서 원본 구독용. BEST_EFFORT 구독자는 Unity RELIABLE와
# usb_cam BEST_EFFORT 발행자 모두와 호환된다. 제어에는 최신 프레임만 유효하다.
QOS_SENSOR = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

MODEL_WIDTH = 160
MODEL_HEIGHT = 120
MODEL_MASK_ROWS = 45
MODEL_STEERING_SCALE = 60.0
MODEL_STEERING_OFFSET = -8.3
MODEL_ERROR_LOG_INTERVAL = 5.0

# ── TFLite 기본 모델 변경 위치 ──
# 모델을 바꿔가며 테스트할 때는 아래 파일명 한 줄만 바꾸면 된다.
# 선택 가능: best_model_old.tflite / best_model_new.tflite /
#              best_model_shift.tflite
DEFAULT_MODEL_FILENAME = 'best_model_shift.tflite'
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'models',
    DEFAULT_MODEL_FILENAME,
)

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
        self.declare_parameter('model_path', DEFAULT_MODEL_PATH)
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('model_speed', 17.0)
        self.declare_parameter('model_steering_gain', 1.0)
        self.declare_parameter('model_num_threads', 2)

        self.bridge = CvBridge()
        self.model_speed = float(self.get_parameter('model_speed').value)
        self.model_steering_gain = float(
            self.get_parameter('model_steering_gain').value)
        self._last_model_error_time = 0.0
        self._load_steering_model(
            str(self.get_parameter('model_path').value),
            int(self.get_parameter('model_num_threads').value),
        )

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT)
        # ROS2 Publisher & Subscriber 설정
        # self.motor_pub = self.create_publisher(Float32MultiArray, 'xycar_motor', 10)
        self.dwa_marker_pub = self.create_publisher(MarkerArray, '/dwa_markers', 10)
        self.lidar_marker_pub = self.create_publisher(MarkerArray, '/lidar_markers', 10)
        #################
        self.ultra_sub = self.create_subscription(Int32MultiArray,'/xycar_ultrasonic',self.ultra_callback,10)
        #####################
        self.sub_front = self.create_subscription(
            CamData, '/cam_data', self.cam_callback, 10)
        self.scan_subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.state_pub = self.create_publisher(Float32MultiArray, "/xycar_motor",10)
        self.get_logger().info("Track Driver Node Initialized")
        camera_topic = str(self.get_parameter('camera_topic').value)
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.model_image_callback, QOS_SENSOR)
        self.get_logger().info(
            f"TFLite steering active: camera={camera_topic}, "
            f"model={self.model_path}, speed={self.model_speed:.1f}")
        # 상수값 및 초기값 설정
        self.image = None  # 카메라 토픽 데이터를 저장할 변수
        self.state_msg = Float32MultiArray()  # 모터토픽 메시지
        self.lidar_ranges = None
        self.scan_msg = None   # scan_callback은 이 값만 저장, 처리는 lidar_scan_process에서
        self.dwa = DWA()
        self.lidar = LidarPerception()
        self.s = 0.
        self.speed = 0.               # 속도
        self.steer = 0.               # 조향각
        self.lane_steer = 0.          # 레인 조향
        self.traffic3 = "GREEN"          
        self.traffic4 = 'GREEN'                    
        self.obs = []                 # 장애물 좌표 리스트 [(x, y), ...]
        self.point = []
        self.lane_detected = False    # 차선 검출 여부
        self.stop_line_detected = False  # 정지선 검출 여부
        self.police_car = False       # 경찰차 존재 여부
        self.overtake_mode = False    # 추월 모드 진입 여부
        self.overtake_done = False    # 추월 완료 여부
        self.current_lane = None
        self.ob_dir = None
        self.car_location = None
        self.front_clear = False      # 전방 라이다 클리어 여부 (추월 복귀 트리거)
        self.overtake_phase = None    # 추월 단계: None / 'CHANGE_OUT' / 'CHANGE_BACK'
        self.overtake_origin_lane = None  # 추월 시작 시 원래 차선
        self.passing_lane = None      # 추월용으로 빠져나갈 옆 차선
        self.car_count = None
        self.prev_car_count = None
        self.target_speed = 0.
        self.left_turn_trigger = None
        self.cone_lidar_value = None
        self.car_lidar_value = None
        self.x  = None
        self.y = None
        self.origin_x = None
        self.origin_y = None
        self.abs_overtake = False
        self.go_front_clear = False
        self.dwa_marker_frame = 'laser_frame'
        self.dwa_marker_lifetime = 0.2
        self.lidar_marker_angle_offset = np.deg2rad(-90.0)
        self.lidar_marker_max_range = 2.0
        self.cone_count = None
        self.return_count = None
        self.ret = False
        self.ultra = None
        self.cone_detect = False
        self.car = None
        self.d = None
        self.left = None
        self.right = None
        self.cv_steer = None
       # roi
        self.dwa_roi_half_width = 0.7
        self.dwa_roi_min_y = 0.02
        self.dwa_roi_max_y = 1.7

        self.visualizer = RvizVisualizer(
            self,
            self.dwa_marker_pub,
            self.lidar_marker_pub,
            marker_frame=self.dwa_marker_frame,
            marker_lifetime=self.dwa_marker_lifetime,
            roi_half_width=self.dwa_roi_half_width,
            roi_min_y=self.dwa_roi_min_y,
            roi_max_y=self.dwa_roi_max_y,
        )

        StateMachine.__init__(self)
        # self.create_timer(0.1,self.run)

    def _load_steering_model(self, model_path, num_threads):
        if Interpreter is None:
            raise RuntimeError(
                'TFLite runtime is missing. Install tflite-runtime before '
                'starting nuclear.')

        self.model_path = os.path.abspath(os.path.expanduser(model_path))
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f'TFLite steering model not found: {self.model_path}')
        if num_threads < 1:
            raise ValueError('model_num_threads must be at least 1')

        self.interpreter = Interpreter(
            model_path=self.model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        signatures = self.interpreter.get_signature_list()
        signature = signatures.get('serving_default', {})
        if signature.get('inputs') != ['img_in']:
            raise RuntimeError(
                f'Unexpected TFLite input signature: {signature!r}')
        if 'n_outputs0' not in signature.get('outputs', []):
            raise RuntimeError(
                f'TFLite steering output n_outputs0 is missing: {signature!r}')
        self.model_infer = self.interpreter.get_signature_runner(
            'serving_default')

    @staticmethod
    def _preprocess_model_image(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb, (MODEL_WIDTH, MODEL_HEIGHT), interpolation=cv2.INTER_AREA)
        rgb[:MODEL_MASK_ROWS, :, :] = 0
        model_input = rgb.astype(np.float32) / 255.0
        return np.expand_dims(model_input, axis=0)

    def _log_model_error(self, message):
        now = time.monotonic()
        if now - self._last_model_error_time >= MODEL_ERROR_LOG_INTERVAL:
            self.get_logger().error(message)
            self._last_model_error_time = now

    def model_image_callback(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            model_input = self._preprocess_model_image(bgr)
            outputs = self.model_infer(img_in=model_input)
            normalized_steer = float(outputs['n_outputs0'][0, 0])
        except Exception as error:
            self._log_model_error(f'TFLite steering inference failed: {error}')
            return

        if not np.isfinite(normalized_steer):
            self._log_model_error(
                f'TFLite returned invalid steering: {normalized_steer!r}')
            return

        # n_outputs0은 [-1, 1] 정규화 조향값이다. 학습 런타임 계약의
        # 역변환을 적용해 /xycar_motor의 원시 조향 단위로 복원한다.
        normalized_steer = float(np.clip(normalized_steer, -1.0, 1.0))
        motor_steer = (
            normalized_steer * MODEL_STEERING_SCALE + MODEL_STEERING_OFFSET)
        motor_steer *= self.model_steering_gain

        # 모델의 n_outputs1(throttle)은 학습 계약상 항상 0이므로 무시한다.
        # drive()는 기존 CV 조향에 1.5 보정을 더하므로, 이미 원시 단위인
        # 모델 출력은 중복 스케일링 없이 직접 발행한다.
        self.state_msg.data = [float(motor_steer), self.model_speed]
        self.state_pub.publish(self.state_msg)

    def cam_callback(self, msg):
        self.target_speed = msg.target_speed
        self.current_lane = msg.current_lane
        # self.traffic = msg.traffic_light_color
        # self.stop_line_detected = msg.stop_line_detected # 정지선
        # self.left_turn_trigger = msg.left_turn_trigger # 좌회전
        self.lane_detected = msg.lane_detected
        self.lane_steer = msg.steering_angle
        # if self.state == 'START' or self.state == 'GO':
        #     self.traffic3 = self.traffic
        # else:
        #     self.traffic4 = self.traffic

    def ultra_callback(self, msg):
        d = msg.data 
        self.left      = d[LEFT]
        self.right     = d[RIGHT]

            
            
    def overtake(self):
        if self.overtake_phase is None:
            if self.current_lane == None:
                    return
            if self.current_lane == 1 and self.ob_dir == "right": # 1에 오른쪽 차
                self.overtake_phase = 'straight'
                self.passing_lane = 2
                self.overtake_origin_lane = 1
            elif self.current_lane == 2 and self.ob_dir == 'left': # 2에 왼쪽 차
                self.overtake_phase = 'straight'
                self.passing_lane = 1
                self.overtake_origin_lane = 2
            else:
                self.overtake_origin_lane = self.current_lane
                self.passing_lane = 2 if self.current_lane == 1 else 1
                self.overtake_phase = 'CHANGE_OUT'
        
                
        if self.overtake_phase == 'straight':
            self.speed = 10.5
            if self.current_lane == 1 and self.ob_dir == "right": # 1에 오른쪽 차
                self.passing_lane = 2
                self.overtake_origin_lane = 1
            elif self.current_lane == 2 and self.ob_dir == 'left': # 2에 왼쪽 차
                self.passing_lane = 1
                self.overtake_origin_lane = 2
            if self.front_clear:
                self.overtake_phase = 'CHANGE_BACK'
            else: 
                self.steer = self.lane_steer
        
        OFF = 40
        if self.overtake_phase == 'CHANGE_OUT':
            sign = 1.0 if self.passing_lane > self.overtake_origin_lane else -1.0 # 차선변경
            if self.current_lane !=self.passing_lane:
                self.steer = sign * OFF + self.lane_steer*0.3
            elif self.current_lane == self.passing_lane: # 차선변경 완료
                self.overtake_phase = 'straight'

        elif self.overtake_phase == 'CHANGE_BACK':
            sign = -1. if self.overtake_origin_lane > self.passing_lane else 1.
            self.speed = 9.5
            self.steer = sign * OFF + self.lane_steer*0.4
            # 원래 차선 복귀 완료 -> 추월 종료
            if self.current_lane == self.passing_lane:
                self.overtake_done = True
                
            
    def is_in_dwa_roi(self, ox, oy):
        return (
            abs(ox) < self.dwa_roi_half_width
            and self.dwa_roi_min_y < oy < self.dwa_roi_max_y
        )

    def scan_callback(self, msg):
        self.scan_msg = msg


    def lidar_scan_process(self):
        msg = self.scan_msg
        if msg is None:
            return
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
            self.origin_x = []
            self.origin_y = []
            self.x = []
            self.y = []
            self.point = []
            self.visualizer.publish_lidar_markers([], msg.header.frame_id)
            return

        ranges = ranges[valid_mask]
        angles = angles[valid_mask]
        points = np.column_stack((-ranges * np.cos(angles), -ranges * np.sin(angles)))
        self.origin_x = points[:, 0].astype(float).tolist()
        self.origin_y = points[:, 1].astype(float).tolist()
        

        self.point = []
        for ox, oy in zip(self.origin_x, self.origin_y):
            ox = float(ox)
            oy = float(oy)
            if self.is_in_dwa_roi(ox, oy):
                self.point.append((ox, oy))
        self.visualizer.publish_lidar_markers(self.point, msg.header.frame_id)

        clusters = self.lidar.clusterize(self.point)
        cone_detect, car_detect, car_center_point= self.lidar.classfi(clusters)
        self.cone_lidar_value = cone_detect
        self.car_lidar_value = car_detect
        self.get_logger().info(f"""
con= {self.cone_lidar_value}, car= {self.car_lidar_value}
""")      ######### 코드 터져서 바꾼거 민찬이형 확인해주세요 ########
        if car_center_point is not None and len(car_center_point) > 0:
            pts = np.asarray(car_center_point, dtype=float).reshape(-1, 2)
            self.x = pts[:, 0].tolist()
            self.y = pts[:, 1].tolist()
        else:
            self.x = []
            self.y = []
        

    def lidar_prepro(self):
        if self.car_lidar_value:
            if self.left and self.right:
                self.d = self.left if self.right > self.left else self.right
            if self.d:
                self.ultra = True if self.d < 40 else False 
            if self.x and self.y:
                current_location = np.array([0,1]) # 바로앞을 벡터로 잡음
                car_vec = np.array([
                                    np.sign(self.x[0]),
                                    np.sign(self.y[0])
                                ])
                self.car_location = np.cross(current_location,car_vec)
                if self.car_location > 0: # 좌측
                    self.ob_dir = "left"
                elif np.isclose(self.car_location, 0.0):
                    self.ob_dir = 'front'
                else: 
                    self.ob_dir = 'right'

        if self.x and self.y:
            FRONT_X = 2.   # 좌우 폭(±, m)
            FRONT_Y = 1.5   # 전방 거리(m)
            front_clear = True 
            for px, py in zip(self.x, self.y):
                if abs(px) < FRONT_X and 0.0 < py < FRONT_Y or self.ultra :
                    front_clear = False
                    break
            self.front_clear = front_clear
        else:
            self.front_clear = True

        if self.cone_lidar_value == 'rubbercone':
            self.cone_detect = True
        else:
            self.cone_detect = False

    def dwa_steer(self):
        obs = []
        if self.origin_x and self.origin_y:
            for ox, oy in zip(self.origin_x, self.origin_y):
                ox = float(ox)
                oy = float(oy)
                if self.is_in_dwa_roi(ox, oy):
                    obs.append((ox, oy))

        obs = np.asarray(obs, dtype=float) if obs else None
        best_path, best_steer, all_path = self.dwa.dwa(self.s, obs)
        self.visualizer.publish_dwa_visualization(
            best_path,
            all_path,
            obs,
            self.dwa.end_safe_dist,
        )
        self.s = best_steer
        return self.s

    def drive(self, angle, speed):
        self.state_msg.data = [float(angle)*1.7, float(speed)]

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
        node.drive(angle=0., speed=0.)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
