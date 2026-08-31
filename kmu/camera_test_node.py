#!/usr/bin/env python3
"""
ROS 2 차선 유지 주행 노드 (Lane Control Node)
- 카메라 이미지를 구독(Subscribe)하여 전처리 및 차선 검출 수행
- lanedetect_tuner.py 에서 확정한 전처리 로직·파라미터를 그대로 반영
- lane_detection3.py (슬라이딩 윈도우) 로직으로 조향각(Steering)/속도(Speed) 계산
- 계산된 제어값을 /xycar_motor(Float32MultiArray: [steer, speed])로 퍼블리시(Publish)
- 디버그 뷰도 튜너와 동일:
  [BEV_debug]  슬라이딩 윈도우 사각형(초록) / margin 밴드(주황)
               검출 픽셀(좌=빨강, 우=파랑) / fit 곡선 / lookahead 목표점
  [drive_view] 원본 + 차선영역 역투영(초록) + HUD(steer/speed/mode/conf) + 조향 바

조이스틱 MANUAL / AUTO 전환 (검증용):
  MANUAL  조이스틱 축으로 직접 주행. 카메라 검출과 디버그 뷰는 그대로 돌아가므로
          '수동으로 몰면서 검출이 제대로 붙는지' 를 눈으로 보고, 믿을 만해지면
          AUTO 로 넘기면 된다. 발행은 조이 타이머(joy_rate)가 맡는다.
  AUTO    기존 동작 — image_callback 이 차선 추종 제어값을 발행한다.
  두 모드 모두 같은 발행 경로(_publish_motor)를 쓰므로 조향 트림
  (xycar_steer_offset) 이 동일하게 적용된다. 수동 직진 = 자동 직진.

  안전: 시작 모드는 manual (start_mode 로 변경 가능). 모드가 바뀌는 순간 0 을
  한 번 발행해 직전 소스의 마지막 명령이 남지 않게 한다. MANUAL 중 조이 입력이
  joy_timeout 동안 끊기면 정지 명령을 유지한다 (마지막 조향 유지는 위험).

  버튼/축 인덱스 기본값은 jeju_8 teleop_key.py 관례를 따랐다. 패드가 다르면
  미할당 버튼을 누를 때마다 인덱스가 로그로 찍히니 그걸로 파라미터를 맞추면 된다.

실행:
  python3 test_camera_node.py
  python3 test_camera_node.py --ros-args -p manual_button:=3 -p auto_button:=1
  python3 test_camera_node.py --ros-args -p start_mode:=auto   # 조이 없이 기존처럼
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Joy
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

# 전처리는 preprocessing.py 가 단일 소스 (camera_node.py 와 동일 코드 경로).
# 파라미터도 그쪽 상단에 모여 있으니 튜닝 결과는 preprocessing.py 에만 반영하면 된다.
try:
    # ros2 run kmu camera_test — 패키지 모듈로 임포트되는 경우
    from kmu.lane_detection3 import LaneDetector, MARGIN_SEARCH
    from kmu import preprocessing as pp
except ImportError:
    # python3 camera_test_node.py — 스크립트로 직접 실행하는 경우
    from lane_detection3 import LaneDetector, MARGIN_SEARCH
    import preprocessing as pp


def _geom_line(res):
    """기하 위생 진단 한 줄 — 수직폭/평행비/위반항목(W=폭 P=평행 C=중심).
    conf 가 1.00 인데 fit 이 깨져 보이면 이 줄을 봐야 원인이 보인다."""
    g = res.get('geom')
    if not g:
        return "geom  -"
    return (f"geom  w{g['perp_near']:.0f} par{g['parallel_ratio']:.2f}"
            f"  {'OK' if not g['bad'] else 'BAD:' + g['bad']}")


class LaneControlNode(Node):
    def __init__(self):
        super().__init__('lane_control_node')

        self.bridge = CvBridge()
        self.detector = LaneDetector()
        self._Minv = None   # pp._Minv 캐시 (drive_view 역투영)

        # ROS 2 파라미터 선언 (외부에서 동적으로 수정 가능하도록)
        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('xycar_motor_topic', '/xycar_motor')
        self.declare_parameter('xycar_steer_scale', 1.0)
        # 조향 모터의 우측 편향 보정: 발행 조향각에서 항상 8도를 뺀다.
        self.declare_parameter('xycar_steer_offset', -8.3)
        self.declare_parameter('xycar_speed_scale', 0.4)
        self.declare_parameter('xycar_speed_offset', 0.0)
        self.declare_parameter('debug_view', True)

        # ── 조이스틱 MANUAL/AUTO ──
        # 버튼/축 인덱스는 jeju_8 teleop_key.py 와 같은 관례. 패드가 바뀌면 여기만 손댄다.
        self.declare_parameter('joy_topic', '/joy')
        self.declare_parameter('manual_button', 3)
        self.declare_parameter('auto_button', 1)
        self.declare_parameter('joy_steer_axis', 2)     # 좌우
        self.declare_parameter('joy_speed_axis', 1)     # 상하
        # ROS joy 는 스틱을 왼쪽으로 밀 때 +1 이라 조향은 부호를 뒤집어야 직관과 맞는다.
        self.declare_parameter('joy_invert_steer', True)
        self.declare_parameter('joy_invert_speed', False)
        # 스택 내부 단위(= AUTO 가 내는 값과 같은 스케일)로 잡는다. 그래야 아래
        # xycar_* 스케일 한 벌로 두 모드가 같이 보정된다.
        self.declare_parameter('joy_steer_max', 50.0)   # lane_detection3 MAX_STEER
        self.declare_parameter('joy_speed_max', 12.0)   # 자동 순항 속도대와 비슷하게
        self.declare_parameter('joy_deadzone', 0.08)
        self.declare_parameter('joy_timeout', 0.5)      # 입력 끊김 판정 (초)
        self.declare_parameter('joy_rate', 20.0)        # MANUAL 발행 주기 (Hz)
        self.declare_parameter('start_mode', 'manual')  # 'manual' | 'auto'

        camera_topic = self.get_parameter('camera_topic').value
        xycar_motor_topic = self.get_parameter('xycar_motor_topic').value
        self.xycar_steer_scale = float(self.get_parameter('xycar_steer_scale').value)
        self.xycar_steer_offset = float(self.get_parameter('xycar_steer_offset').value)
        self.xycar_speed_scale = float(self.get_parameter('xycar_speed_scale').value)
        self.xycar_speed_offset = float(self.get_parameter('xycar_speed_offset').value)
        self.debug_view = self.get_parameter('debug_view').value

        joy_topic = self.get_parameter('joy_topic').value
        self.manual_button = int(self.get_parameter('manual_button').value)
        self.auto_button = int(self.get_parameter('auto_button').value)
        self.joy_steer_axis = int(self.get_parameter('joy_steer_axis').value)
        self.joy_speed_axis = int(self.get_parameter('joy_speed_axis').value)
        self.joy_invert_steer = bool(self.get_parameter('joy_invert_steer').value)
        self.joy_invert_speed = bool(self.get_parameter('joy_invert_speed').value)
        self.joy_steer_max = float(self.get_parameter('joy_steer_max').value)
        self.joy_speed_max = float(self.get_parameter('joy_speed_max').value)
        self.joy_deadzone = float(self.get_parameter('joy_deadzone').value)
        self.joy_timeout = float(self.get_parameter('joy_timeout').value)
        joy_rate = float(self.get_parameter('joy_rate').value)
        self.mode = 'auto' if self.get_parameter('start_mode').value == 'auto' else 'manual'

        # 퍼블리셔 (차량 제어 명령)
        self.motor_pub = self.create_publisher(
            Float32MultiArray, xycar_motor_topic, 10
        )

        # 서브스크라이버 (카메라 영상) - 센서 데이터용 QoS 설정 권장
        qos_profile = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE
        )
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            qos_profile
        )

        # 조이스틱: 버튼으로 모드 전환, 축으로 MANUAL 주행
        self._joy_axes = []
        self._joy_prev_buttons = None
        self._joy_stamp = None       # None = 아직 한 번도 못 받음 → stale 취급
        self._joy_stale = True
        self._joy_cmd = (0.0, 0.0)   # HUD 표시용 (스택 내부 단위)
        self.joy_sub = self.create_subscription(Joy, joy_topic, self.joy_callback, 10)
        # MANUAL 발행은 조이 콜백이 아니라 타이머가 맡는다. joy_node 의 autorepeat
        # 설정에 상관없이 주기가 보장되고, 입력이 끊긴 것도 여기서 잡힌다.
        self.create_timer(1.0 / max(1.0, joy_rate), self.manual_tick)

        self.prev_time = time.time()
        self.proc_ms = 0.0
        self.get_logger().info(
            f"Lane Control Node Started. Subscribing to: {camera_topic}, "
            f"publishing motor: {xycar_motor_topic}"
        )
        self.get_logger().info(
            f"joy: {joy_topic}  MANUAL=button {self.manual_button} / "
            f"AUTO=button {self.auto_button}  (start: {self.mode.upper()})"
        )

    def preprocess_image(self, frame):
        """preprocessing.lane_masks 위임 → (white, yellow, bev, orig, (y0,y1))."""
        white, yellow, bev, span = pp.lane_masks(frame)
        self._Minv = pp._Minv          # drive_view 역투영용
        orig = cv2.resize(frame, (pp.SIM_W, pp.SIM_H))
        return white, yellow, bev, orig, span

    def image_callback(self, msg):
        """카메라 영상 수신 시 호출되는 콜백 함수"""
        current_time = time.time()
        dt = current_time - self.prev_time
        self.prev_time = current_time

        try:
            # 1. ROS Image -> OpenCV 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # 2. 전처리 (BEV 변환, 이진화 마스크 생성) + 3. 검출/조향 계산
        t0 = time.time()
        white_mask, yellow_mask, bev_img, orig, roi_span = self.preprocess_image(cv_image)
        result = self.detector.process(white_mask, yellow_mask, white_mask.shape, dt=dt)
        self.proc_ms = 0.8*self.proc_ms + 0.2*(time.time()-t0)*1000

        steer = float(result['steering_angle'])
        speed = float(result['target_speed'])

        # 4. 실차 모터 메시지 퍼블리시 — AUTO 일 때만.
        #    MANUAL 이어도 위의 검출/시각화는 계속 돈다. 수동 주행 중에 검출이
        #    제대로 붙는지 보는 게 이 노드의 검증 목적이라 여기서 끊으면 안 된다.
        if self.mode == 'auto':
            self._publish_motor(steer, speed)

        # 5. 디버그 뷰 (튜너와 동일 시각화)
        if self.debug_view:
            canvas, vis = self.visualize(result, white_mask, yellow_mask, orig, roi_span)
            cv2.putText(vis, f"proc {self.proc_ms:.1f}ms", (10, vis.shape[0]-48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.imshow("BEV_debug", canvas)
            cv2.imshow("drive_view", vis)
            cv2.waitKey(1)

    # ════════════════════════════════════════════════
    #  발행 / 모드 전환 / 조이스틱
    # ════════════════════════════════════════════════
    def _publish_motor(self, steer, speed):
        """스택 내부 단위 → 실차 스케일 보정 후 발행. AUTO/MANUAL 공통 경로.

        조향 트림(xycar_steer_offset)이 두 모드에 똑같이 걸려야 '수동으로 똑바로
        가는 스틱 위치' 와 '자동의 직진' 이 같은 물리 조향이 된다."""
        motor_msg = Float32MultiArray()
        motor_msg.data = [
            float(steer * self.xycar_steer_scale + self.xycar_steer_offset),
            float(speed * self.xycar_speed_scale + self.xycar_speed_offset),
        ]
        self.motor_pub.publish(motor_msg)

    def _set_mode(self, mode):
        if mode == self.mode:
            return
        self.mode = mode
        # 전환 순간 직전 소스의 마지막 명령이 그대로 남아 달리지 않도록 0 을 한 번 낸다.
        self._joy_cmd = (0.0, 0.0)
        self._publish_motor(0.0, 0.0)
        self.get_logger().info(f"[mode] → {mode.upper()}")

    def _axis(self, idx, invert):
        """데드존 적용 후 [-1,1] 로 재정규화 (데드존 경계에서 값이 튀지 않게)."""
        if idx < 0 or idx >= len(self._joy_axes):
            return 0.0
        v = float(self._joy_axes[idx])
        dz = self.joy_deadzone
        if abs(v) <= dz:
            return 0.0
        v = np.sign(v) * (abs(v) - dz) / max(1e-6, 1.0 - dz)
        return float(-v if invert else v)

    def joy_callback(self, msg: Joy):
        if self._joy_prev_buttons is None:
            self._joy_prev_buttons = [0] * len(msg.buttons)

        # rising edge 만 본다 (누르고 있는 동안 반복 전환되지 않도록)
        for idx, btn in enumerate(msg.buttons):
            was = self._joy_prev_buttons[idx] if idx < len(self._joy_prev_buttons) else 0
            if btn == 1 and was == 0:
                if idx == self.manual_button:
                    self._set_mode('manual')
                elif idx == self.auto_button:
                    self._set_mode('auto')
                else:
                    # 패드가 바뀌었을 때 인덱스를 찾는 수단. 파라미터를 이 값으로 맞추면 된다.
                    self.get_logger().info(f"[joy] button {idx} 눌림 (미할당)")
        self._joy_prev_buttons = list(msg.buttons)

        self._joy_axes = list(msg.axes)
        self._joy_stamp = time.time()

    def manual_tick(self):
        """MANUAL 모드 주행 명령 발행 (joy_rate Hz)."""
        if self.mode != 'manual':
            return

        stale = (self._joy_stamp is None
                 or (time.time() - self._joy_stamp) > self.joy_timeout)
        if stale:
            # 패드가 끊겼거나 아직 한 번도 안 왔다 → 정지. 마지막 조향/속도를
            # 유지하면 조종 불가 상태로 계속 달린다.
            if not self._joy_stale:
                self._joy_stale = True
                self.get_logger().warn("[joy] 입력 끊김 → 정지")
            self._joy_cmd = (0.0, 0.0)
            self._publish_motor(0.0, 0.0)
            return

        if self._joy_stale:
            self._joy_stale = False
            self.get_logger().info("[joy] 입력 복구")

        steer = self._axis(self.joy_steer_axis, self.joy_invert_steer) * self.joy_steer_max
        speed = self._axis(self.joy_speed_axis, self.joy_invert_speed) * self.joy_speed_max
        self._joy_cmd = (steer, speed)
        self._publish_motor(steer, speed)

    def visualize(self, res, white, yellow, orig, roi_span):
        """lanedetect_tuner.visualize 와 동일 로직."""
        det = self.detector
        h, w = white.shape

        # ── [BEV_debug] ──
        canvas = cv2.cvtColor(np.clip(white.astype(np.int16)+yellow, 0, 255).astype(np.uint8),
                              cv2.COLOR_GRAY2BGR)
        # 검출 픽셀
        if det._left_px is not None:
            lx, ly = det._left_px
            canvas[np.clip(ly,0,h-1), np.clip(lx,0,w-1)] = (0, 0, 255)
        if det._right_px is not None:
            rx, ry = det._right_px
            canvas[np.clip(ry,0,h-1), np.clip(rx,0,w-1)] = (255, 100, 0)
        # 슬라이딩 윈도우 사각형
        for (x0, y0, x1, y1) in det._dbg_rects:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 200, 0), 1)
        # margin 밴드
        for side, fit in (('left', res['left_fit']), ('right', res['right_fit'])):
            if det._dbg_search.get(side) == 'margin' and fit is not None:
                for y in range(0, h, 8):
                    cx = int(np.polyval(fit, y))
                    cv2.line(canvas, (cx-MARGIN_SEARCH, y), (cx-MARGIN_SEARCH, y+4), (255,180,0), 1)
                    cv2.line(canvas, (cx+MARGIN_SEARCH, y), (cx+MARGIN_SEARCH, y+4), (255,180,0), 1)
        # fit 곡선
        self.draw_fit(canvas, res['left_fit'],  (0, 255, 255))
        self.draw_fit(canvas, res['right_fit'], (255, 255, 0))
        # lookahead 목표점 + 차량중심
        y_eval = int(h * (1.0 - res['lookahead']))
        if res['left_fit'] is not None and res['right_fit'] is not None:
            tx = int((np.polyval(res['left_fit'], y_eval)+np.polyval(res['right_fit'], y_eval))/2)
            cv2.circle(canvas, (tx, y_eval), 6, (0,0,255), -1)
            cv2.line(canvas, (tx, y_eval), (w//2, h-1), (0,0,255), 1)
        cv2.line(canvas, (w//2, 0), (w//2, h), (120,120,120), 1)
        # 탐색 모드 + 좌우 쌍 붕괴 폐기 누적 (REJ 가 계속 오르면 검출이 반대쪽 차선으로
        # 드리프트하고 있다는 뜻 — 전처리/노출 쪽을 봐야 함)
        lbl = f"L:{det._dbg_search['left']} R:{det._dbg_search['right']}"
        if res.get('pair_reject'):
            lbl += f"  REJ:{res['pair_reject']}"
        cv2.putText(canvas, lbl, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

        # ── [drive_view] ── 원본 + 차선영역 역투영 + HUD
        vis = orig.copy()
        y0r, y1r = roi_span
        if res['left_fit'] is not None and res['right_fit'] is not None:
            ys = np.arange(0, h, 6)
            lpts = np.array([[np.polyval(res['left_fit'], y),  y] for y in ys], np.float32)
            rpts = np.array([[np.polyval(res['right_fit'], y), y] for y in ys[::-1]], np.float32)
            poly = np.vstack([lpts, rpts]).reshape(-1,1,2)
            poly_roi = cv2.perspectiveTransform(poly, self._Minv).astype(np.int32)
            poly_roi[:,0,1] += y0r
            layer = vis.copy()
            cv2.fillPoly(layer, [poly_roi], (0,180,0))
            vis = cv2.addWeighted(layer, 0.3, vis, 0.7, 0)

        steer, speed = res['steering_angle'], res['target_speed']

        # 모드 배너 — 지금 바퀴를 돌리고 있는 게 누구인지가 제일 먼저 보여야 한다.
        if self.mode == 'manual':
            js, jv = self._joy_cmd
            banner = (f"MANUAL(joy) steer {js:+6.1f} speed {jv:4.1f}"
                      + ("  [JOY LOST]" if self._joy_stale else ""))
            banner_col = (0, 0, 255) if self._joy_stale else (0, 165, 255)
        else:
            banner = "AUTO(lane)"
            banner_col = (0, 255, 0)
        cv2.putText(vis, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(vis, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_col, 2)

        # MANUAL 에서도 검출은 계속 도니, 아래 steer/speed 는 '자동이라면 냈을 값'이다.
        lead = "steer" if self.mode == 'auto' else "want "
        for i, tline in enumerate([
            f"{lead} {steer:+6.1f}   speed {speed:4.1f}",
            f"mode  {res['driving_mode']:<9} lane {res['current_lane']}",
            f"conf  {res['lane_confidence']:.2f}  detected {res['lane_detected']}",
            f"curv  {res['curvature']:+.5f}  bend {res['bend_ahead']:.0f}",
            _geom_line(res),
        ]):
            cv2.putText(vis, tline, (10, 50+22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3)
            cv2.putText(vis, tline, (10, 50+22*i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
        # 조향 바
        bw_, bh_, cy = 300, 16, vis.shape[0]-24
        cx = vis.shape[1]//2
        cv2.rectangle(vis, (cx-bw_//2, cy-bh_//2), (cx+bw_//2, cy+bh_//2), (200,200,200), 1)
        fill = int(np.clip(steer, -100, 100)/100 * (bw_//2))
        col = (0,128,255) if abs(steer) > 60 else (0,255,0)
        cv2.rectangle(vis, (cx, cy-bh_//2+2), (cx+fill, cy+bh_//2-2), col, -1)
        cv2.line(vis, (cx, cy-bh_//2-3), (cx, cy+bh_//2+3), (255,255,255), 1)

        return canvas, vis

    def draw_fit(self, canvas, fit, color):
        """차선 곡선을 디버그 뷰에 그리는 함수"""
        if fit is None:
            return
        h = canvas.shape[0]
        pts = [(int(np.polyval(fit, y)), y) for y in range(0, h, 4)]
        for i in range(len(pts)-1):
            if 0 <= pts[i][0] < canvas.shape[1]:
                cv2.line(canvas, pts[i], pts[i+1], color, 2)


def main(args=None):
    rclpy.init(args=args)
    node = LaneControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()