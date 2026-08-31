#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vehicle_node.py — 전방 차량 YOLO 검출 노드. /vehicle_ahead (Bool) 발행.

왜 camera_node 와 분리했는가 (중요):
  camera_node.image_cb 는 차선 파이프라인 전체를 콜백 안에서 동기로 돈다.
  거기에 36ms 짜리 YOLO 를 얹으면 조향 경로에 그만큼의 지연이 직렬로 붙는다.
  (camera_node.py:75-78 — dt 가 어긋나 조향 전체가 조용히 틀렸던 전례가 있다.)
  차량 검출은 조향과 달리 30Hz 가 필요 없는 저주기 판정이므로 별도 프로세스에서
  10Hz 로 돌리는 편이 모든 면에서 낫다.

왜 CamData 가 아니라 std_msgs/Bool 인가:
  1) CamData 는 이 워크스페이스 밖(~/Kookmin_sim/custom_msgs)에 있어 필드를
     추가하려면 외부 패키지 수정 + 재빌드가 필요하다.
  2) /cam_data 에 퍼블리셔가 둘이 되면 서로 덮어쓴다.
  3) std_msgs 만 쓰므로 xycar_msgs/custom_msgs 없이도 이 노드는 단독 실행·검증이
     가능하다(camera_node 는 불가).

발행하는 값은 원시 검출이 아니라 **래치된 게이트**다. 근거는
vehicle_detection.VehicleDetector 독스트링 참조 — 추월 트리거가 car_count 2→1
'엣지' 라서 순간값 AND 는 시점이 어긋난다.

실행:
    python3 vehicle_node.py --ros-args -p debug_view:=true
    python3 vehicle_node.py --ros-args -p image_topic:=/usb_cam/image_raw/front
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import time

try:
    from kmu.vehicle_detection import VehicleDetector, DEFAULT_MODEL
except ImportError:                     # 패키지 밖에서 단독 실행할 때
    from vehicle_detection import VehicleDetector, DEFAULT_MODEL


# camera_node.py 와 동일 규약: 센서 원본은 BEST_EFFORT/depth=1.
# depth=1 은 여기서 특히 중요하다 — 추론이 프레임 주기보다 느리므로 큐를 두면
# 낡은 프레임이 쌓여 판정이 계속 과거를 본다.
QOS_SENSOR = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# 판정 결과는 track_drive 의 QOS_CMD 와 맞춘다(RELIABLE/depth=1).
QOS_CMD = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class VehicleNode(Node):
    def __init__(self):
        super().__init__('vehicle_node')

        p = self.declare_parameters('', [
            ('image_topic', '/image_raw'),
            ('out_topic',   '/vehicle_ahead'),
            ('model_path',  DEFAULT_MODEL),
            # 파인튜닝 모델이라 원본 해상도 그대로면 충분하다(held-out 재현율
            # 100%/95%). COCO 사전학습을 쓸 때만 960 업스케일이 필요했다 —
            # 폴백이 걸리면 960 으로 올릴 것. 실측표는 검출기 독스트링.
            ('imgsz',       640),
            ('conf',        0.30),     # w/h 필터가 오탐을 막으므로 임계를 낮춰 recall 확보
            ('rate_hz',     10.0),
            ('threads',     2),        # 차선 파이프라인이 나머지 코어를 쓰도록 상한
            ('confirm_n',   2),        # 최근 window 중 2프레임 검출 시 확정
            ('window',      3),
            ('hold_sec',    1.5),      # 영역 체류가 0.6초로 짧은 사례가 있어 1.5→2.0
            ('debug_view',  False),
            # ── 추월영역(zone). 640x480 픽셀 좌표, 음수면 그 축 비활성 ──
            # car_1~5 실차 영상 라벨링으로 확정된 값이다(FP 7/7 제거, 사용자 지정
            # '판단 시작' 프레임 4/4 통과). 크기(box_h)가 아니라 접지점 y_bot(거리축)
            # + 박스중심 xc(측방축) + 종횡비 w/h(오탐축)로 자르는 근거는
            # vehicle_detection.VehicleDetector 독스트링 참조.
            # ★ 카메라를 재장착하면 y_bot_min/xc 는 반드시 재측정.
            ('y_bot_min',   335.0),
            ('xc_min',      170.0),
            ('xc_max',      470.0),
            ('wh_min',      0.6),
        ])
        g = {q.name: q.value for q in p}

        self.debug_view = bool(g['debug_view'])
        self.bridge = CvBridge()
        self.det = VehicleDetector(
            model_path=g['model_path'], imgsz=g['imgsz'], conf=g['conf'],
            confirm_n=g['confirm_n'], window=g['window'],
            hold_sec=g['hold_sec'], threads=g['threads'],
            y_bot_min=g['y_bot_min'], xc_min=g['xc_min'], xc_max=g['xc_max'],
            wh_min=g['wh_min'],
        )

        self._frame     = None   # 최신 프레임만 보관 (콜백은 저장만, 추론은 타이머)
        self._seq       = 0      # 도착 카운터
        self._seq_done  = -1     # 마지막으로 추론한 카운터 → 같은 프레임 재추론 방지
        self._n_pub     = 0
        self._t_log     = time.monotonic()   # 실측 게시율 산출용
        self._t_err     = 0.0

        # 구독과 추론 타이머를 다른 콜백 그룹에 두고 MultiThreadedExecutor 로 돈다.
        # 단일 스레드 실행기로 돌렸더니 추론(60~90ms) 동안 image_cb 가 아예 실행되지
        # 못해 실측 게시율이 10Hz 목표 대비 6.1Hz 로 떨어졌고, 처리하는 프레임도
        # 그만큼 낡은 것이었다(depth=1 의 '최신 프레임' 보장이 무의미해짐).
        self.cbg_sub   = MutuallyExclusiveCallbackGroup()
        self.cbg_timer = MutuallyExclusiveCallbackGroup()

        self.sub = self.create_subscription(
            Image, g['image_topic'], self.image_cb, QOS_SENSOR,
            callback_group=self.cbg_sub)
        self.pub = self.create_publisher(Bool, g['out_topic'], QOS_CMD)
        self.create_timer(1.0 / float(g['rate_hz']), self.tick,
                          callback_group=self.cbg_timer)

        self.get_logger().info(
            f"vehicle_node 활성화 | {g['image_topic']} → {g['out_topic']} | "
            f"imgsz={g['imgsz']} conf={g['conf']} {g['rate_hz']}Hz "
            f"threads={g['threads']}")
        # 추월영역이 비활성이면 원거리/측방 차량까지 True 가 되므로, 그 상태로
        # 주행에 물리는 사고가 없도록 시작 시점에 반드시 눈에 띄게 남긴다.
        if self.det.zone_active:
            self.get_logger().info(self.det.zone_desc())
        else:
            self.get_logger().warn(self.det.zone_desc())

    def image_cb(self, msg):
        # 콜백에서는 디코드만 한다. 추론을 여기서 돌리면 구독 스레드가 막혀
        # depth=1 의 '최신 프레임' 보장이 무너진다.
        try:
            self._frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self._seq += 1
        except Exception as e:
            self.get_logger().warn(f"image decode fail: {e}")

    def tick(self):
        if self._frame is None or self._seq == self._seq_done:
            return                      # 새 프레임 없음 → CPU 낭비 안 함
        self._seq_done = self._seq
        frame = self._frame

        try:
            out = self.det.update(frame)
        except Exception as e:
            # 프레임 단위 예외로 노드를 죽이지 않되 조용히 묻히지도 않게
            # (camera_node.py 와 같은 1초 throttle 규약)
            t = time.monotonic()
            if t - self._t_err > 1.0:
                self._t_err = t
                self.get_logger().error(f"yolo fail: {type(e).__name__}: {e}")
            return

        self.pub.publish(Bool(data=bool(out['vehicle_ahead'])))

        self._n_pub += 1
        if self._n_pub % 10 == 0:       # 10Hz 기준 1초에 한 번
            t = time.monotonic()
            hz = 10.0 / max(t - self._t_log, 1e-6)   # rate_hz 는 상한일 뿐,
            self._t_log = t                          # 실제 값은 추론 시간에 묶인다
            # xc/y_bot 은 zone 임계를 정하는 근거값, w/h 는 시점(후면≈1.2 /
            # 측면≈1.8) 관찰용이다. 셋 다 판정에는 쓰이지 않는다 — 데이터를
            # 모아 측면 판별 도입 여부를 정하기 위한 로그.
            self.get_logger().info(
                f"AHEAD={out['vehicle_ahead']!s:<5} raw={out['raw']!s:<5} "
                f"n={out['count']}(+{out['count_out']}out) "
                f"conf={out['best_conf']:.2f} xc={out['best_xc']:3.0f} "
                f"ybot={out['best_ybot']:3.0f} wh={out['best_wh']:.2f} "
                f"{out['latency_ms']:4.0f}ms {hz:4.1f}Hz")

        if self.debug_view:
            cv2.imshow('vehicle', self.det.draw(frame, out))
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = VehicleNode()
    executor = MultiThreadedExecutor(num_threads=2)   # 구독 / 추론 분리
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        # SIGTERM 으로 죽을 때는 rclpy 시그널 핸들러가 이미 컨텍스트를 내린 뒤라
        # 무조건 shutdown() 하면 'rcl_shutdown already called' 로 종료가 지저분해진다.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

# 추월영역(y_bot_min / xc_min / xc_max) 캘리브레이션 요약:
#   1) 장애물 차를 자기 차선 전방, 추월을 시작하고 싶은 실제 거리에 정지시켜 녹화
#   2) zone 을 끈 채 `python3 vehicle_detection.py <영상> --save out/` 로 y_bot 읽기
#   3) 옆차선에도 세워 xc 를 읽고 xc_min/xc_max 밴드 결정
# 전체 절차와 확인 방법은 vehicle_detection.py 의 오프라인 검증 섹션에 있다.