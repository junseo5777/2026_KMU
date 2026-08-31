import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

from . import preprocessing7 as pp
from .lane_detection8 import LaneDetector
# from .road_marks import RoadMarkDetector
# from .school_zone import SchoolZoneTracker
from custom_msgs.msg import CamData
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from .perception import Perception
from .lubbercone_cam import ConeCamGate
from .shortcut_cam import ShortcutCamGate
from .checkerboard_cam import CheckerCamGate

# 센서 원본 구독용 (/image_raw)
#   BEST_EFFORT: RELIABLE 퍼블리셔(Unity)와도 호환된다(구독자가 더 약한 보증을
#     요구하는 방향은 호환). 반대로 RELIABLE 구독자는 BEST_EFFORT 퍼블리셔(usb_cam
#     기본값)와 매칭이 아예 안 되므로, 양쪽 다 받으려면 BEST_EFFORT 가 유일한 선택.
#   depth=1: 콜백이 밀리면 큐에 쌓인 낡은 프레임을 뒤늦게 처리하게 된다. 제어
#     루프에는 '가장 최신 프레임' 만 의미가 있으므로 오래된 것은 버린다.
#     (기존 depth=10 은 최대 0.3초 지연된 영상으로 조향하는 상태였음)
QOS_SENSOR = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# 처리 결과/제어 명령용 (/cam_data) — track_drive.py 의 QOS_CMD 와 동일해야 매칭.
#   RELIABLE: 상태 전이(정지선/신호)는 한 프레임만 놓쳐도 시퀀스가 어긋난다.
#   depth=1: 소비자(track_drive)가 30ms 타이머로 최신값만 읽으므로 큐잉 불필요.
QOS_CMD = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


LANE_UNKNOWN = 0    # current_lane 미확정


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.bridge = CvBridge()
        self.lane   = LaneDetector()
        self.per    = Perception(logger=self.get_logger())
        self.scut  = ShortcutCamGate(logger=self.get_logger())
        self.cone   = ConeCamGate(logger=self.get_logger())
        self.ckb = CheckerCamGate(logger=self.get_logger())
        # self.rm     = RoadMarkDetector()
        # self.sz     = SchoolZoneTracker()

        self.frame_count = 0
        self._t_prev     = None     # dt 폴백용 (monotonic, 콜백 호출 시각)
        self._t_img_prev = None     # dt 주 소스 (header.stamp, 촬영 시각)
        self._dt_src     = None     # 'stamp' / 'clock' — 바뀔 때만 로그
        self._t_src_log  = 0.0      # dt 소스 로그 throttle
        self._lane_valid = False    # 노란선으로 차로가 한 번이라도 확정됐는가
        self._t_err      = 0.0      # 예외 로그 throttle
        self._cone_warn  = False    # CamData 에 cone 필드가 없을 때 1회만 경고
    

        self.sub = self.create_subscription(Image, '/image_raw', self.image_cb, QOS_SENSOR)
        self.pub = self.create_publisher(CamData, '/cam_data', QOS_CMD)

        self.get_logger().info("camera_node 활성화 (lane + 통합인지: 신호등/차량/정적장애물)")

    def image_cb(self, msg):
        try:
            # CvBridge 가 인코딩(rgb8/bgr8) 자동 처리 → 항상 BGR 로 받음
            bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().warn(f"image decode fail: {e}")
            return

        # dt = '직전 프레임 이후 실제 경과 시간'. 루프 주기를 정하는 값이 아니라
        # 관측값이다. lane_detection3 의 D항(Δerr/dt)·조향 레이트 리미터(단위/s × dt)·
        # 속도 슬루(ACCEL × dt)가 전부 dt 스케일이라 틀리면 셋이 통째로 어긋난다
        # (기존 dt=0.33 고정은 실제의 10배였고 DT_MAX=0.5 안이라 조용히 통과했다).
        #
        # 소스는 header.stamp(촬영 시각) 우선, monotonic 은 폴백:
        #   - monotonic 은 '콜백 호출 간격'이라 스케줄링이 밀리면 영상 간격이
        #     일정한데도 dt 가 부풀려진다 (지터 25% → D항 27% 잡음).
        #     촬영 시각 차이는 그 영향을 안 받는다.
        #   - 고정값을 쓰지 않는 이유: QoS 가 BEST_EFFORT+depth=1 이라 부하 시
        #     프레임 드롭이 예외가 아니라 설계 동작이고, 고정 dt 는 드롭율만큼
        #     그대로 편향된다 (10% 드롭 → 속도슬루/레이트제한 -10%).
        #     스탬프 차이는 드롭된 만큼을 정확히 반영한다.
        #   - 폴백이 필요한 이유: 시뮬레이터·일부 드라이버는 스탬프를 0 으로 채운다.
        t_now = time.monotonic()
        st = msg.header.stamp
        t_img = st.sec + st.nanosec * 1e-9
        dt = None
        if t_img > 0.0 and self._t_img_prev is not None:
            d_img = t_img - self._t_img_prev
            if 0.0 < d_img < 1.0:       # 역행(bag 루프)·정지·비정상 스탬프 배제
                dt, src = d_img, 'stamp'
        if dt is None:
            dt = 0.033 if self._t_prev is None else (t_now - self._t_prev)
            src = 'clock'
        self._t_img_prev = t_img if t_img > 0.0 else None
        self._t_prev = t_now
        if src != self._dt_src and t_now - self._t_src_log > 5.0:
            self._dt_src, self._t_src_log = src, t_now
            self.get_logger().info(
                "dt 소스: " + ("header.stamp (촬영시각)" if src == 'stamp'
                               else "monotonic (폴백 — 스탬프 없음/비정상)"))

        try:
            # 전처리 (preprocessing.py 가 마스크 생성의 단일 소스)
            pre = pp.process(bgr)
            lane = self.lane.process(pre['white'], pre['yellow'],
                                     pre['white'].shape, dt=dt)
            scut     = self.scut.update(pre['bev'])   # 지름길 갈색바닥 (BEV 재사용)
            per = self.per.update(bgr)       # 추론 1회 → 신호등·차량·정적장애물
            cone     = self.cone.update(bgr)  # 라바콘 구간 카메라 확인
            ckb = self.ckb.update(bgr)
        except Exception as e:
            # 프레임 단위 예외로 노드가 죽지는 않게 하되, 조용히 묻히지 않도록
            # 1초에 한 번 예외 타입까지 error 로 남긴다.
            if t_now - self._t_err > 1.0:
                self._t_err = t_now
                self.get_logger().error(
                    f"lane pipeline fail: {type(e).__name__}: {e}")
            return

        if not self._lane_valid and self.lane.candidate_count >= 3:
            self._lane_valid = True
            self.get_logger().info(f"current_lane 확정: {self.lane.current_lane}")
        current_lane = int(lane['current_lane']) if self._lane_valid else LANE_UNKNOWN

        # CamData 취합
        out = CamData()

        # ── 차선 / 조향 ─────────────────────────
        out.steering_angle        = float(lane['steering_angle'])
        out.curvature             = float(lane['curvature'])
        out.lane_detected         = bool(lane['lane_detected'])
        out.target_speed          = float(lane['target_speed'])
        out.current_lane          = current_lane
        out.driving_mode          = str(lane['driving_mode'])

        # ── 신호등 ──────────────────────────────
        out.traffic_light_color   = str(per['traffic_light_color'])  # left 포함

        # ── 전방 차량 / 정적 장애물 ──
        out.dynamic_detected      = bool(per['dynamic_detected'])
        out.static_detected       = bool(per['static_detected'])

        # ── 라바콘 구간 (라이다 오탐 방어) ────────

        out.cone_cam_detected = bool(cone['zone'])

        # ── 지름길/체커보드 ─────────────────
        out.shortcut_detected = bool(scut['on'])
        out.checkerboard_detected = bool(ckb['on'])
        out.shortcut_on = bool(scut['raw_on']) # 좌회전 끝내기용 임계없이 지름길 감지 시 즉시 on

        # ── 메시지 publish ──────────────────────────
        self.pub.publish(out)

        # ── 주기적 상태 로그(90프레임 마다) ──
        self.frame_count += 1
        if self.frame_count % 90 == 0:
            self.get_logger().info(
                f"steer={out.steering_angle:+6.1f} "
                f"spd={out.target_speed:4.1f} "
                f"mode={out.driving_mode:<8} "
                f"lane={out.current_lane} "
                f"det={out.lane_detected!s:<5} "
                f"conf={lane['lane_confidence']:.2f} "
                f"cone={cone['zone']!s:<5}({cone['ratio']:.1f}%/{cone['cols']}) "
                f"dt={dt*1000:4.0f}ms({src})"
                f"scut={scut['on']!s:<5}({scut['ratio']:.0f}%) "
                f"ckb={ckb['on']!s:<5}({ckb['ratio']:.1f}) "
            )


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()