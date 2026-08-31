#!/usr/bin/env python3
"""
lane_width_calib.py — lane_width(BEV px) 실측 캘리브레이션 노드

사용법:
  1. 차체를 직선 구간 차로 정중앙에, 차선과 평행하게 정렬
  2. python3 lane_width_calib.py
     (토픽 변경: --ros-args -p camera_topic:=/usb_cam/image_raw/front)
  3. 양쪽 차선이 실검출된 프레임에서만 폭 샘플 수집
  4. 목표 샘플 수(기본 100) 도달 시 자동 종료, q 조기 종료, r 샘플 리셋
  5. 종료 시 출력되는 median 값을 lane_detection3.py 의 LANE_WIDTH_PX 에 반영

측정 정의 (런타임 detect_lane 의 measured 와 동일):
  y = h-1 (BEV 최하단)에서 np.polyval(right_fit, h-1) - np.polyval(left_fit, h-1)
  fit 은 마킹 픽셀 무게중심을 지나므로 '마킹 중심 ↔ 마킹 중심' 거리.

주의:
  - detect_lane 만 호출하고 process() 는 호출하지 않으므로 prev fit 이 쌓이지 않음
    → 매 프레임 히스토그램+슬라이딩 윈도우 (정지 상태 측정에 결정적/재현적)
  - 제어 토픽은 일절 퍼블리시하지 않음 (실차에 물려도 안전)
  - HUD 의 slope 경고가 뜨면 차체가 차선과 평행하지 않은 것 (폭 과대측정) → 재정렬
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# ════════════════════════════════════════════════
#  전처리 파라미터 — test_camera_node.py 와 동일 (lanedetect_tuner 확정값)
# ════════════════════════════════════════════════
SIM_W, SIM_H     = 640, 480
ROI_TOP_RATIO    = 0.70
ROI_BOTTOM_RATIO = 0.93
BEV_SRC_RATIO = [(0.21,0.92), (0.37,0.01), (0.63,0.01), (0.81,0.92)]
BEV_MARGIN = 0.20

# 흰색
TOPHAT_KERNEL_W = 35
TOPHAT_THRESH   = 30
WHITE_L_MIN     = 70
WHITE_S_MAX     = 230
# 노란색
LAB_B_MIN       = 155
YEL_S_MIN       = 30
YEL_V_MIN       = 80
# 형태 (흰색에만)
MIN_LEN_WHITE   = 100
MAX_LINE_W      = 30
CLOSE_K         = (3, 3)

# ── 캘리브레이션 설정 ──
N_SAMPLES_DEFAULT = 100
SLOPE_WARN        = 0.15   # y=h-1 fit 기울기 |dx/dy| 경고 임계 (~8.5°: 폭 ~1% 과대측정)

try:
    from kmu.lane_detection3 import LaneDetector
except ImportError:
    print("[경고] lane_detection3.py 모듈을 찾을 수 없습니다. 같은 디렉토리에 위치시켜 주세요.")
    import sys
    sys.exit(1)


class LaneWidthCalibNode(Node):
    def __init__(self):
        super().__init__('lane_width_calib')
        self.bridge = CvBridge()
        self.detector = LaneDetector()
        self._M = None
        self.samples = []
        self.done = False

        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('n_samples', N_SAMPLES_DEFAULT)
        camera_topic  = self.get_parameter('camera_topic').value
        self.n_target = int(self.get_parameter('n_samples').value)

        qos_profile = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE
        )
        self.image_sub = self.create_subscription(
            Image, camera_topic, self.image_callback, qos_profile)

        self.get_logger().info(
            f"lane_width 캘리브레이션 시작 (topic={camera_topic}, 목표 {self.n_target} 샘플)")
        self.get_logger().info("차체를 차로 정중앙, 차선과 평행하게 정렬하세요. q 종료 | r 리셋")

    # ── 전처리 (test_camera_node 와 동일) ──
    def _init_bev_matrices(self, rh, rw):
        src = np.float32([(x*rw, y*rh) for x, y in BEV_SRC_RATIO])
        dst = np.float32([[rw*BEV_MARGIN, rh], [rw*BEV_MARGIN, 0],
                          [rw*(1-BEV_MARGIN), 0], [rw*(1-BEV_MARGIN), rh]])
        self._M = cv2.getPerspectiveTransform(src, dst)

    def _filter_white(self, mask):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = np.zeros_like(mask)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 50:
                continue
            la = max(bw, bh)
            if la < MIN_LEN_WHITE:
                continue
            if area / max(1, la) > MAX_LINE_W:
                continue
            out[labels == i] = 255
        return out

    def _filter_yellow(self, mask):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        out = np.zeros_like(mask)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < 40:
                continue
            la = max(bw, bh)
            if area / max(1, la) > MAX_LINE_W:
                continue
            out[labels == i] = 255
        return out

    def preprocess_image(self, frame):
        img = cv2.resize(frame, (SIM_W, SIM_H))
        h, w = img.shape[:2]
        y0, y1 = int(h*ROI_TOP_RATIO), int(h*ROI_BOTTOM_RATIO)
        roi = img[y0:y1, :]
        rh, rw = roi.shape[:2]
        if self._M is None:
            self._init_bev_matrices(rh, rw)
        bev = cv2.warpPerspective(roi, self._M, (rw, rh))
        kc = cv2.getStructuringElement(cv2.MORPH_RECT, CLOSE_K)

        hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
        L, S = hls[:,:,1], hls[:,:,2]
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (TOPHAT_KERNEL_W | 1, 1))
        tophat = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, k)
        _, m_top = cv2.threshold(tophat, TOPHAT_THRESH, 255, cv2.THRESH_BINARY)
        m_col = ((L >= WHITE_L_MIN) & (S <= WHITE_S_MAX)).astype(np.uint8)*255
        white = cv2.bitwise_and(m_top, m_col)
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kc)
        white = self._filter_white(white)

        lab = cv2.cvtColor(bev, cv2.COLOR_BGR2LAB); b_ch = lab[:,:,2]
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        H, Sv, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
        yellow_raw = (((b_ch >= LAB_B_MIN) & (H>=10)&(H<=50) &
                       (Sv>=YEL_S_MIN) & (V>=YEL_V_MIN)).astype(np.uint8)*255)
        yellow_raw = cv2.morphologyEx(yellow_raw, cv2.MORPH_CLOSE, kc)
        yellow = self._filter_yellow(yellow_raw)

        kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        white = cv2.bitwise_and(white, cv2.bitwise_not(cv2.dilate(yellow_raw, kd)))

        return white, yellow

    # ── 측정 ──
    def image_callback(self, msg):
        if self.done:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        white, yellow = self.preprocess_image(cv_image)
        h, w = white.shape

        # detect_lane 만 사용: 가상 채움 전의 실검출 fit 만 측정에 사용
        (_, _, _, (real_l, real_r), (rlf, rrf), _) = \
            self.detector.detect_lane(white, yellow)

        measured = slope = None
        if real_l and real_r:
            xl = float(np.polyval(rlf, h-1))
            xr = float(np.polyval(rrf, h-1))
            measured = xr - xl
            # 최하단 접선 기울기: 차선-차체 정렬 확인용
            slope = float((np.polyval(np.polyder(rlf), h-1) +
                           np.polyval(np.polyder(rrf), h-1)) / 2.0)
            if 0 < measured < w:
                self.samples.append(measured)

        self._show(white, yellow, rlf, rrf, measured, slope, h, w)

        if len(self.samples) >= self.n_target:
            self.finish()
            raise SystemExit

    # ── 시각화 ──
    def _show(self, white, yellow, rlf, rrf, measured, slope, h, w):
        canvas = cv2.cvtColor(np.clip(white.astype(np.int16)+yellow, 0, 255).astype(np.uint8),
                              cv2.COLOR_GRAY2BGR)
        for fit, color in ((rlf, (0,255,255)), (rrf, (255,255,0))):
            if fit is None:
                continue
            pts = [(int(np.polyval(fit, y)), y) for y in range(0, h, 4)]
            for i in range(len(pts)-1):
                if 0 <= pts[i][0] < w:
                    cv2.line(canvas, pts[i], pts[i+1], color, 2)
        if measured is not None:
            xl = int(np.polyval(rlf, h-1)); xr = int(np.polyval(rrf, h-1))
            cv2.line(canvas, (xl, h-3), (xr, h-3), (0,0,255), 2)

        arr = np.array(self.samples)
        lines = [f"samples {len(arr)}/{self.n_target}"]
        if measured is not None:
            lines.append(f"cur {measured:6.1f}px  slope {slope:+.3f}")
        else:
            lines.append("waiting for both lanes...")
        if len(arr) > 0:
            lines.append(f"median {np.median(arr):6.1f}  mean {arr.mean():6.1f}")
            lines.append(f"std {arr.std():5.2f}  min {arr.min():5.1f}  max {arr.max():5.1f}")
        if slope is not None and abs(slope) > SLOPE_WARN:
            lines.append("WARN: not parallel to lane! realign")
        for i, t in enumerate(lines):
            cv2.putText(canvas, t, (8, 18+18*i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 3)
            cv2.putText(canvas, t, (8, 18+18*i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)

        cv2.imshow("lane_width_calib", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.finish()
            raise SystemExit
        elif key == ord('r'):
            self.samples.clear()
            self.get_logger().info("샘플 리셋")

    # ── 결과 출력 ──
    def finish(self):
        if self.done:
            return
        self.done = True
        arr = np.array(self.samples)
        print("\n" + "="*48)
        if len(arr) == 0:
            print("수집된 샘플 없음 — 양쪽 차선이 검출되는지 확인하세요.")
        else:
            print(f"lane_width 실측 결과 ({len(arr)} 샘플)")
            print(f"  median {np.median(arr):7.1f} px   ← 권장값")
            print(f"  mean   {arr.mean():7.1f} px")
            print(f"  std    {arr.std():7.2f} px")
            print(f"  range  [{arr.min():.1f}, {arr.max():.1f}] px")
            print(f"\nlane_detection3.py 에 반영:")
            print(f"  LANE_WIDTH_PX  = {round(float(np.median(arr)))}")
        print("="*48)


def main(args=None):
    rclpy.init(args=args)
    node = LaneWidthCalibNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.finish()
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()