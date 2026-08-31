#!/usr/bin/env python3
"""
detect_tuner.py — 실시간 카메라(/image_raw) 기반 전처리·검출 튜너

lanedetect_tuner.py 가 '녹화 영상'으로 하는 검증을 라이브 스트림에서 수행한다.
녹화본과의 결정적 차이는 프레임 스크럽이 불가능하다는 점이라, 그 자리를
[F]리즈 + 라이브 트랙바가 대신한다. 프레임을 얼려놓고 임계를 돌리면
같은 장면에 대한 파라미터 반응을 즉시 비교할 수 있다.

윈도우 4개:
  [controls]    전처리 파라미터 트랙바 (W:흰색 / Y:노란색 / M:형태필터)
  [masks]       BEV 컬러 위에 흰(시안)·노랑(노랑) 마스크 오버레이
                → "눈에는 보이는데 왜 안 잡히나" 를 단계에서 바로 확인
  [BEV_debug]   슬라이딩 윈도우(초록) / margin 밴드(주황) / 검출픽셀(좌빨강·우파랑)
                / fit 곡선 / lookahead 목표점 / L,R 탐색모드 + REJ
  [drive_view]  원본 + 차선영역 역투영(초록) + HUD + 조향 바

조작:
  f  프리즈 토글 (얼린 프레임에 대해 파라미터 반응 확인 — 튜닝의 핵심)
  x  검출기 상태 리셋 (prev fit 비움 → 슬라이딩 윈도우 재획득 강제)
  s  스냅샷 저장 (raw/bev/drive PNG)
  v  원본 프레임 녹화 토글 → 저장본을 lanedetect_tuner.py 로 오프라인 재분석
  p  현재 파라미터 출력
  q  종료 (종료 시에도 파라미터 출력)

실행:
  python3 detect_tuner.py
  python3 detect_tuner.py --ros-args -p camera_topic:=/usb_cam/image_raw/front

주의: 프리즈 중에는 매 재처리마다 검출기를 새로 만든다. 같은 프레임에 대해
      process() 를 반복 호출하면 prev fit 이 수렴해 margin search 로 넘어가므로,
      전처리 변경의 효과가 상태에 가려진다. 깨끗한 상태에서 평가해야 비교가 성립.
"""

import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

try:
    from lane_detection3 import LaneDetector, MARGIN_SEARCH
    import preprocessing as pp
except ImportError:
    print("[경고] lane_detection3.py / preprocessing.py 를 찾을 수 없습니다. 같은 디렉토리에 두세요.")
    raise SystemExit(1)

SAVE_DIR = os.path.expanduser("~/Videos/lane/detect_tuner_out")

# 기하(ROI/BEV)는 트랙바로 노출하지 않는다.
# 바꾸면 BEV 픽셀 스케일이 변해 lane_detection3 의 LANE_WIDTH_PX(350, 실측)와
# pairing/윈도우 임계가 전부 무효가 되므로, lane_width_calib.py 재실행이 선행되어야 한다.
# 초기 트랙바 값 = preprocessing.py 확정값 (단일 소스)
INIT = dict(pp.DEFAULTS)

CTRL = 'controls'


def preprocess(frame, p):
    """
    preprocessing.lane_masks 에 트랙바 값을 그대로 주입.
    튜너가 프로덕션과 '같은 코드'를 돌리므로 튜너/노드 로직이 갈라질 수 없다.
    """
    white, yellow, bev, span = pp.lane_masks(frame, p)
    return white, yellow, bev, cv2.resize(frame, (pp.SIM_W, pp.SIM_H)), span


def draw_fit(canvas, fit, color):
    if fit is None:
        return
    h = canvas.shape[0]
    pts = [(int(np.polyval(fit, y)), y) for y in range(0, h, 4)]
    for i in range(len(pts) - 1):
        if 0 <= pts[i][0] < canvas.shape[1]:
            cv2.line(canvas, pts[i], pts[i + 1], color, 2)


def make_masks_view(bev, white, yellow):
    """BEV 컬러 위에 마스크 오버레이 — 검출 실패의 원인 단계를 눈으로 확인."""
    vis = bev.copy()
    vis[white > 0] = (255, 255, 0)     # 흰 마스크 = 시안
    vis[yellow > 0] = (0, 255, 255)    # 노랑 마스크 = 노랑
    cv2.putText(vis, f"white {int((white>0).sum()):5d}px   yellow {int((yellow>0).sum()):5d}px",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(vis, f"white {int((white>0).sum()):5d}px   yellow {int((yellow>0).sum()):5d}px",
                (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return vis


def visualize(det, res, white, yellow, orig, roi_span, frozen, rec):
    h, w = white.shape

    # ── BEV_debug ──
    canvas = cv2.cvtColor(np.clip(white.astype(np.int16) + yellow, 0, 255).astype(np.uint8),
                          cv2.COLOR_GRAY2BGR)
    if det._left_px is not None:
        lx, ly = det._left_px
        canvas[np.clip(ly, 0, h - 1), np.clip(lx, 0, w - 1)] = (0, 0, 255)
    if det._right_px is not None:
        rx, ry = det._right_px
        canvas[np.clip(ry, 0, h - 1), np.clip(rx, 0, w - 1)] = (255, 100, 0)
    for (x0, y0, x1, y1) in det._dbg_rects:
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 200, 0), 1)
    for side, fit in (('left', res['left_fit']), ('right', res['right_fit'])):
        if det._dbg_search.get(side) == 'margin' and fit is not None:
            for y in range(0, h, 8):
                cx = int(np.polyval(fit, y))
                cv2.line(canvas, (cx - MARGIN_SEARCH, y), (cx - MARGIN_SEARCH, y + 4), (255, 180, 0), 1)
                cv2.line(canvas, (cx + MARGIN_SEARCH, y), (cx + MARGIN_SEARCH, y + 4), (255, 180, 0), 1)
    draw_fit(canvas, res['left_fit'], (0, 255, 255))
    draw_fit(canvas, res['right_fit'], (255, 255, 0))
    y_eval = int(h * (1.0 - res['lookahead']))
    if res['left_fit'] is not None and res['right_fit'] is not None:
        tx = int((np.polyval(res['left_fit'], y_eval) + np.polyval(res['right_fit'], y_eval)) / 2)
        cv2.circle(canvas, (tx, y_eval), 6, (0, 0, 255), -1)
        cv2.line(canvas, (tx, y_eval), (w // 2, h - 1), (0, 0, 255), 1)
    cv2.line(canvas, (w // 2, 0), (w // 2, h), (120, 120, 120), 1)
    lbl = f"L:{det._dbg_search['left']} R:{det._dbg_search['right']}"
    if res.get('pair_reject'):
        lbl += f"  REJ:{res['pair_reject']}"
    cv2.putText(canvas, lbl, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ── drive_view ──
    vis = orig.copy()
    y0r, _ = roi_span
    if res['left_fit'] is not None and res['right_fit'] is not None:
        ys = np.arange(0, h, 6)
        lpts = np.array([[np.polyval(res['left_fit'], y), y] for y in ys], np.float32)
        rpts = np.array([[np.polyval(res['right_fit'], y), y] for y in ys[::-1]], np.float32)
        poly = np.vstack([lpts, rpts]).reshape(-1, 1, 2)
        poly_roi = cv2.perspectiveTransform(poly, pp._Minv).astype(np.int32)
        poly_roi[:, 0, 1] += y0r
        layer = vis.copy()
        cv2.fillPoly(layer, [poly_roi], (0, 180, 0))
        vis = cv2.addWeighted(layer, 0.3, vis, 0.7, 0)

    steer, speed = res['steering_angle'], res['target_speed']
    for i, t in enumerate([
        f"steer {steer:+6.1f}   speed {speed:4.1f}",
        f"mode  {res['driving_mode']:<9} lane {res['current_lane']}",
        f"conf  {res['lane_confidence']:.2f}  detected {res['lane_detected']}",
        f"curv  {res['curvature']:+.5f}  bend {res['bend_ahead']:.0f}",
    ]):
        cv2.putText(vis, t, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, t, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    if frozen:
        cv2.putText(vis, "FROZEN (state reset each eval)", (10, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, "FROZEN (state reset each eval)", (10, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    if rec:
        cv2.circle(vis, (vis.shape[1] - 20, 20), 7, (0, 0, 255), -1)
        cv2.putText(vis, "REC", (vis.shape[1] - 62, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    bw_, bh_, cy = 300, 16, vis.shape[0] - 24
    cx = vis.shape[1] // 2
    cv2.rectangle(vis, (cx - bw_ // 2, cy - bh_ // 2), (cx + bw_ // 2, cy + bh_ // 2), (200, 200, 200), 1)
    fill = int(np.clip(steer, -100, 100) / 100 * (bw_ // 2))
    col = (0, 128, 255) if abs(steer) > 60 else (0, 255, 0)
    cv2.rectangle(vis, (cx, cy - bh_ // 2 + 2), (cx + fill, cy + bh_ // 2 - 2), col, -1)
    cv2.line(vis, (cx, cy - bh_ // 2 - 3), (cx, cy + bh_ // 2 + 3), (255, 255, 255), 1)
    return canvas, vis


class FrameGrabber(Node):
    """콜백은 최신 프레임 보관만. 처리·GUI 는 메인 루프에서 (imshow 스레드 안전 + 자연스러운 프레임 드롭)."""

    def __init__(self):
        super().__init__('detect_tuner')
        self.bridge = CvBridge()
        self.latest = None
        self.n_recv = 0
        self.declare_parameter('camera_topic', '/image_raw')
        topic = self.get_parameter('camera_topic').value
        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Image, topic, self._cb, qos)
        self.topic = topic
        self.get_logger().info(f"구독: {topic}")

    def _cb(self, msg):
        try:
            self.latest = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.n_recv += 1
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")


def make_controls():
    nb = lambda v: None
    cv2.namedWindow(CTRL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL, 460, 400)
    cv2.createTrackbar('W:thW',    CTRL, INIT['tophat_w'], 101, nb)   # 가로 tophat 커널
    cv2.createTrackbar('W:thV',    CTRL, INIT['tophat_v'], 101, nb)   # 세로 tophat 커널
    cv2.createTrackbar('W:thT',    CTRL, INIT['tophat_th'], 120, nb)
    cv2.createTrackbar('W:Lmin',   CTRL, INIT['l_min'], 255, nb)
    cv2.createTrackbar('W:Smax',   CTRL, INIT['s_max'], 255, nb)
    cv2.createTrackbar('Y:labB',   CTRL, INIT['lab_b'], 255, nb)
    cv2.createTrackbar('Y:Smin',   CTRL, INIT['yel_s'], 255, nb)
    cv2.createTrackbar('Y:Vmin',   CTRL, INIT['yel_v'], 255, nb)
    cv2.createTrackbar('Y:dil',    CTRL, INIT['yel_dil'], 15, nb)     # 팽창 반경(px)
    cv2.createTrackbar('M:minLen', CTRL, INIT['min_len'], 110, nb)    # 흰색 세로범위 하한
    cv2.createTrackbar('M:maxLW',  CTRL, INIT['max_lw'], 100, nb)


def read_params():
    g = lambda n: cv2.getTrackbarPos(n, CTRL)
    return dict(tophat_w=g('W:thW'), tophat_v=g('W:thV'), tophat_th=g('W:thT'),
                l_min=g('W:Lmin'), s_max=g('W:Smax'),
                lab_b=g('Y:labB'), yel_s=g('Y:Smin'), yel_v=g('Y:Vmin'),
                yel_dil=g('Y:dil'), min_len=g('M:minLen'), max_lw=g('M:maxLW'))


def print_params(p):
    d = p['yel_dil']
    print("\n" + "=" * 52)
    print("preprocessing.py 상단에 복사 (전 파일에 자동 전파):")
    print(f"  TOPHAT_KERNEL_W = {p['tophat_w']}")
    print(f"  TOPHAT_KERNEL_V = {p['tophat_v']}")
    print(f"  TOPHAT_THRESH   = {p['tophat_th']}")
    print(f"  WHITE_L_MIN     = {p['l_min']}")
    print(f"  WHITE_S_MAX     = {p['s_max']}")
    print(f"  LAB_B_MIN       = {p['lab_b']}")
    print(f"  YEL_S_MIN       = {p['yel_s']}")
    print(f"  YEL_V_MIN       = {p['yel_v']}")
    print(f"  YEL_DILATE_K    = ({d*2+1}, {d*2+1})")
    print(f"  MIN_LEN_WHITE   = {p['min_len']}")
    print(f"  MAX_LINE_W      = {p['max_lw']}")
    print("=" * 52)


def main(args=None):
    rclpy.init(args=args)
    node = FrameGrabber()
    os.makedirs(SAVE_DIR, exist_ok=True)
    make_controls()

    det = LaneDetector()
    frozen_frame = None
    writer = None
    prev_t = time.time()
    proc_ms = 0.0
    idx = 0
    p = read_params()

    print("f 프리즈 | x 상태리셋 | s 스냅샷 | v 녹화 | p 파라미터 | q 종료")
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.005)
            frame = frozen_frame if frozen_frame is not None else node.latest
            if frame is None:
                blank = np.zeros((120, 560, 3), np.uint8)
                cv2.putText(blank, f"waiting for {node.topic} ...", (12, 66),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
                cv2.imshow('drive_view', blank)
                if (cv2.waitKey(30) & 0xFF) in (ord('q'), 27):
                    break
                continue

            now = time.time()
            dt = now - prev_t
            prev_t = now
            p = read_params()

            t0 = time.time()
            white, yellow, bev, orig, span = preprocess(frame, p)
            # 프리즈 중에는 상태를 배제하고 전처리 효과만 평가 (docstring 주의 참조)
            if frozen_frame is not None:
                det = LaneDetector()
            res = det.process(white, yellow, white.shape, dt=dt)
            proc_ms = 0.8 * proc_ms + 0.2 * (time.time() - t0) * 1000

            canvas, vis = visualize(det, res, white, yellow, orig, span,
                                    frozen_frame is not None, writer is not None)
            cv2.putText(vis, f"proc {proc_ms:.1f}ms  rx{node.n_recv}", (10, vis.shape[0] - 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow('masks', make_masks_view(bev, white, yellow))
            cv2.imshow('BEV_debug', canvas)
            cv2.imshow('drive_view', vis)

            if writer is not None:
                writer.write(cv2.resize(frame, (pp.SIM_W, pp.SIM_H)))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('f'):
                if frozen_frame is None:
                    frozen_frame = frame.copy()
                    print("[freeze] 현재 프레임 고정 — 트랙바로 파라미터 반응 확인")
                else:
                    frozen_frame = None
                    det = LaneDetector()
                    print("[freeze] 해제 (검출기 리셋)")
            elif key == ord('x'):
                det = LaneDetector()
                print("[reset] 검출기 상태 초기화")
            elif key == ord('s'):
                idx += 1
                cv2.imwrite(os.path.join(SAVE_DIR, f"s{idx:04d}_raw.png"), frame)
                cv2.imwrite(os.path.join(SAVE_DIR, f"s{idx:04d}_bev.png"), canvas)
                cv2.imwrite(os.path.join(SAVE_DIR, f"s{idx:04d}_drive.png"), vis)
                print(f"[save] s{idx:04d}_*.png → {SAVE_DIR}")
            elif key == ord('v'):
                if writer is None:
                    path = os.path.join(SAVE_DIR, f"rec_{time.strftime('%m%d_%H%M%S')}.avi")
                    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'),
                                             20.0, (pp.SIM_W, pp.SIM_H))
                    print(f"[rec] 시작 → {path}  (lanedetect_tuner.py 로 재분석 가능)")
                else:
                    writer.release(); writer = None
                    print("[rec] 정지")
            elif key == ord('p'):
                print_params(p)
    except KeyboardInterrupt:
        pass
    finally:
        print_params(p)
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()