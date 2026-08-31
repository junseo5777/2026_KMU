#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
traffic_light_cell.py — 4구 신호등 상태 판정 (실전용)

    YOLO 는 **케이스 박스만** 찾고, 상태 판정은 박스를 4등분한 셀 분석이 한다.

camera_node 에서 이렇게 쓴다:

    from kmu.traffic_light_cell import TrafficLightDetector
    self.tl = TrafficLightDetector()                 # 노드 생성 시 1회
    ...
    color = self.tl.update(bgr)                      # 'NONE'/'RED'/'YELLOW'/'GREEN'/'LEFT'
    out.traffic_light_color = color                  # CamData 로 publish

── 왜 YOLO 클래스를 안 쓰는가 (실측) ────────────────────────────────────────

학습 데이터에서 LEFT 와 GREEN 이 **같은 영상에 한 번도 같이 나오지 않았다**
(left 425개 중 415개가 tl_left/tl_left_2 두 영상 전용, 그 두 영상에 green 은 0개).
그래서 YOLO 는 '3번칸 화살표가 켜졌나' 대신 '이 세션의 배경·조명·각도' 라는
지름길을 학습했다. 학습에 안 쓴 영상(lane_10)에서 정확도가 이렇게 갈렸다:

        판정 방식          학습 세션 5개      held-out(lane_10)
        YOLO 클래스        96~100%            33%
        셀 판정            80~100%            98%
        둘의 AND           86~98%             15%
        YOLO+다수결(9)     97~100%            13%   ← 편향이라 필터가 악화시킴

시간 필터는 랜덤 노이즈를 지우지 **편향을 지우지 못한다**. 세그먼트 전체가
일관되게 틀리므로 다수결이 오답을 굳힌다. 그래서 클래스는 셀 판정만 쓴다.

셀 판정의 실패는 대부분 '기권'이다 — 오답률 1~4%, 나머지는 판정 불가(NONE).
오답도 1~2프레임 단위로 흩어져 있어 CONFIRM_FRAMES 로 걸러진다.

★ 학습 데이터를 '한 세션에 4상태가 모두 나오도록' 다시 찍어 재학습하면 YOLO
  클래스가 쓸 만해질 수 있다. 그때는 셀 판정과의 교차검증(AND)을 얹는 게 낫다.

── 셀 판정 규칙 (실측 근거) ─────────────────────────────────────────────────

 1) 절대 임계를 쓰지 않고 **셀끼리 비교**한다. 상단 ROI 에 색 마스크를 걸면
    천장 형광등·주황 펜던트등 때문에 황색이 표본의 29%에서 오검출됐다. 케이스
    안에서는 소등/점등 대비가 압도적이라 조명이 바뀌어도 상대차가 유지된다.

 2) 기준선 = 4개 지표 중 **두 번째로 작은 값**. 동시 점등은 최대 2칸(좌회전 =
    화살표+초록)이므로 최소 2칸은 소등이고, 두 번째로 작은 값은 항상 소등 셀이다.

 3) 점등 = (밝기 Δ >= LIT_DV) **OR** (유채색도 Δ >= LIT_DC). 하나만 쓰면 깨진다.
    - 밝기만: 소등 램프가 흰색으로 반사되는 프레임이 있다(근거리). 실측
      [220,161,205,255] — 소등 1번(220)이 점등된 화살표(205)보다 밝아 기준선이
      점등 셀 자신이 되어 화살표를 놓친다.
    - 유채색도만: 적색 점등은 픽셀의 58%가 클리핑돼 무채색에 가까워 못 잡는다.
    유채색도는 반드시 **밝은 픽셀에 한해** 잰다. 어두운 셀의 채도는 HSV 특성상
    노이즈라, 따로 재면 소등 셀이 기준선을 +90 넘기기도 한다.

 4) 색은 **채도 상위 픽셀의 중앙 hue**. 면적 비율로 읽으면 적색이 클리핑 때문에
    무너진다. 실측 hue: 적 171~173 / 황 21~30 / 녹 79~86, 거리 무관하게 안정.

 5) 셀 경계 좌우 18%, 상하 15%를 잘라내고 샘플링한다. 좌우는 이웃 램프의
    빛번짐, 상하는 케이스 위로 지나가는 천장 형광등을 막는다.

 6) 좌회전 = **3번 화살표 + 4번 초록 동시 점등** (규정 변경 반영). 그래서 3번
    칸을 4번보다 **먼저** 검사한다. 순서를 바꾸면 좌회전이 전부 GREEN 이 된다.
"""

import os
import time

import cv2
import numpy as np

# ══════════════════════════════════════════════════════════════════════
#  파라미터 (별도 config 파일 없이 여기서 관리)
# ══════════════════════════════════════════════════════════════════════

# 배포 가중치. 4클래스 모델이어도 **클래스는 무시하고 박스만** 쓴다.
MODEL_PATH = '~/xycar_ws/src/2026_KMU_BD/kmu/tl_yolo11n.pt'

# 추론 해상도. 640 학습본을 416 으로 추론해도 정확도가 사실상 같고(±1.6%p)
# 지연이 44.8 → 27.8ms 로 준다(2스레드 실측). veh 모델과 합쳐 10Hz 예산 안.
IMGSZ = 416
CONF = 0.35
IOU_NMS = 0.5
TORCH_THREADS = 2          # 차량 PC 에서 다른 노드와 코어를 나눠 쓴다

# 프레임 해상도. 다르면 1회 경고만 한다(셀 판정은 상대값이라 살아남는다).
CALIB_SHAPE = (480, 640)

# ── 셀 분석 ──
N_CELLS = 4
CELL_INSET = 0.18          # 좌우 잘라내기 비율 (근거 5)
CELL_VINSET = 0.15         # 상하 잘라내기 비율 (근거 5)
BRIGHT_PCT = 80            # 셀 밝기 = 상위 20% V 의 평균
SAT_PCT = 90               # hue 를 읽을 픽셀 = 채도 상위 10%
LIT_DV = 40.0              # 밝기 Δ 임계 (근거 3)
LIT_DC = 20.0              # 유채색도 Δ 임계 (근거 3)
LIT_V_MIN = 110.0          # 절대 밝기 하한 — 어두운 셀은 점등으로 안 봄
SAT_MIN = 30.0             # 점등 셀의 hue 신뢰 하한

# hue 대역 (OpenCV H 는 0~180)
HUE_RED = ((0, 12), (158, 180))
HUE_YELLOW = ((13, 40),)
HUE_GREEN = ((60, 100),)

# ── 처리 주기 (적응) ──
# 카메라 콜백은 30fps 로 들어오지만 신호는 초 단위로 바뀐다. 매 프레임 추론하면
# 초당 30회 x 20ms = 600ms, 한 코어의 60% 를 주행 내내 신호등에 쓴다.
#
# **트리거는 검출기 자신이 만든다.** 신호등을 최근에 못 봤으면 느리게 훑고,
# 보이면 즉시 빨라진다. 상태머신·정지선 같은 외부 트리거가 필요 없다 —
# 그런 트리거는 틀리면 신호등을 통째로 놓치지만, 이 방식의 최악은 '0.5초 늦게
# 인지' 다. (게다가 현재 state.py 에는 StateMachine 이 없고 정지선 검출도
#  비활성이라 쓸 만한 외부 트리거가 실제로 존재하지 않는다.)
#
# 실측(lane_10, 실주행·신호등이 6% 프레임에만 보임):
#     고정 10Hz  추론 2872회  CPU 20.3%  검출구간 8/8  오답 0
#     적응       추론  871회  CPU  6.2%  검출구간 8/8  오답 0
# 신호등이 계속 보이는 영상(tl_*)에서는 내내 active 라 절감이 없다 — 의도대로다.
#
# 안전 여유: lane_10 에서 신호등이 보이는 구간은 57~100 프레임(1.9~3.3초)이다.
# 2Hz 유휴 스캔이면 그 안에 3~6회 샘플링하므로 첫 검출 최대 0.5초 + 확정 0.3초
# = 0.8초면 읽는다. 가장 짧은 1.9초 구간에도 여유가 있다.
IDLE_EVERY = 15            # 못 봤을 때: 30fps 기준 2Hz (항상 감시는 하고 있다)
ACTIVE_EVERY = 3           # 봤을 때:   30fps 기준 10Hz
ACTIVE_HOLD_FRAMES = 90    # 마지막 검출 후 이만큼(3초)은 active 유지

# ※ 프레임 카운트 기준이라 입력 fps 가 떨어지면 처리 주기도 같이 떨어진다.
#   대신 온라인/오프라인 재생이 동일하게 동작해 검증 결과가 그대로 재현된다.
#   오프라인으로 전 프레임을 평가하려면 생성자에 process_every=1 을 준다.

# ── 시간 안정화 (아래 값은 '처리한 프레임' 기준이다) ──
# PROCESS_EVERY=3, 입력 30fps 기준 → 확정 0.3초, 유지 0.5초.
CONFIRM_FRAMES = 3         # 같은 판정이 이만큼 연속되면 확정
# 판정이 끊겨도 이만큼은 유지, 넘으면 NONE.
# (판정 없을 때 이전 값을 계속 내보내면 정지 실패로 직결된다)
HOLD_FRAMES = 5

# ── 로그 ──
LOG_PERIOD = 5.0           # 주기 상태 로그 간격(초)

# 셀 배치: [적][황][좌화살표][녹]
CELL_LAYOUT = ('red', 'yellow', 'left_arrow', 'green')

OUT_NONE = 'NONE'
_OUT = {'red': 'RED', 'yellow': 'YELLOW', 'left': 'LEFT', 'green': 'GREEN',
        None: OUT_NONE}


# ══════════════════════════════════════════════════════════════════════
#  셀 분석
# ══════════════════════════════════════════════════════════════════════

def _cell_metrics(bgr, box):
    """케이스 박스를 4등분해 셀별 (밝기, hue, 채도, 유채색도) 를 잰다.

    밝기/유채색도 모두 '상위 20% V 픽셀' 위에서 잰다. 발광부가 셀의 일부만
    차지하므로 셀 전체 평균은 소등 배경에 희석된다.
    반환: [(bright, hue, sat, chroma), ...] 또는 None(박스가 너무 작음)
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    h, w = bgr.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < N_CELLS * 3 or y1 - y0 < 6:
        return None
    dy = int((y1 - y0) * CELL_VINSET)
    y0, y1 = y0 + dy, y1 - dy
    if y1 - y0 < 3:
        return None

    hsv = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    cw = (x1 - x0) / N_CELLS
    out = []
    for i in range(N_CELLS):
        a = int(i * cw + cw * CELL_INSET)
        b = max(a + 1, int((i + 1) * cw - cw * CELL_INSET))
        cell = hsv[:, a:b]
        if cell.size == 0:
            out.append((0.0, -1.0, 0.0, 0.0))
            continue
        hc = cell[:, :, 0].astype(np.int16)
        sc = cell[:, :, 1].astype(np.int16)
        vc = cell[:, :, 2].astype(np.int16)

        top = vc >= np.percentile(vc, BRIGHT_PCT)
        bright = float(vc[top].mean())
        chroma = float(sc[top].mean())      # 밝은 픽셀의 채도 (근거 3)

        # 채도 절대 하한을 여기 걸면 원거리 적색이 통째로 탈락한다(테두리가
        # 1~2px 뿐이라 하한을 넘는 픽셀이 5개 미만). 상대 기준만 쓴다.
        m = sc >= max(float(np.percentile(sc, SAT_PCT)), 1.0)
        if m.sum() > 4:
            hue, sat = float(np.median(hc[m])), float(np.median(sc[m]))
        else:
            hue, sat = -1.0, 0.0
        out.append((bright, hue, sat, chroma))
    return out


def _hue_color(hue):
    for lo, hi in HUE_RED:
        if lo <= hue <= hi:
            return 'red'
    for lo, hi in HUE_YELLOW:
        if lo <= hue <= hi:
            return 'yellow'
    for lo, hi in HUE_GREEN:
        if lo <= hue <= hi:
            return 'green'
    return None


def _cell_colors(metrics):
    """셀별 지표 → 셀별 색. 점등 판정을 통과한 셀만 색을 읽는다.

    소등 셀에서 채도 상위 픽셀을 뽑으면 암부 노이즈가 hue 76~88(녹색대)로
    나오므로, 순서를 바꾸면 소등 셀이 녹색으로 잡힌다.
    """
    brights = [m[0] for m in metrics]
    chromas = [m[3] for m in metrics]
    if len(brights) < 3:
        return tuple(None for _ in metrics)
    base_v = sorted(brights)[1]         # 두 번째로 작은 값 = 소등 셀 (근거 2)
    base_c = sorted(chromas)[1]
    out = []
    for bright, hue, sat, chroma in metrics:
        lit = (bright - base_v >= LIT_DV) or (chroma - base_c >= LIT_DC)
        if not lit or bright < LIT_V_MIN:
            out.append(None)
            continue
        if hue < 0 or sat < SAT_MIN:
            out.append(None)            # 점등은 맞지만 색을 신뢰할 수 없음
            continue
        out.append(_hue_color(hue))
    return tuple(out)


def _classify(colors):
    """셀 색 패턴 → 상태 문자열. 판정 불가면 None.

    판정 순서(위에서부터 우선):
      2번 황      → 황색   (전환 신호, 가장 보수적)
      3번 화살표  → 좌회전 (4번 동시 점등이 표준. 화살표 단독도 좌회전)
      4번 녹 단독 → 직진
      1번 적      → 정지
    ★ 3번을 4번보다 먼저 봐야 한다 (근거 6).
    """
    if colors is None or len(colors) < 4:
        return None
    c1, c2, c3, c4 = colors[0], colors[1], colors[2], colors[3]
    if not any(colors):
        return None
    if c2 == 'yellow':
        return 'yellow'
    if c3 == 'green':
        return 'left'
    if c4 == 'green':
        return 'green'
    if c1 == 'red':
        return 'red'
    return None


# ══════════════════════════════════════════════════════════════════════
#  검출기
# ══════════════════════════════════════════════════════════════════════

class TrafficLightDetector:
    """update(bgr) -> 'NONE'/'RED'/'YELLOW'/'GREEN'/'LEFT'

    logger 에 rclpy 노드의 get_logger() 를 넘기면 그쪽으로 로그가 나간다.
    """

    def __init__(self, model_path=MODEL_PATH, imgsz=IMGSZ, conf=CONF,
                 logger=None, torch_threads=TORCH_THREADS,
                 idle_every=IDLE_EVERY, active_every=ACTIVE_EVERY,
                 active_hold=ACTIVE_HOLD_FRAMES, process_every=None):
        self.imgsz = imgsz
        self.conf = conf
        self.log = logger
        if process_every is not None:      # 고정 주기 (오프라인 평가·비교용)
            idle_every = active_every = int(process_every)
        self.idle_every = max(1, int(idle_every))
        self.active_every = max(1, int(active_every))
        self.active_hold = max(0, int(active_hold))
        self._path = os.path.expanduser(model_path)
        if not os.path.exists(self._path):
            raise FileNotFoundError(
                f"신호등 가중치를 찾을 수 없다: {self._path}\n"
                f"  tools/yolo_ds.py 로 학습 후 이 경로에 복사할 것.")

        import torch
        torch.set_num_threads(torch_threads)
        from ultralytics import YOLO
        self.model = YOLO(self._path)

        self._enabled = True
        self._shape_warned = False
        self._t_log = time.monotonic()
        self._n_in = 0             # 들어온 프레임 (스킵 포함)
        self._n_run = 0            # 실제 추론한 프레임
        self._n_box = 0
        self._lat = 0.0
        self._err = 0
        self._t_err = 0.0
        self.reset()
        self._info(f"신호등 검출기 준비 — {self._path} (imgsz={imgsz}, "
                   f"유휴 {self.idle_every}f / 활성 {self.active_every}f당 1회)")

    # ── 로그 ────────────────────────────────────────────────────
    def _info(self, msg):
        if self.log is not None:
            self.log.info(msg)
        else:
            print(f"[TL] {msg}")

    def _warn(self, msg):
        if self.log is not None:
            self.log.warn(msg)
        else:
            print(f"[TL][!] {msg}")

    # ── 상태 ────────────────────────────────────────────────────
    def reset(self):
        self.state = None          # 확정 상태 (None = 모름)
        self._cand = None
        self._cand_n = 0
        self._miss = 0
        # 첫 프레임을 즉시 처리하도록 큰 값에서 시작. _since_seen 이 크면 유휴 주기.
        self._since_proc = 1 << 30      # 마지막 '추론' 이후 들어온 프레임 수
        self._since_seen = 1 << 30      # 마지막 '박스 검출' 이후 들어온 프레임 수

    def set_enabled(self, on):
        """신호등이 의미 없는 구간에서 추론을 건너뛴다(연산 절약).

        veh 모델과 같은 프레임에서 돌면 10Hz 예산이 빠듯하므로, 상태머신이
        출발 대기·정지선 접근 구간에서만 켜면 그만큼 돌려받는다.
        """
        was = self._enabled
        self._enabled = bool(on)
        if not on:
            self.reset()           # 껐다 켜면 첫 프레임을 즉시 처리한다
        elif not was:
            self._since_proc = 1 << 30

    # ── 메인 ────────────────────────────────────────────────────
    def update(self, bgr):
        """한 프레임 처리 → 확정 상태 문자열.

        최근에 신호등을 봤으면 ACTIVE_EVERY, 아니면 IDLE_EVERY 프레임당 1회만
        실제로 추론하고, 스킵한 프레임에서는 직전 확정값을 그대로 돌려준다.
        호출부는 매 프레임 불러도 된다.
        """
        if not self._enabled or bgr is None:
            return _OUT[self.state]

        self._n_in += 1
        self._since_proc += 1
        self._since_seen += 1
        every = (self.active_every if self._since_seen <= self.active_hold
                 else self.idle_every)
        if self._since_proc < every:
            return _OUT[self.state]
        self._since_proc = 0

        if bgr.shape[:2] != CALIB_SHAPE and not self._shape_warned:
            self._shape_warned = True
            self._warn(f"프레임 {bgr.shape[:2]} 가 기준 {CALIB_SHAPE} 와 다르다. "
                       f"박스 품질을 확인할 것.")

        t0 = time.perf_counter()
        obs = None
        try:
            box = self._detect_box(bgr)
            if box is not None:
                self._since_seen = 0        # 봤다 → 다음부터 활성 주기
                self._n_box += 1
                metrics = _cell_metrics(bgr, box)
                if metrics is not None:
                    obs = _classify(_cell_colors(metrics))
        except Exception as e:
            # 한 프레임 실패로 노드가 죽거나 publish 가 끊기면 안 된다.
            self._err += 1
            now = time.monotonic()
            if now - self._t_err > 2.0:
                self._t_err = now
                self._warn(f"검출 실패({self._err}회): {type(e).__name__}: {e}")

        self._lat += (time.perf_counter() - t0) * 1000.0
        self._n_run += 1
        self._observe(obs)
        self._periodic_log()
        return _OUT[self.state]

    def _detect_box(self, bgr):
        """가장 신뢰도 높은 케이스 박스 하나. 클래스는 쓰지 않는다.

        여러 신호등이 보이면 conf 우선, 화면 중앙에 가깝고 큰 쪽을 가산한다.
        코스가 고정이라면 RTK 위치로 대상을 지정하는 편이 확실하지만 여기서는
        카메라 단독 기준만 쓴다.
        """
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf,
                               iou=IOU_NMS, verbose=False, device='cpu')[0]
        if len(r.boxes) == 0:
            return None
        w = bgr.shape[1]
        cxw = w / 2.0
        best, best_s = None, -1.0
        for b in r.boxes:
            x0, y0, x1, y1 = (float(v) for v in b.xyxy[0].tolist())
            c = float(b.conf)
            center_pen = abs((x0 + x1) / 2.0 - cxw) / cxw
            size = min((x1 - x0) / (w * 0.2), 1.0)
            s = c * (1.0 - 0.4 * center_pen) * (0.6 + 0.4 * size)
            if s > best_s:
                best, best_s = (x0, y0, x1, y1), s
        return best

    def _observe(self, obs):
        """N연속 일치로 확정, 판정이 끊기면 HOLD 후 NONE 으로 강등.

        '판정 없으면 직전 값 유지' 는 금지다. 신호가 안 보이는데 직전 GREEN 을
        계속 내보내면 정지 실패로 직결된다.
        """
        prev = self.state
        if obs is None:
            self._cand, self._cand_n = None, 0
            self._miss += 1
            if self._miss > HOLD_FRAMES:
                self.state = None
        else:
            self._miss = 0
            if obs == self._cand:
                self._cand_n += 1
            else:
                self._cand, self._cand_n = obs, 1
            if self._cand_n >= CONFIRM_FRAMES:
                self.state = self._cand
        if self.state != prev:
            self._info(f"신호 {_OUT[prev]} → {_OUT[self.state]}")

    def _periodic_log(self):
        now = time.monotonic()
        dt = now - self._t_log
        if dt < LOG_PERIOD:
            return
        self._t_log = now
        n = max(self._n_run, 1)
        # 처리 Hz 를 같이 남긴다 — 스킵이 의도대로 걸리는지, 입력이 밀리지
        # 않는지가 이 한 줄로 드러난다.
        mode = 'active' if self._since_seen <= self.active_hold else 'idle'
        self._info(f"신호등 {_OUT[self.state]} [{mode}] | "
                   f"검출 {100*self._n_box/n:.0f}% ({self._n_box}/{n}) | "
                   f"지연 {self._lat/n:.0f}ms | "
                   f"처리 {n/max(dt,1e-6):.1f}Hz (입력 {self._n_in/max(dt,1e-6):.1f}Hz)"
                   + (f" | 실패 {self._err}" if self._err else ""))
        self._n_in = self._n_run = self._n_box = 0
        self._lat = 0.0


# ══════════════════════════════════════════════════════════════════════
#  배포 점검 (주행 전 가중치·지연 확인용. 시각화 없음)
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    det = TrafficLightDetector()
    dummy = np.zeros((CALIB_SHAPE[0], CALIB_SHAPE[1], 3), np.uint8)
    det.update(dummy)                      # 워밍업
    t = time.perf_counter()
    for _ in range(10):
        det.update(dummy)
    print(f"[TL] imgsz={IMGSZ} 지연 {(time.perf_counter()-t)*100:.1f}ms/frame "
          f"({TORCH_THREADS}스레드, 유휴 주기라 대부분 스킵됨)")
    print(f"[TL] 모델 클래스 {getattr(det.model, 'names', {})} (판정에는 쓰지 않음)")
