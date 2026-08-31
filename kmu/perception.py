#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
perception.py — 통합 카메라 인지 (신호등 + 추월차량 + 정적장애물)

    YOLO 1회 추론 → 클래스별 분기 → 신호등은 셀 판정, 나머지는 기하 필터

camera_node 에서 이렇게 쓴다 (검출기 하나만 둔다):

    from .perception import Perception
    self.per = Perception(logger=self.get_logger())      # 노드 생성 시 1회
    ...
    p = self.per.update(bgr)
    out.traffic_light_color = p['traffic_light_color']   # 'NONE'/'RED'/...
    out.dynamic_detected    = p['dynamic_detected']      # 추월 대상 차량
    out.static_detected     = p['static_detected']       # 정적 장애물

라바콘 구간 판별은 여기서 하지 않는다. 학습 데이터에서 라바콘 구간을 일부러
null 로 두었으므로(정적 장애물만 static 으로 보게 하려는 의도) YOLO 는 콘을
검출하지 않는다. 구간 판별은 lubbercone_cam.ConeCamGate(색+열분포) 와 라이다가
맡고, 이 모듈은 static_detected 만 올린다.

── 왜 통합했나 ──────────────────────────────────────────────────────────────

분리 운영하면 프레임마다 추론이 2회 돌아 활성 구간에서 16.7ms/frame 이 든다.
통합하면 1회로 끝나 10.0ms/frame 이다(40% 절감). **클래스 수는 추론 비용에
영향이 없다** — 같은 백본에서 1클래스 vs 4클래스 지연 차이가 부호까지 뒤집혀
노이즈 수준이었다. 즉 6클래스를 유지하는 데 드는 연산 비용은 0이다.

── 클래스별 판정 방식 ───────────────────────────────────────────────────────

 신호등(red/yellow/left/green)
   YOLO 의 **클래스는 쓰지 않고 박스만** 쓴다. 학습 데이터에서 세션 편향이
   생기기 쉬워(과거 held-out 정확도 33%) 클래스를 신뢰할 수 없었고, 박스를
   4등분해 셀을 읽는 쪽이 held-out 100% 였다. 판정 로직은
   traffic_light_cell.py 의 검증된 함수를 그대로 재사용한다.
   ★ 재학습 후 YOLO 클래스가 held-out 에서 쓸 만해지면 셀 판정과 AND 로 묶어
     교차검증을 복원할 수 있다. TL_USE_YOLO_CLASS 를 켜면 된다.

 rccar
   추월영역 기하 필터(접지점·중심·종횡비) → N-of-M 확정 → 시간 래치.
   ★ **정적 장애물의 차체를 추월 대상으로 오인하지 않도록**, 바로 위에 static
     박스가 겹쳐 있는 rccar 는 후보에서 뺀다. 실측(static_1 f1135)에서 차체가
     추월영역 게이트를 통과하는 순간 그 위에 콘 박스가 겹쳐 있었다.

 static (라바콘 + 정적 장애물, 같은 클래스)
   **개수로 모드를 가른다.** 라바콘 구간은 콘이 밀집해 여러 개가 잡히고,
   정적 장애물은 하나뿐이다. 실측(y_bot>=300 필터 + 2/4 진입 / 5연속 이탈):
       정적 구간  개수>=3 비율  3% / 0%
       콘 구간    개수>=3 비율 81% / 88%
   거리 필터가 없으면 지평선 쪽 콘까지 세어 콘 구간 5.4초 전부터 켜진다.

── 처리 주기 (적응) ─────────────────────────────────────────────────────────

아무것도 못 봤으면 느리게 훑고(유휴), 뭐라도 보이면 빨라진다(활성). 외부
트리거가 필요 없고, 트리거가 틀려 대상을 통째로 놓칠 위험도 없다. 최악은
'한 유휴 주기만큼 늦게 인지' 다.
"""

import os
import time
from collections import deque

from .traffic_light_cell import (          # 검증된 셀 판정 로직 재사용
    _cell_metrics, _cell_colors, _classify, _OUT,
)


def _box_ok(bgr, box):
    """Return False when a detector box is clipped by the image boundary."""
    if bgr is None or len(box) != 4:
        return False
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = box
    return 1.0 < x0 < x1 < w - 1.0 and 1.0 < y0 < y1 < h - 1.0

# ══════════════════════════════════════════════════════════════════════
#  파라미터
# ══════════════════════════════════════════════════════════════════════

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'all_yolo11n.pt',
)

IMGSZ = 416                # 학습은 640, 배포는 실측 후 결정 (기본 416)
CONF = 0.35
IOU_NMS = 0.45
TORCH_THREADS = 2

CALIB_SHAPE = (480, 640)   # 아래 픽셀 임계는 전부 이 해상도 기준

TL_NAMES = ('red', 'yellow', 'left', 'green')
TL_USE_YOLO_CLASS = False  # True 면 YOLO 클래스와 셀 판정이 일치할 때만 확정

# ── 추월차량 (rccar) ──
VEH_Y_BOT_MIN = 275.0
VEH_XC_MIN, VEH_XC_MAX = 170.0, 470.0
VEH_WH_MIN = 0.6
VEH_MAX_AREA_RATIO = 0.5
VEH_WINDOW, VEH_CONFIRM_N = 3, 2
VEH_HOLD_SEC = 0.5

# ── 정적장애물 (static) ──
# static 클래스는 '콘을 얹은 rccar' 복합체다. 라바콘 구간의 맨 콘에는 거의
# 붙지 않지만(lane_7 1% / lane_8 3% / lane_9 0%) 0 은 아니다.
STATIC_Y_BOT_MIN = 240.0

# ★ CONFIRM 을 15 로 둔 근거 — 개수로는 못 거른다.
#   y_bot 게이트를 통과한 박스 개수는 정탐이든 오탐이든 **항상 1개**였다
#   (2개 이상인 프레임: static_1/2/3/5, lane_7/8/9 전부 0.0%). 그래서 '1개일
#   때만 인정' 류의 개수 규칙은 아무것도 거부하지 못하는 무동작 규칙이다.
#   가르는 것은 지속시간이다. 이 모듈의 실제 주기로 잰 연속 hit 런의 최댓값:
#       진짜 정적   static_1 152 / static_2 124(held-out) / static_3 34 / static_5 21
#       콘 구간     lane_8 11 / lane_7 2 / lane_9 0
#   11 과 21 사이를 끊어야 한다. 15 면 콘 구간 오탐 0%, 정적 4영상 모두 유지.
#   대가는 확정 지연 15×3프레임 = 1.5초(활성 주기). static_2 는 y_bot>=300 을
#   12초간 유지하므로 충분하지만 static_5(2.1초) 는 여유가 0.6초뿐이다 —
#   그 구간은 라이다 차선 확인이 받아야 한다. 낮추면 lane_8 이 되살아난다.
STATIC_CONFIRM = 6
STATIC_HOLD_SEC = 0.5

# ── 처리 주기 (적응) ──
IDLE_EVERY = 6
ACTIVE_EVERY = 3
ACTIVE_HOLD_FRAMES = 45

LOG_PERIOD = 5.0


class Perception:
    """update(bgr) -> dict. 추론 1회로 세 가지 판정을 모두 낸다."""

    def __init__(self, model_path=MODEL_PATH, imgsz=IMGSZ, conf=CONF,
                 logger=None, torch_threads=TORCH_THREADS,
                 idle_every=IDLE_EVERY, active_every=ACTIVE_EVERY,
                 active_hold=ACTIVE_HOLD_FRAMES, process_every=None):
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.log = logger
        if process_every is not None:
            idle_every = active_every = int(process_every)
        self.idle_every = max(1, int(idle_every))
        self.active_every = max(1, int(active_every))
        self.active_hold = max(0, int(active_hold))

        self._path = os.path.expanduser(model_path)
        if not os.path.exists(self._path):
            raise FileNotFoundError(
                f"통합 가중치를 찾을 수 없다: {self._path}\n"
                f"  tools/yolo_ds.py --ds all 로 학습 후 이 경로에 복사할 것.")

        import torch
        torch.set_num_threads(int(torch_threads))
        from ultralytics import YOLO
        self.model = YOLO(self._path)
        names = getattr(self.model, 'names', {}) or {}
        # id -> 이름. **개수로 분기하지 않는다** — 클래스가 늘어난 순간 조용히
        # 깨지기 때문이다. 이름으로 찾고, 없으면 시작 시점에 알린다.
        self.id2name = {int(i): str(n).lower() for i, n in names.items()}
        have = set(self.id2name.values())
        missing = [c for c in ('rccar', 'static') + TL_NAMES if c not in have]
        if missing:
            self._warn(f"가중치에 없는 클래스: {missing} (names={names})")

        self._enabled = True
        self._shape_warned = False
        self._t_log = time.monotonic()
        self._n_in = self._n_run = 0
        self._n_tl = self._n_veh = self._n_static = 0
        self._lat = 0.0
        self._err = 0
        self._t_err = 0.0
        self.reset()
        self._info(f"통합 인지 준비 — {self._path} (imgsz={self.imgsz}, "
                   f"유휴 {self.idle_every}f / 활성 {self.active_every}f당 1회)")

    # ── 로그 ────────────────────────────────────────────────────
    def _info(self, msg):
        (self.log.info if self.log else (lambda m: print(f"[PER] {m}")))(msg)

    def _warn(self, msg):
        (self.log.warn if self.log else (lambda m: print(f"[PER][!] {m}")))(msg)

    # ── 상태 ────────────────────────────────────────────────────
    def reset(self):
        # 신호등
        self.tl_state = None
        self._tl_cand, self._tl_cand_n, self._tl_miss = None, 0, 0
        # 추월차량
        self.dynamic = False
        self._veh_hist = deque(maxlen=VEH_WINDOW)
        self._veh_t = None
        # 정적장애물
        self.static = False
        self._st_n, self._st_t = 0, None
        # 주기
        self._since_proc = 1 << 30
        self._since_seen = 1 << 30

    def set_enabled(self, on):
        was = self._enabled
        self._enabled = bool(on)
        if not on:
            self.reset()
        elif not was:
            self._since_proc = 1 << 30

    # ── 메인 ────────────────────────────────────────────────────
    def update(self, bgr):
        if not self._enabled or bgr is None:
            return self._result()

        self._n_in += 1
        self._since_proc += 1
        self._since_seen += 1
        every = (self.active_every if self._since_seen <= self.active_hold
                 else self.idle_every)
        if self._since_proc < every:
            return self._result(expire_only=True)
        self._since_proc = 0

        if bgr.shape[:2] != CALIB_SHAPE and not self._shape_warned:
            self._shape_warned = True
            self._warn(f"프레임 {bgr.shape[:2]} 가 기준 {CALIB_SHAPE} 와 다르다. "
                       f"기하 임계가 픽셀 좌표라 무의미해진다.")

        t0 = time.perf_counter()
        try:
            tl_boxes, veh_boxes, st_boxes = self._detect(bgr)
            if tl_boxes or veh_boxes or st_boxes:
                self._since_seen = 0
            self._update_tl(bgr, tl_boxes)
            self._update_static(st_boxes)
            self._update_vehicle(veh_boxes, st_boxes, bgr.shape)
        except Exception as e:
            # 한 프레임 실패로 노드가 죽거나 publish 가 끊기면 안 된다.
            self._err += 1
            now = time.monotonic()
            if now - self._t_err > 2.0:
                self._t_err = now
                self._warn(f"인지 실패({self._err}회): {type(e).__name__}: {e}")

        self._lat += (time.perf_counter() - t0) * 1000.0
        self._n_run += 1
        self._periodic_log()
        return self._result()

    def _detect(self, bgr):
        """추론 1회 → 클래스별 박스 목록. (tl, rccar, static)"""
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf,
                               iou=IOU_NMS, verbose=False, device='cpu')[0]
        tl, veh, st = [], [], []
        for b in r.boxes:
            name = self.id2name.get(int(b.cls), '')
            box = tuple(float(v) for v in b.xyxy[0].tolist())
            conf = float(b.conf)
            if name in TL_NAMES:
                tl.append((conf, box, name))
            elif name == 'rccar':
                veh.append((conf, box))
            elif name == 'static':
                st.append((conf, box))
        return tl, veh, st

    # ── 신호등 ──────────────────────────────────────────────────
    def _update_tl(self, bgr, tl_boxes):
        obs = None
        if tl_boxes:
            self._n_tl += 1
            w = bgr.shape[1]

            # 여러 신호등이 보이면 conf 우선, 화면 중앙에 가깝고 큰 쪽 가산
            def score(t):
                c, (x0, y0, x1, y1), _ = t
                pen = abs((x0 + x1) * 0.5 - w * 0.5) / (w * 0.5)
                size = min((x1 - x0) / (w * 0.2), 1.0)
                return c * (1.0 - 0.4 * pen) * (0.6 + 0.4 * size)
            conf, box, cls_name = max(tl_boxes, key=score)
            # 잘린 박스는 4등분 전제가 깨지므로 판정하지 않는다
            if _box_ok(bgr, box):
                m = _cell_metrics(bgr, box)
                if m is not None:
                    cell = _classify(_cell_colors(m))
                    if not TL_USE_YOLO_CLASS:
                        obs = cell
                    elif cell is not None and cell == cls_name:
                        obs = cell          # 교차검증 일치 시에만 확정
        # N연속 확정 / 끊기면 hold 후 NONE
        prev = self.tl_state
        if obs is None:
            self._tl_cand, self._tl_cand_n = None, 0
            self._tl_miss += 1
            if self._tl_miss > 5:
                self.tl_state = None
        else:
            self._tl_miss = 0
            self._tl_cand_n = self._tl_cand_n + 1 if obs == self._tl_cand else 1
            self._tl_cand = obs
            if self._tl_cand_n >= 3:
                self.tl_state = obs
        if self.tl_state != prev:
            self._info(f"신호 {_OUT[prev]} → {_OUT[self.tl_state]}")

    # ── 정적장애물 ──────────────────────────────────────────────
    def _update_static(self, st_boxes):
        if st_boxes:
            self._n_static += 1

        # 가까운 static 박스가 하나라도 있으면 관측. 구간 판별은 하지 않는다.
        hit = any(b[3] >= STATIC_Y_BOT_MIN for _, b in st_boxes)
        self._st_n = self._st_n + 1 if hit else 0
        if self._st_n >= STATIC_CONFIRM:
            self._st_t = time.monotonic()
        prev_st = self.static
        self.static = (self._st_t is not None
                       and time.monotonic() - self._st_t <= STATIC_HOLD_SEC)
        if self.static != prev_st:
            self._info(f"정적 장애물 {'감지' if self.static else '해제'}")

    # ── 추월차량 ────────────────────────────────────────────────
    def _update_vehicle(self, veh_boxes, st_boxes, shape):
        h, w = shape[:2]
        hit = False
        for _, (x0, y0, x1, y1) in veh_boxes:
            bw, bh = x1 - x0, y1 - y0
            if bw <= 0 or bh <= 0:
                continue
            if bw * bh > VEH_MAX_AREA_RATIO * w * h:
                continue                       # 화면 대부분을 덮는 배경 오탐
            if bw / bh < VEH_WH_MIN:
                continue                       # 세로로 긴 박스 = 잘린 오탐
            if y1 < VEH_Y_BOT_MIN:
                continue                       # 아직 멀다
            if not (VEH_XC_MIN <= (x0 + x1) * 0.5 <= VEH_XC_MAX):
                continue                       # 이미 옆으로 지나가는 중
            # ★ 위에 static 이 얹혀 있으면 정적 장애물의 차체다 → 추월 대상 아님
            if any(sx0 < x1 and sx1 > x0 and sy1 >= y0 - 10
                   for _, (sx0, sy0, sx1, sy1) in st_boxes):
                continue
            hit = True
            break
        if veh_boxes:
            self._n_veh += 1
        self._veh_hist.append(hit)
        if sum(self._veh_hist) >= VEH_CONFIRM_N:
            self._veh_t = time.monotonic()
        prev = self.dynamic
        self.dynamic = (self._veh_t is not None
                        and time.monotonic() - self._veh_t <= VEH_HOLD_SEC)
        if self.dynamic != prev:
            self._info(f"전방 차량 {'감지' if self.dynamic else '해제'}")

    # ── 출력 ────────────────────────────────────────────────────
    def _result(self, expire_only=False):
        if expire_only:
            # 스킵 프레임에서도 시간 래치는 만료시킨다(hold 가 늘어나면 안 된다)
            now = time.monotonic()
            self.dynamic = (self._veh_t is not None
                            and now - self._veh_t <= VEH_HOLD_SEC)
            self.static = (self._st_t is not None
                           and now - self._st_t <= STATIC_HOLD_SEC)
        return {
            'traffic_light_color': _OUT[self.tl_state],
            'dynamic_detected': bool(self.dynamic),
            'static_detected': bool(self.static),
        }

    def _periodic_log(self):
        now = time.monotonic()
        dt = now - self._t_log
        if dt < LOG_PERIOD:
            return
        self._t_log = now
        n = max(self._n_run, 1)
        mode = 'active' if self._since_seen <= self.active_hold else 'idle'
        self._info(
            f"인지 [{mode}] TL={_OUT[self.tl_state]} "
            f"dyn={int(self.dynamic)} static={int(self.static)} | "
            f"검출 tl {100*self._n_tl/n:.0f}% / veh {100*self._n_veh/n:.0f}% "
            f"/ st {100*self._n_static/n:.0f}% | 지연 {self._lat/n:.0f}ms | "
            f"처리 {n/max(dt,1e-6):.1f}Hz (입력 {self._n_in/max(dt,1e-6):.1f}Hz)"
            + (f" | 실패 {self._err}" if self._err else ""))
        self._n_in = self._n_run = 0
        self._n_tl = self._n_veh = self._n_static = 0
        self._lat = 0.0
