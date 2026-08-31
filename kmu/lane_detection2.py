#!/usr/bin/env python3
"""
lane_detection4.py — 무상태(stateless) 검출 변형

lane_detection3 을 상속해 '검출'만 갈아끼우고 제어는 그대로 물려받는다.
제어 게인·평활·모드 판별이 완전히 동일하므로, 실차 A/B 에서 나타나는 차이는
전적으로 검출 방식에서 온다.

── 무엇이 무상태인가 ────────────────────────────────────────
탐색이 과거 fit 을 절대 참조하지 않는다. 매 프레임 히스토그램 W×Y pairing 으로
base 를 새로 잡고 코리도 fit 을 처음부터 다시 수행한다.
lane_detection3 의 margin_search(이전 fit ±80px 재탐색)는 완전히 제거했다.

  이번 캠페인에서 잡은 검출 버그는 대부분 '이전 fit 을 탐색에 되먹이는' 구조에서 나왔다:
    · 반사 락온      margin 밴드가 흰 반사를 흡수 → 밴드가 따라 이동 → 고착 (+53px 드리프트)
    · 가상 fit 자기 오염 루프
    · 쌍 붕괴 후 conf 고정 → 자력 회복 불가
  무상태 탐색에서는 이 셋이 원리적으로 성립할 수 없다.

  대가: 프레임 간 연속성을 탐색이 제공하지 않는다. 점선 gap·순간 가림을 매 프레임
  base 재획득으로 다시 풀어야 하므로 검출 실패율이 오를 수 있다.
  ★ 이 트레이드오프를 실차에서 재보려고 만든 파일이다.

── 의도적으로 상태를 남긴 곳 ──────────────────────────────
  · 제어 — 레이트 제한, 이동평균, D항, 모드 히스테리시스, 속도 슬루 (기저 클래스 그대로)
  · prev_left/right — 기저 process() 가 '검출 실패 프레임의 출력 유지'에만 쓴다.
    탐색에 되먹이지 않으므로 오염 루프가 생기지 않는다. (오염은 '탐색에 되먹임'이
    원인이었지 '값을 들고 있는 것' 자체가 아니었다.)

── 검출 파이프라인 ────────────────────────────────────────
  흰/노랑 히스토그램 → peak → W×Y 쌍 선택(간격 ≈ lane_width)
    → 각 측은 자기 색 채널만 본다
       pass1  base ±CORRIDOR_WIDE, 밴드 무게중심 → 저차 fit   (잡음에 강함)
       pass2  pass1 fit ±CORRIDOR_NARROW, 전체 픽셀 → 최종 fit (정밀)
    → 곡률/기하 위생 → 쌍 붕괴 검사 → 가상 차선 채움

  2-pass 인 이유: 슬라이딩 윈도우의 '재중심 walk'(윈도우가 한 칸씩 옆 잡음으로
  걸어가 인위적 곡률을 만들던 것, 실측 A=0.005~0.008)를 없애기 위해서다.
  코리도는 전역 fit 에 고정되므로 점진적으로 끌려가지 않는다.
  pass1 을 무게중심으로, pass2 를 전체 픽셀로 나눈 이유는 정밀도 실측 때문이다:
  무게중심만 쓰면 std 2.48px, 전체 픽셀이면 1.51px — 강건성은 pass1 이,
  정밀도는 pass2 가 담당한다.

사용: lane_contest/camera_node.py 에서 import 한 줄만 바꾸면 된다.
  from .lane_detection3 import LaneDetector   →   from .lane_detection4 import LaneDetector
"""

import numpy as np

from .lane_detection3 import (
    LaneDetector as _Base,
    polyfit_xy, _find_peaks,
    MIN_LANE_PIX, HIST_RATIO, LANE_WIDTH_PX, PEAK_MIN_MASS,
    PAIR_SEP_LO, PAIR_SEP_HI,
    LANE_MIN_SEP_RATIO,
)

# ── 히스토그램 평활 (무상태 전용) ──
# 무상태는 매 프레임 base 를 새로 잡아야 하므로 peak 검출 실패가 곧 검출 실패다.
# 기울어진 '점선'은 질량이 여러 컬럼에 흩어져 컬럼당 최대값이 임계 아래로 떨어진다.
#   실측(노랑 점선 컬럼 최대 질량):  psi=0.0 → 5100,  psi=0.6 → 3570,  psi=0.9 → 2295
#   박스 '합' 평활(k=31) 후:        psi=0.0 → 40800, psi=0.6 → 40800, psi=0.9 → 36720
# 즉 합 평활은 기울기 불변이 된다. lane_detection3 는 margin_search 가 있어 base 실패를
# 흡수하므로 이 문제가 드러나지 않았다 (기준선 유지를 위해 L3 는 건드리지 않는다).
HIST_SMOOTH_RATIO = 0.12          # 평활 창 폭 / lane_width (lane_width=260 → 31px)
# 합 평활 후 스케일에 맞춘 임계. 실측: 정상 점선 최저 36720 / 스펙클(10x4) 10200
# → 그 사이인 PEAK_MIN_MASS*8 을 쓴다 (양쪽으로 2배 이상 여유).
PEAK_MIN_MASS_SMOOTH = PEAK_MIN_MASS * 8

# ── 코리도 fit (차선폭 비율 — BEV/차선폭을 바꿔도 자동으로 따라온다) ──
# WIDE: pass1 코리도 반폭 / lane_width. 반대쪽 차선이 lane_width 거리에 있으므로
#   0.5 미만이어야 침범하지 않는다. 스윕 실측에서 0.25 가 무릎점(0.15 는 S자 검출률
#   100%→73% 로 붕괴, 0.25/0.35/0.45 는 전 지표 동일)이라 여유를 둬 0.30.
#   ※ 상단 이탈을 담으려 사다리꼴(위로 갈수록 넓어짐)도 시험했으나 어떤 지표에도
#     영향이 없어(GROW 0.0~0.5 전부 동일) 직사각으로 되돌렸다.
CORRIDOR_WIDE_RATIO   = 0.30
# NARROW: pass1 fit 에 고정돼 있으므로 좁아도 선을 놓치지 않는다. 좁을수록 인접 잡음 배제.
CORRIDOR_NARROW_RATIO = 0.20
N_BANDS      = 12    # pass1 밴드 수 (밴드 하나가 점선 대시보다 얇아야 gap 이 구분됨)
MIN_BAND_PX  = 10    # 밴드를 유효 점으로 인정할 최소 픽셀

# 시각화 호환: 튜너가 탐색 밴드 폭으로 이 이름을 읽는다 (기본 차선폭 기준 공칭값).
MARGIN_SEARCH = int(LANE_WIDTH_PX * CORRIDOR_NARROW_RATIO)


def _band_centroids(xs, ys, h, n=N_BANDS, min_px=MIN_BAND_PX):
    """가로 밴드별 무게중심. 노랑이 없는 밴드는 건너뛰므로 점선 gap 이 자동 무시된다."""
    bh = max(1, h // n)
    pts = []
    for i in range(n):
        lo, hi = i * bh, min(h, (i + 1) * bh)
        m = (ys >= lo) & (ys < hi)
        if int(m.sum()) < min_px:
            continue
        pts.append((float(xs[m].mean()), float(ys[m].mean())))
    return pts


def _fit_points(pts):
    """밴드 점 → [A,B,C]. 점이 적거나 y 범위가 좁으면 1차로 강등(곡률 오염 방지)."""
    if len(pts) < 3:
        return None
    px = np.array([p[0] for p in pts])
    py = np.array([p[1] for p in pts])
    order = 2 if ((py.max() - py.min()) >= 40 and len(pts) >= 4) else 1
    f = np.polyfit(py, px, order)
    return np.array([0.0, f[0], f[1]]) if order == 1 else f


class LaneDetector(_Base):
    """검출만 무상태로 교체. 제어·출력 규약은 lane_detection3 과 100% 동일."""

    # ------------------------------------------------
    #  base + 색 결정 (매 프레임 새로)
    # ------------------------------------------------
    def _paired_bases(self, white_mask, yellow_mask, h, vehicle_x):
        """
        W×Y peak pairing. 반환 ((left_x, left_color), (right_x, right_color)).

        하드룰 근거(트랙 기하): 유효 경계쌍은 항상 [흰 외곽선, 노란 중앙선] 이고
        흰 외곽선끼리는 2*lane_width 간격이라 흰+흰 정상 쌍은 존재할 수 없다.
        빛반사는 거의 흰 마스크로 들어오므로 반사가 낀 쌍은 여기서 배제된다.
        """
        lo = int(h * (1 - HIST_RATIO))
        k = max(3, int(self.lane_width * HIST_SMOOTH_RATIO)) | 1
        box = np.ones(k)

        def _peaks(mask):
            hist = np.sum(mask[lo:, :], axis=0).astype(np.float64)
            return _find_peaks(np.convolve(hist, box, mode='same'),
                               min_mass=PEAK_MIN_MASS_SMOOTH)

        pw = _peaks(white_mask)
        py = _peaks(yellow_mask)

        best, best_score = None, 0.0
        for xw, mw in pw:
            for xy, my in py:
                sep = abs(xw - xy)
                if not (self.lane_width * PAIR_SEP_LO < sep < self.lane_width * PAIR_SEP_HI):
                    continue
                # 질량 x 근접 우대 — 직선에서는 기대폭에 가까운 쌍이 이기고,
                # 커브에서는 후보가 하나뿐이라 가중치가 결과를 바꾸지 않는다.
                score = (mw + my) / (1.0 + abs(sep - self.lane_width) / self.lane_width)
                if score > best_score:
                    best_score = score
                    best = (((xw, 'W'), (xy, 'Y')) if xw < xy else ((xy, 'Y'), (xw, 'W')))
        if best is not None:
            return best

        # 쌍 실패 → 최강 peak 단일선. 반대쪽 base 를 ±lane_width 로 찍으면 마침 그
        # 자리에 있던 반사를 물어 잘못된 쌍이 부활하므로 일부러 비운다.
        cand = [(x, m, 'W') for x, m in pw] + [(x, m, 'Y') for x, m in py]
        if not cand:
            return (None, None), (None, None)
        x0, _m0, c0 = max(cand, key=lambda t: t[1])
        return ((x0, c0), (None, None)) if x0 < vehicle_x else ((None, None), (x0, c0))

    @staticmethod
    def _chan(color, white_mask, yellow_mask):
        if color == 'Y':
            return yellow_mask
        if color == 'W':
            return white_mask
        return None

    # ------------------------------------------------
    #  2-pass 코리도 fit
    # ------------------------------------------------
    def _corridor_fit(self, mask, base_x, h, side):
        if mask is None or base_x is None:
            self._dbg_search[side] = 'none'
            return None, None, None
        nz = mask.nonzero()
        ys, xs = nz[0], nz[1]
        if len(xs) == 0:
            self._dbg_search[side] = 'none'
            return None, None, None

        wide   = self.lane_width * CORRIDOR_WIDE_RATIO
        narrow = self.lane_width * CORRIDOR_NARROW_RATIO

        # pass1: base 중심 코리도 → 밴드 무게중심 → 저차 fit
        k1 = np.abs(xs - base_x) < wide
        x1, y1 = xs[k1], ys[k1]
        fit1 = _fit_points(_band_centroids(x1, y1, h))
        if fit1 is None:
            self._dbg_search[side] = 'none'
            return None, x1, y1

        # pass2: pass1 fit 주변 좁은 코리도 → 전체 픽셀 polyfit (정밀)
        k2 = np.abs(xs - np.polyval(fit1, ys)) < narrow
        x2, y2 = xs[k2], ys[k2]
        fit2 = polyfit_xy(y2, x2) if len(x2) >= MIN_LANE_PIX else None
        if fit2 is None:
            self._dbg_search[side] = 'p1'      # pass1 결과로 만족
            return fit1, x1, y1

        self._dbg_search[side] = 'p2'
        bh = max(1, h // N_BANDS)
        for i in range(N_BANDS):               # 디버그: 코리도 시각화
            yy = i * bh
            cx = int(np.polyval(fit2, yy + bh / 2))
            self._dbg_rects.append((int(cx - narrow), yy,
                                    int(cx + narrow), min(h, yy + bh)))
        return fit2, x2, y2

    # ------------------------------------------------
    #  검출 (기저 detect_lane 대체)
    # ------------------------------------------------
    def detect_lane(self, white_mask, yellow_mask):
        h, w = white_mask.shape
        vehicle_x = w // 2
        self._dbg_rects = []
        self._dbg_search = {'left': None, 'right': None}

        # 노란선 위치로 차로(1/2) 판별 — base 선정과 무관, 기저와 동일
        yx = yellow_mask.nonzero()[1]
        observed_lane = None
        if len(yx) > MIN_LANE_PIX:
            observed_lane = 1 if float(yx.mean()) > vehicle_x else 2

        (lb, lc), (rb, rc) = self._paired_bases(white_mask, yellow_mask, h, vehicle_x)
        self._col_left, self._col_right = lc, rc

        left_fit,  lx, ly = self._corridor_fit(
            self._chan(lc, white_mask, yellow_mask), lb, h, 'left')
        right_fit, rx, ry = self._corridor_fit(
            self._chan(rc, white_mask, yellow_mask), rb, h, 'right')

        # 위생: prev 가 없으므로 급변 검사는 성립하지 않고, 비물리적 곡률만 본다
        left_fit  = self._reject_bad_fit(left_fit,  None, w, h)
        right_fit = self._reject_bad_fit(right_fit, None, w, h)

        self._left_px  = (lx, ly) if left_fit  is not None else None
        self._right_px = (rx, ry) if right_fit is not None else None

        real_left  = left_fit  is not None
        real_right = right_fit is not None

        # 좌우 쌍 붕괴 — 무상태에서도 한 프레임 안에서 두 코리도가 같은 선을
        # 물 수 있으므로(예: 한쪽 색 채널이 비어 단일 강등된 직후) 그대로 검사한다.
        if real_left and real_right:
            sep = float(np.polyval(right_fit, h - 1) - np.polyval(left_fit, h - 1))
            if sep < self.lane_width * LANE_MIN_SEP_RATIO:
                left_fit = right_fit = None
                self._left_px = self._right_px = None
                real_left = real_right = False
                self._pair_reject += 1
                self._dbg_search['left'] = self._dbg_search['right'] = 'reject'

        real_left_fit  = left_fit  if real_left  else None
        real_right_fit = right_fit if real_right else None

        left_fit, right_fit = self._fill_missing(left_fit, right_fit)
        return (left_fit, right_fit, observed_lane, (real_left, real_right),
                (real_left_fit, real_right_fit))