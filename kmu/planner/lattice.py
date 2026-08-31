#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planner/lattice.py — 바디프레임 quintic lattice 플래너 본체.

★ 플래너는 단 하나다. 모드별로 함수를 복사하지 않는다 ★
모드가 바꾸는 것은 (a) 레퍼런스 생성기 (b) 하드 제약 (c) 가중치뿐이고,
후보 생성·필터·코스트·선택 로직은 세 모드가 그대로 공유한다.

하드 제약과 코스트를 구분하는 이유 (3-1절)
  차선을 ``w_lane * d^2`` 같은 코스트로 넣으면, 가중치가 크면 콘을 못 피하고
  작으면 차선을 넘는다. 그 사이에 안전한 값이 존재한다는 보장이 없다.
  그래서 차선·충돌·곡률은 전부 **위반 시 즉시 탈락**으로 처리하고,
  살아남은 후보끼리만 코스트로 비교한다.
"""

from dataclasses import dataclass, replace

import numpy as np

from . import params as P
from .quintic import QuinticEdge
from .quintic import batch_edges, curvature_energy


# 탈락 사유 — replay 도구가 색으로 구분해서 그린다.
REJECT_BOUNDS = 'bounds'
REJECT_CLEARANCE = 'clearance'
REJECT_CURVATURE = 'curvature'


@dataclass(frozen=True)
class PlanMode:
    """
    모드 정의. 플래너 동작을 바꾸는 파라미터만 담는다.

    ``d_bounds`` 는 기본값이고, OVERTAKE 처럼 옆차선 방향에 따라 비대칭이
    되는 경우 ``dataclasses.replace(mode, d_bounds=...)`` 로 런타임에 덮어쓴다.
    """

    name: str
    ref_source: str        # 'lane_center' | 'gap_or_gate'
    d_bounds: tuple        # 횡방향 하드 제약. ★ 레퍼런스 중심선 기준 ★ (m)
    w_ref: float           # 레퍼런스 추종 선호
    w_gap: float           # 전방 진행(gap) 정렬 선호
    w_cons: float          # 이전 해와의 일관성 — 0 으로 두지 말 것
    R_safe: float          # 장애물 안전반경 (m)
    v_target: float        # 목표 속도 (xycar 명령 단위, m/s 아님)


# ── 모드 인스턴스 ────────────────────────────────────────────────
#
# R_safe 근거
#   콘  : 차량반폭 0.150 + 콘반폭 0.0969 + 추종오차 0.05 + 인지오차 0.03
#   차량: 차량반폭 0.150 + 상대차반폭 0.115(ridar.py 차량 폭 게이트 0.23 의 절반)
#         + 추종오차 0.05 + 인지오차 0.03
#   ⚠ 추종오차는 Phase 4 실측 전 잠정값이다 (params.TRACKING_ERROR_M).
_R_SAFE_CAR_M = (
    P.VEHICLE_HALF_WIDTH_M + 0.115
    + P.TRACKING_ERROR_M + P.PERCEPTION_ERROR_M
)

# v_target 근거: 기존 track_drive/state.py 가 쓰던 xycar 명령 단위 속도.
#   LANE_KEEP 은 카메라 노드가 주는 target_speed 를 그대로 쓰므로 여기 값은
#   lattice 로 전환했을 때의 기본값이다 (3-7절 3단계 전까지는 미사용).
LANE_KEEP = PlanMode(
    name='LANE_KEEP',
    ref_source='lane_center',
    d_bounds=(-(P.LANE_GEOMETRIC_MARGIN_M + P.LANE_DEPART_TOLERANCE_M),
              +(P.LANE_GEOMETRIC_MARGIN_M + P.LANE_DEPART_TOLERANCE_M)),
    w_ref=8.0,
    w_gap=0.0,
    w_cons=2.0,
    R_safe=_R_SAFE_CAR_M,
    v_target=9.0,
)

# d_bounds 는 자리표시자다. 실제로는 reference.lane_bounds(passing_side) 로
# 옆차선 방향을 넣어 런타임에 replace 한다.
OVERTAKE = PlanMode(
    name='OVERTAKE',
    ref_source='lane_center',
    d_bounds=(-0.10, 0.50),
    w_ref=3.0,
    w_gap=0.0,
    w_cons=4.0,
    R_safe=_R_SAFE_CAR_M,
    v_target=5.5,
)

# CONE 은 중심선을 만들지 않는다 (reference.py 헤더 참조). 콘은 전적으로
# 하드 제약이 막고, 살아남은 후보 중 선택은 방향 정렬·일관성·곡률이 한다.
CONE = PlanMode(
    name='CONE',
    ref_source='gap_or_gate',
    d_bounds=(-np.inf, np.inf),   # 콘 구간은 차선이 무의미하다
    w_ref=0.0,
    w_gap=6.0,
    w_cons=3.0,
    R_safe=P.R_SAFE_CONE_M,
    v_target=4.0,
)


@dataclass
class VehicleState:
    """플래너가 필요로 하는 차량 상태."""

    kappa_cur: float = 0.0    # 현재 곡률 (1/m). 경계조건 y''(0) 에 들어간다
    v_mps: float = 0.0        # 현재 속도 (m/s). 추종기에만 쓰인다


class Candidate:
    """후보 경로 하나와 그 평가 결과."""

    __slots__ = ('edge', 'x', 'y', 'yaw', 'kappa', 's',
                 'reject', 'cost', 'terms')

    def __init__(self, edge, samples):
        """Edge 와 이산화 결과를 묶는다."""
        self.edge = edge
        self.x, self.y, self.yaw, self.kappa, self.s = samples
        self.reject = None
        self.cost = np.inf
        self.terms = {}

    @property
    def accepted(self):
        """하드 제약을 모두 통과했는지."""
        return self.reject is None


class PlanResult:
    """한 사이클의 계획 결과."""

    __slots__ = ('best', 'candidates', 'mode', 'reference')

    def __init__(self, best, candidates, mode, reference):
        """선택된 후보와 전체 후보군을 함께 보관한다 (replay 용)."""
        self.best = best
        self.candidates = candidates
        self.mode = mode
        self.reference = reference

    @property
    def success(self):
        """살아남은 후보가 있었는지."""
        return self.best is not None

    def reject_counts(self):
        """탈락 사유별 개수."""
        counts = {REJECT_BOUNDS: 0, REJECT_CLEARANCE: 0,
                  REJECT_CURVATURE: 0, 'accepted': 0}
        for candidate in self.candidates:
            counts[candidate.reject or 'accepted'] += 1
        return counts


# ══════════════════════════════════════════════════════════════════
#  후보 생성
# ══════════════════════════════════════════════════════════════════


def lateral_offsets():
    """레퍼런스 중심선 기준 횡방향 종점 오프셋 (m)."""
    return np.linspace(-P.LATERAL_SPAN_M, P.LATERAL_SPAN_M,
                       P.LATERAL_TARGETS_M)


def generate_candidates(kappa_cur, reference):
    """
    (horizon, 레퍼런스 기준 오프셋, psi_T) 격자로 후보를 만든다.

    ★ 오프셋은 바디축이 아니라 레퍼런스 중심선 기준이다 (Werling 2010) ★
    바디축 기준으로 뽑으면 레퍼런스가 옆으로 밀려 있을 때 그 위를 정확히
    지나는 후보가 격자에 없다. 실제로 게이트 중점을 짚는 후보가 없어
    최선 후보의 여유가 기하 최적보다 30 mm 나 부족했고, 오차 예산이 75 mm
    뿐인 회랑에서는 그 30 mm 가 통과/실패를 갈랐다.
    offset=0 후보는 정의상 레퍼런스 위에 정확히 놓인다.

    조합 수 = 3 x 13 x 3 = 117 개. quintic 계수 계산은 캐싱된 3x3 역행렬
    곱셈 하나뿐이라 이 정도는 10 Hz 에서 충분히 싸다.
    """
    candidates = []
    offsets = lateral_offsets()
    for horizon in P.HORIZON_LAYERS_M:
        anchor = float(reference.lateral_at(horizon))
        coeffs, d_flat, psi_flat, x, y, yaw, kappa, s = batch_edges(
            horizon, anchor + offsets, P.TERMINAL_HEADINGS_RAD, kappa_cur,
            P.CANDIDATE_SAMPLES, P.EVAL_HORIZON_M,
        )
        for index in range(len(d_flat)):
            edge = QuinticEdge(coeffs[index], horizon, float(d_flat[index]),
                               float(psi_flat[index]), kappa_cur)
            candidates.append(Candidate(
                edge,
                (x, y[index], yaw[index], kappa[index], s[index]),
            ))
    return candidates


# ══════════════════════════════════════════════════════════════════
#  하드 제약
# ══════════════════════════════════════════════════════════════════


def violates_bounds(candidate, d_bounds, reference):
    """
    경로 어느 지점이든 횡방향 제약을 벗어나면 True.

    ★ 제약은 레퍼런스 중심선 기준이다 ★ 바디축 기준으로 재면, 차가 차로중심에서
    0.06 m 밀려 있을 때 실제 허용 범위가 [-0.04, +0.16] 인데 [-0.10, +0.10] 로
    잘못 걸려 차로 안쪽으로 되돌아오는 후보까지 탈락시킨다.

    ★ 이미 제약 밖에 있는 경우를 따로 다룬다 ★
    모든 후보는 차량 현재 위치에서 시작하므로, 차가 이미 차선을 넘어 있으면
    x=0 에서 전부 위반이 되어 후보가 하나도 남지 않는다. 편측 여유가 5 cm
    뿐인 이 코스에서는 인지·추종 오차만으로도 쉽게 벌어지는 상황이고,
    그때마다 Fallback 으로 퇴각하면 영원히 복귀하지 못한다.
    그래서 "밖에서 시작해 안으로 들어온 뒤 다시 나가지 않는" 경로는 허용한다.
    지금 위치는 바꿀 수 없고 되돌아오는 것이 우리가 원하는 동작이기 때문이다.
    """
    low, high = d_bounds
    offset = candidate.y - reference.lateral_at(candidate.x)
    inside = (offset >= low) & (offset <= high)
    if inside.all():
        return False
    if not inside.any():
        return True                      # horizon 안에 복귀하지 못한다
    entry = int(np.argmax(inside))        # 처음 제약 안으로 들어오는 지점
    return not bool(inside[entry:].all())  # 들어온 뒤 다시 나가면 위반


def min_clearance(candidate, obstacles):
    """경로 표본과 장애물 사이의 최소 거리. 장애물이 없으면 inf."""
    if obstacles is None or len(obstacles) == 0:
        return np.inf
    obstacles = np.asarray(obstacles, dtype=np.float64).reshape(-1, 2)
    dx = candidate.x[:, np.newaxis] - obstacles[np.newaxis, :, 0]
    dy = candidate.y[:, np.newaxis] - obstacles[np.newaxis, :, 1]
    return float(np.min(np.hypot(dx, dy)))


def all_clearances(candidates, obstacles):
    """
    모든 후보의 최소 여유를 한 번에 계산한다.

    후보마다 따로 부르면 numpy 호출이 후보 수만큼(378번) 나서, 스캔 점이
    수백 개인 raw scan 입력에서 계획 시간의 대부분을 먹는다. 후보 표본을
    한 배열로 쌓아 거리 행렬을 한 번만 만든다.
    ``obstacles`` 가 없으면 전부 inf.
    """
    if obstacles is None or len(obstacles) == 0:
        return np.full(len(candidates), np.inf)
    obstacles = np.asarray(obstacles, dtype=np.float64).reshape(-1, 2)

    # 후보마다 표본 개수가 다를 수 있어(꼬리 길이) 이어붙인 뒤 구간별로 자른다.
    lengths = np.array([len(c.x) for c in candidates])
    xs = np.concatenate([c.x for c in candidates])
    ys = np.concatenate([c.y for c in candidates])
    dx = xs[:, np.newaxis] - obstacles[np.newaxis, :, 0]
    dy = ys[:, np.newaxis] - obstacles[np.newaxis, :, 1]
    per_sample = np.min(dx * dx + dy * dy, axis=1)

    # 후보별 구간 최소값. 파이썬 반복문으로 378번 슬라이싱하면 그것만
    # 20 ms 가 나온다. reduceat 이 구간 축약을 numpy 안에서 한 번에 한다.
    starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    return np.sqrt(np.minimum.reduceat(per_sample, starts))


def max_curvature(candidate):
    """경로의 최대 곡률 크기."""
    return float(np.max(np.abs(candidate.kappa)))


# ══════════════════════════════════════════════════════════════════
#  코스트
# ══════════════════════════════════════════════════════════════════


def ref_deviation(candidate, reference):
    """레퍼런스 중심선과의 평균 제곱 거리."""
    gap = candidate.y - reference.lateral_at(candidate.x)
    return float(np.mean(gap * gap))


def lateral_diff(candidate, prev_path):
    """
    이전 사이클 해와의 평균 제곱 횡방향 차이.

    ★ 이 항이 3-4절의 w_cons 다. 옵션이 아니다 ★
    좌/우 회피 코스트가 정확히 대칭이 되는 순간(콘이 정면)은 반드시 오고,
    그때 이 항이 없으면 매 프레임 답이 뒤집히며 조향이 떤다.

    ``prev_path`` 는 **현재 바디프레임으로 옮겨진** 이전 경로여야 한다.
    프레임 이동을 호출자가 책임진다 (Phase 5 에서 주행거리로 시프트).
    """
    if prev_path is None:
        return 0.0
    overlap = np.minimum(candidate.x, prev_path.horizon)
    gap = candidate.y - prev_path.lateral(overlap)
    return float(np.mean(gap * gap))


def evaluate(candidate, mode, obstacles, reference, prev_path,
             clearance=None):
    """
    후보 하나를 평가한다. 하드 제약 위반이면 reject 를 채우고 코스트는 inf.

    3-4절 그대로: 제약은 코스트가 아니라 탈락이다.
    ``clearance`` 를 주면 다시 계산하지 않는다 (all_clearances 결과 재사용).
    """
    if violates_bounds(candidate, mode.d_bounds, reference):
        candidate.reject = REJECT_BOUNDS
        return candidate
    if clearance is None:
        clearance = min_clearance(candidate, obstacles)
    if clearance < mode.R_safe:
        candidate.reject = REJECT_CLEARANCE
        return candidate
    if max_curvature(candidate) > P.KAPPA_MAX:
        candidate.reject = REJECT_CURVATURE
        return candidate

    terms = {
        'ref': mode.w_ref * ref_deviation(candidate, reference),
        'gap': mode.w_gap * (candidate.edge.psi_target
                             - reference.theta_gap) ** 2,
        'cons': mode.w_cons * lateral_diff(candidate, prev_path),
        'curv': P.W_CURV * curvature_energy(candidate.kappa, candidate.s),
    }
    candidate.terms = terms
    candidate.cost = float(sum(terms.values()))
    return candidate


# ══════════════════════════════════════════════════════════════════
#  진입점
# ══════════════════════════════════════════════════════════════════


def plan(state, mode, obstacles, reference, prev_path=None):
    """
    한 사이클 계획. 살아남은 후보 중 코스트 최소를 고른다.

    모든 후보가 탈락하면 ``PlanResult.success`` 가 False 다. 이 경우는 반드시
    오므로 호출자가 3-5절 (d) Fallback 을 준비해야 한다 (감속 + 이전 경로 유지,
    또는 기존 DWA 로 인계).
    """
    candidates = generate_candidates(state.kappa_cur, reference)
    clearances = all_clearances(candidates, obstacles)
    best = None
    for candidate, clearance in zip(candidates, clearances):
        evaluate(candidate, mode, obstacles, reference, prev_path, clearance)
        if candidate.accepted and (best is None or candidate.cost < best.cost):
            best = candidate
    return PlanResult(best, candidates, mode, reference)


def with_bounds(mode, d_bounds):
    """Frozen 모드의 d_bounds 만 런타임에 바꾼 사본을 만든다."""
    return replace(mode, d_bounds=tuple(d_bounds))


def with_r_safe(mode, r_safe):
    """Frozen 모드의 R_safe 만 바꾼 사본을 만든다 (오차 예산 탐색용)."""
    return replace(mode, R_safe=float(r_safe))
