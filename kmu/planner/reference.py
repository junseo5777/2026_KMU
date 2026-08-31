#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planner/reference.py — 모드별 레퍼런스 생성기.

3-2절: 모드가 바꾸는 것은 코스트 함수가 아니라 (a) 레퍼런스 생성기 (b) 제약
(c) 가중치다. 그래서 lattice 본체는 레퍼런스가 어디서 왔는지 몰라야 하고,
``Reference.lateral_at(x)`` 하나만 알면 되게 만든다.

두 생성기
  lane_center : 카메라가 준 (d, psi, kappa_lane) 로 차로중심선을 세운다.
  gap_or_gate : 중심선을 만들지 않는다. Follow-The-Gap 으로 목표 '방향'만
                내고, 콘은 전적으로 하드 제약(R_safe)이 막는다.

★ 입력은 클러스터 중심이 아니라 raw scan 점군이다 ★
  클러스터링은 히스토리 누적(ridar.py 의 min_positive_frames)까지 거쳐야
  판정이 나와 반응이 늦다. DWA 가 raw scan 을 쓰는 이유가 그것이고,
  lattice 도 같은 입력을 쓴다.

★ 콘 구간에 중심선을 세우지 않는 이유 ★
  콘을 짝지어 게이트를 만들거나 좌우 벽을 피팅하려면 어느 쪽이든 점군을
  구조화해야 하고, 그 추정이 틀리면 중심선이 통째로 어긋난다. 회랑을
  통과하는 데 필요한 정보는 이미 하드 제약이 다 갖고 있다 — 콘에서
  R_safe 안으로 들어오는 후보는 전부 탈락하므로, 살아남은 후보는 정의상
  회랑 안을 지나간다. 남은 선택은 '그중 어느 것이 가장 곧고 이전 해와
  이어지는가'이고 그건 w_gap, w_cons, W_CURV 가 결정한다.

좌표 규약은 params.py 와 동일하다 — 바디프레임 +x 전방, +y 좌측.
"""

import math

import numpy as np

from . import params as P


# Follow-The-Gap 은 '지금 보이는 범위의 목표 방향'이라 멀리까지 믿을 수 없다.
# 이 거리 너머로는 레퍼런스를 외삽하지 않고 마지막 값을 유지한다.
FTG_VALID_RANGE_M = 1.5


class Reference:
    """
    lattice 가 추종을 선호할 중심선.

    ``coeffs`` 는 y_ref(x) = c0 + c1 x + c2 x^2 형태의 계수다. 2차까지만 쓰는
    이유는 카메라가 주는 정보(횡오차·헤딩·곡률)가 딱 3개라서 그 이상은
    근거 없는 외삽이 되기 때문이다.
    """

    __slots__ = ('coeffs', 'theta_gap', 'valid', 'source',
                 'confidence', 'max_x')

    def __init__(self, coeffs=(0.0, 0.0, 0.0), theta_gap=0.0, valid=True,
                 source='', confidence=1.0, max_x=np.inf):
        """계수와 목표 방향, 신뢰도, 유효 거리를 보관한다."""
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.theta_gap = float(theta_gap)
        self.valid = bool(valid)
        self.source = source
        self.confidence = float(confidence)
        self.max_x = float(max_x)

    def lateral_at(self, x):
        """
        레퍼런스 중심선의 횡변위 y_ref(x).

        ``max_x`` 밖에서는 외삽하지 않고 마지막 값을 유지한다.
        Follow-The-Gap 은 '지금 보이는 범위의 목표 방향'일 뿐인데 이를 3 m 까지
        직선 외삽하면 -39 도 방향으로 2.5 m 떨어진 곳을 종점으로 삼는
        말이 안 되는 후보가 만들어진다. 게이트 다항식도 마지막 게이트 너머로
        외삽하면 근거 없는 곡선이 이어진다.
        """
        c = self.coeffs
        clipped = np.minimum(np.asarray(x, dtype=np.float64), self.max_x)
        return c[0] + c[1] * clipped + c[2] * clipped * clipped


# ══════════════════════════════════════════════════════════════════
#  lane_center
# ══════════════════════════════════════════════════════════════════


def build_lane_reference(lateral_m, heading_rad, kappa_lane, confidence=1.0):
    """
    카메라 측정값으로 차로중심선을 세운다.

    ``lateral_m`` 은 차량 기준 차로중심의 횡변위(좌측 양수)이고,
    ``heading_rad`` 은 차로중심선의 헤딩, ``kappa_lane`` 은 그 곡률이다.

    부호 주의 — lane_detection3 의 lat_err 는 'BEV 픽셀에서 차로중심이
    이미지중심 대비 얼마나 오른쪽인가'라서 부호와 단위가 모두 다르다.
    변환은 Phase 3 에서 차선 노드가 직접 (d, psi, kappa) 를 발행하게 만들면서
    한 곳에 몰아넣는다. 여기서는 이미 바디프레임 미터로 온다고 본다.
    """
    return Reference(
        coeffs=(lateral_m, math.tan(heading_rad), 0.5 * kappa_lane),
        theta_gap=heading_rad,
        valid=True,
        source='lane_center',
        confidence=confidence,
    )


# ══════════════════════════════════════════════════════════════════
#  gap_or_gate
# ══════════════════════════════════════════════════════════════════


def follow_the_gap(obstacles, max_range_m=3.0, half_fov_rad=math.pi / 2.0):
    """
    목표 진행 방향을 뽑는다 (Sezer & Gokasan 2012).

    장애물 방위각을 정렬해 **장애물 사이**의 가장 넓은 빈 구간을 찾고 그
    중앙을 목표 방향으로 준다.

    ★ 시야 가장자리를 경계로 넣으면 안 된다 ★
    회랑에서는 좌우 벽 바깥쪽(시야 끝까지)이 아무것도 없는 넓은 구간이라,
    가장자리를 후보에 넣으면 그쪽이 항상 이긴다. 실제로 회랑 정면을 두고
    -55도를 목표로 내놨다. 장애물 사이 구간만 보면 좌우 벽의 안쪽 점
    사이 간격 = 회랑 입구가 자연히 최대가 된다.

    장애물이 1개 이하라 '사이'가 없으면, 그 점을 피하는 쪽을 준다.
    """
    obstacles = np.asarray(obstacles, dtype=np.float64).reshape(-1, 2)
    if len(obstacles) == 0:
        return 0.0

    distance = np.hypot(obstacles[:, 0], obstacles[:, 1])
    angle = np.arctan2(obstacles[:, 1], obstacles[:, 0])
    keep = (distance <= max_range_m) & (np.abs(angle) <= half_fov_rad)
    angle = np.sort(angle[keep])
    if len(angle) == 0:
        return 0.0
    if len(angle) == 1:
        # 단일 장애물 — 시야가 더 남는 쪽으로 비켜간다
        margin = half_fov_rad - abs(angle[0])
        return float(angle[0] + math.copysign(0.5 * margin, -angle[0]))

    widths = np.diff(angle)
    best = int(np.argmax(widths))
    return float(0.5 * (angle[best] + angle[best + 1]))


def build_gap_reference(points):
    """
    Raw scan 점군에서 목표 방향만 뽑는다 (중심선 없음).

    중심선 계수는 0 으로 둔다. 그러면 후보의 종점 앵커가 차량 바디축이 되고,
    ``ref_deviation`` 도 '바디축에서 얼마나 벗어나는가'가 된다. CONE 모드는
    ``w_ref=0`` 이라 그 항 자체를 쓰지 않으므로 문제되지 않는다.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    theta = follow_the_gap(points)
    return Reference(
        coeffs=(0.0, 0.0, 0.0),
        theta_gap=theta,
        valid=True,
        source='ftg',
        confidence=1.0 if len(points) else 0.0,
    )


# ══════════════════════════════════════════════════════════════════
#  통합 진입점
# ══════════════════════════════════════════════════════════════════


def build_reference(ref_source, lane=None, cones=None, obstacles=None):
    """
    모드의 ``ref_source`` 문자열에 맞는 레퍼런스를 만든다.

    ``lane`` 은 (lateral_m, heading_rad, kappa_lane, confidence) 튜플이다.
    """
    if ref_source == 'lane_center':
        if lane is None:
            return Reference(valid=False, source='lane_center')
        return build_lane_reference(*lane)
    if ref_source == 'gap_or_gate':
        points = obstacles if obstacles is not None else cones
        return build_gap_reference(
            np.zeros((0, 2)) if points is None else points
        )
    raise ValueError(f'unknown ref_source: {ref_source!r}')


def lane_bounds(passing_side=None):
    """
    자차선 중심 기준 횡방향 하드 제약을 만든다 (미터).

    ``passing_side`` 가 None 이면 자차선만(LANE_KEEP), 'left'/'right' 면
    그쪽 옆차선까지 허용한다(OVERTAKE).

    3-2절의 자리표시자는 좌우 대칭이었지만, 옆차선은 한쪽에만 있으므로
    실제 제약은 비대칭이다. 도로 전체 폭이 0.80 m 뿐이라 대칭 ±1.15 는
    물리적으로 성립하지 않는다.
    """
    inner = P.LANE_GEOMETRIC_MARGIN_M + P.LANE_DEPART_TOLERANCE_M
    if passing_side is None:
        return (-inner, inner)

    # 옆차선 끝까지: 자차선 반폭 + 옆차선 폭 - 차량 반폭 + 허용 이탈
    outer = (P.LANE_WIDTH_M / 2.0 + P.LANE_WIDTH_M
             - P.VEHICLE_HALF_WIDTH_M + P.LANE_DEPART_TOLERANCE_M)
    # 바디프레임에서 좌측이 +y 다. 좌측 차선으로 나가려면 상한을 넓혀야 한다.
    if passing_side == 'left':
        return (-inner, outer)
    if passing_side == 'right':
        return (-outer, inner)
    raise ValueError(f'passing_side must be None/left/right, got {passing_side!r}')
