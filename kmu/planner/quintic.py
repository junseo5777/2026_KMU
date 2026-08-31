#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planner/quintic.py — 바디프레임 5차 다항식 edge 생성.

Global path 가 없으므로 레퍼런스를 차량 자신의 헤딩 축(직선)으로 둔다.
그러면 Frenet 좌표계가 그냥 바디프레임이 되고, 전역 localization 없이
2~3 m horizon 안에서만 유효한 경로를 매 사이클 다시 만들면 된다.

    y(x) = a0 + a1 x + a2 x^2 + a3 x^3 + a4 x^4 + a5 x^5

경계조건 (3-3절)
    y(0)   = 0              현재 위치
    y'(0)  = 0              현재 헤딩 (바디프레임이므로 정의상 0)
    y''(0) = kappa_cur      현재 곡률 — 이전 사이클과 곡률이 이어져야 조향이 안 튄다
    y(X)   = d_T            종점 횡변위
    y'(X)  = tan(psi_T)     종점 헤딩
    y''(X) = 0              종점에서 곡률 0 (직선으로 수렴)

한계 — y(x) 함수 형태라 horizon 안에서 헤딩이 ±90 도를 넘는 기동은 표현할 수 없다.
    x 가 단조 증가해야 하기 때문이다. 이 코스에서는 문제가 되지 않는다:
    최대 후보인 d_T=0.5, X=1.2 조합의 최대 헤딩이 약 40 도이고, 그마저도
    곡률 제약(kappa_max=1.650)에 걸려 탈락한다. 제자리 회전이나 후진이
    필요한 상황(주차 등)에는 이 표현을 쓸 수 없다.
"""

import math

import numpy as np


# y''(0)=kappa_cur 까지가 a0, a1, a2 로 직접 결정되므로 실제로 풀어야 하는 것은
# a3, a4, a5 세 개뿐이다. 3-3절은 6x6 을 캐싱하라고 했지만, 3x3 으로 줄이면
# 수학적으로 동일하면서 역행렬 계산과 곱셈이 모두 싸다.
# 행렬은 X 에만 의존하므로 X 값마다 한 번만 만들어 캐싱한다.
_INVERSE_CACHE = {}

# 표본 x 격자도 X 와 점 개수에만 의존하므로 캐싱한다. 후보 378개가 매 사이클
# 같은 격자를 다시 만들면 linspace 호출만 11000 번이 되어 계획 시간의 절반을
# 잡아먹는다 (프로파일 실측).
_GRID_CACHE = {}
_TAIL_CACHE = {}


def _sample_grid(horizon, count):
    """X 표본 격자를 (X, 점 개수) 별로 캐싱해서 돌려준다."""
    key = (round(float(horizon), 6), int(count))
    grid = _GRID_CACHE.get(key)
    if grid is None:
        grid = np.linspace(0.0, float(horizon), int(count))
        _GRID_CACHE[key] = grid
    return grid


def _tail_grid(horizon, count, eval_x):
    """꼬리 구간의 x 증분 배열을 캐싱해서 돌려준다."""
    key = (round(float(horizon), 6), int(count), round(float(eval_x), 6))
    grid = _TAIL_CACHE.get(key)
    if grid is None:
        tail_x = float(eval_x) - float(horizon)
        tail_count = max(2, int(round(count * tail_x / horizon)))
        grid = np.linspace(0.0, tail_x, tail_count + 1)[1:]
        _TAIL_CACHE[key] = grid
    return grid


def _terminal_inverse(horizon):
    """종점 조건 3x3 선형계의 역행렬을 X 별로 캐싱해서 돌려준다."""
    key = round(float(horizon), 6)
    cached = _INVERSE_CACHE.get(key)
    if cached is not None:
        return cached

    x = float(horizon)
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    matrix = np.array([
        [x3, x4, x5],                  # y(X)
        [3.0 * x2, 4.0 * x3, 5.0 * x4],  # y'(X)
        [6.0 * x, 12.0 * x2, 20.0 * x3],  # y''(X)
    ], dtype=np.float64)
    inverse = np.linalg.inv(matrix)
    _INVERSE_CACHE[key] = inverse
    return inverse


class QuinticEdge:
    """바디프레임 5차 다항식 경로 한 개."""

    __slots__ = ('coeffs', 'horizon', 'd_target', 'psi_target', 'kappa_start')

    def __init__(self, coeffs, horizon, d_target, psi_target, kappa_start):
        """계수와 생성 파라미터를 보관한다."""
        self.coeffs = coeffs           # [a0..a5]
        self.horizon = horizon
        self.d_target = d_target
        self.psi_target = psi_target
        self.kappa_start = kappa_start

    def lateral(self, x):
        """횡변위 y(x)."""
        a = self.coeffs
        return ((((a[5] * x + a[4]) * x + a[3]) * x + a[2]) * x + a[1]) * x + a[0]

    def slope(self, x):
        """기울기 y'(x) = tan(헤딩)."""
        a = self.coeffs
        return (((5.0 * a[5] * x + 4.0 * a[4]) * x + 3.0 * a[3]) * x
                + 2.0 * a[2]) * x + a[1]

    def second(self, x):
        """이차미분 y''(x)."""
        a = self.coeffs
        return ((20.0 * a[5] * x + 12.0 * a[4]) * x + 6.0 * a[3]) * x + 2.0 * a[2]

    def curvature(self, x):
        """
        호길이 기준 곡률 kappa(x) = y'' / (1 + y'^2)^(3/2).

        y'' 를 그대로 곡률로 쓰면 안 된다. 헤딩이 커질수록 실제 곡률은
        y'' 보다 작아지는데(분모>1), 그 차이는 psi=0.35 rad 에서 벌써 6% 다.
        곡률 제약을 y'' 로 걸면 통과 가능한 후보를 버리게 된다.
        """
        slope = self.slope(x)
        return self.second(x) / np.power(1.0 + slope * slope, 1.5)

    def heading(self, x):
        """헤딩 psi(x) = atan(y')."""
        return np.arctan(self.slope(x))

    def sample(self, count):
        """
        경로를 count 점으로 이산화해 (x, y, yaw, kappa, s) 를 돌려준다.

        s 는 호길이 누적값이다. 추종기가 '전방 LD_FF 미터 지점의 곡률'을
        찾을 때 x 가 아니라 s 로 찾아야 속도와 무관하게 일관된다.
        """
        x = _sample_grid(self.horizon, count)
        y = self.lateral(x)
        slope = self.slope(x)
        yaw = np.arctan(slope)
        kappa = self.second(x) / np.power(1.0 + slope * slope, 1.5)
        # 호길이: ds = sqrt(1 + y'^2) dx 를 사다리꼴 적분
        ds = np.sqrt(1.0 + slope * slope)
        step = np.diff(x)
        s = np.concatenate((
            [0.0], np.cumsum(0.5 * (ds[:-1] + ds[1:]) * step)
        ))
        return x, y, yaw, kappa, s


def quintic_edge(horizon, d_target, psi_target, kappa_start):
    """
    경계조건을 만족하는 5차 다항식 edge 를 만든다.

    horizon(X), d_target(d_T), psi_target(psi_T) 는 미터/라디안이고,
    kappa_start 는 현재 곡률(1/m)이다.
    """
    if horizon <= 0.0:
        raise ValueError('horizon must be positive')

    a0 = 0.0
    a1 = 0.0
    a2 = 0.5 * kappa_start

    x = float(horizon)
    residual = np.array([
        d_target - a2 * x * x,
        np.tan(psi_target) - 2.0 * a2 * x,
        -2.0 * a2,
    ], dtype=np.float64)
    a3, a4, a5 = _terminal_inverse(x) @ residual

    coeffs = np.array([a0, a1, a2, a3, a4, a5], dtype=np.float64)
    return QuinticEdge(coeffs, x, d_target, psi_target, kappa_start)


def sample_with_tail(edge, count, eval_x):
    """
    후보를 공통 평가거리 ``eval_x`` 까지 늘려 이산화한다.

    ★ 이게 없으면 짧은 레이어가 구조적으로 유리해진다 ★
    X=1.2 후보는 x=1.8 에 있는 콘을 자기 horizon 밖이라 아예 보지 못해
    충돌 검사를 통과해 버리고, 곡률 에너지도 짧은 만큼 작아 코스트까지
    낮게 나온다. 실제로 지그재그 시나리오에서 콘 정면으로 직진하는 X=1.2
    후보가 선택됐다.

    꼬리는 종점 헤딩 방향의 직선이다. quintic 의 종점 조건이 y''(X)=0 이라
    곡률 0 으로 끝나므로, 그 뒤를 직선으로 잇는 것이 물리적으로도 맞다.
    """
    x, y, yaw, kappa, s = edge.sample(count)
    if eval_x <= edge.horizon:
        # 평가거리보다 긴 후보는 잘라서 짧은 후보와 같은 조건으로 비교한다.
        keep = x <= eval_x + 1.0e-9
        if not keep.all():
            x, y, yaw, kappa, s = (a[keep] for a in (x, y, yaw, kappa, s))
        return x, y, yaw, kappa, s

    heading = float(yaw[-1])
    cosine = math.cos(heading)
    if cosine < 1.0e-3:
        return x, y, yaw, kappa, s      # 거의 옆을 보고 있다 — 연장 무의미

    # x 기준으로 eval_x 까지 늘린다. 호길이로는 (dx / cos) 만큼이다.
    step = _tail_grid(edge.horizon, count, eval_x)
    tail_count = len(step)
    return (
        np.concatenate((x, x[-1] + step)),
        np.concatenate((y, y[-1] + step * math.tan(heading))),
        np.concatenate((yaw, np.full(tail_count, heading))),
        np.concatenate((kappa, np.zeros(tail_count))),
        np.concatenate((s, s[-1] + step / cosine)),
    )


def batch_edges(horizon, d_targets, psi_targets, kappa_start, count, eval_x):
    """
    같은 horizon 을 공유하는 후보들의 계수와 표본을 한 번에 계산한다.

    후보 378개를 하나씩 만들면 quintic 평가와 표본화만으로 40 ms 가 넘는다.
    같은 X 를 쓰는 후보는 x 격자와 Vandermonde 행렬이 동일하므로, 계수를
    (n, 6) 행렬로 모아 행렬곱 몇 번으로 끝낼 수 있다.

    (coeffs, x, y, yaw, kappa, s) 를 돌려준다. y 이후는 모두 (n, m) 배열이고
    x 는 모든 후보가 공유하는 (m,) 격자다.
    """
    d_targets = np.asarray(d_targets, dtype=np.float64)
    psi_targets = np.asarray(psi_targets, dtype=np.float64)
    grid_d, grid_psi = np.meshgrid(d_targets, psi_targets, indexing='ij')
    d_flat = grid_d.ravel()
    psi_flat = grid_psi.ravel()

    x_end = float(horizon)
    a2 = 0.5 * kappa_start
    residual = np.column_stack((
        d_flat - a2 * x_end * x_end,
        np.tan(psi_flat) - 2.0 * a2 * x_end,
        np.full(len(d_flat), -2.0 * a2),
    ))
    tail = residual @ _terminal_inverse(x_end).T        # (n, 3) = a3, a4, a5

    coeffs = np.zeros((len(d_flat), 6))
    coeffs[:, 2] = a2
    coeffs[:, 3:] = tail

    x = _sample_grid(x_end, count)
    powers = np.vstack([x ** k for k in range(6)])                    # (6, m)
    first = np.vstack([k * x ** (k - 1) if k else np.zeros_like(x)
                       for k in range(6)])
    second = np.vstack([k * (k - 1) * x ** (k - 2) if k >= 2
                        else np.zeros_like(x) for k in range(6)])

    y = coeffs @ powers
    slope = coeffs @ first
    curvature = (coeffs @ second) / np.power(1.0 + slope * slope, 1.5)
    yaw = np.arctan(slope)

    ds = np.sqrt(1.0 + slope * slope)
    step = np.diff(x)
    s = np.concatenate((
        np.zeros((len(d_flat), 1)),
        np.cumsum(0.5 * (ds[:, :-1] + ds[:, 1:]) * step, axis=1),
    ), axis=1)

    if eval_x > x_end:
        tail_step = _tail_grid(x_end, count, eval_x)                  # (t,)
        heading = yaw[:, -1:]                                         # (n, 1)
        cosine = np.cos(heading)
        x = np.concatenate((x, x[-1] + tail_step))
        y = np.concatenate((y, y[:, -1:] + tail_step * np.tan(heading)),
                           axis=1)
        yaw = np.concatenate(
            (yaw, np.repeat(heading, len(tail_step), axis=1)), axis=1)
        curvature = np.concatenate(
            (curvature, np.zeros((len(d_flat), len(tail_step)))), axis=1)
        s = np.concatenate((s, s[:, -1:] + tail_step / cosine), axis=1)
    elif eval_x < x_end:
        keep = x <= eval_x + 1.0e-9
        x = x[keep]
        y, yaw, curvature, s = (a[:, keep] for a in (y, yaw, curvature, s))

    return coeffs, d_flat, psi_flat, x, y, yaw, curvature, s


def curvature_energy(kappa, s):
    """
    곡률 에너지 = integral(kappa^2 ds). 같은 조건이면 덜 꺾는 경로를 고르게 한다.

    횡저크가 아니라 곡률 제곱을 쓰는 이유: 저크는 3차 미분이라 다항식 차수가
    낮은 구간에서 수치적으로 불안정하고, 조향 액추에이터가 실제로 힘들어하는
    것은 조향각 크기(≈ 곡률)이기 때문이다.
    """
    if len(s) < 2:
        return 0.0
    squared = kappa * kappa
    step = np.diff(s)
    return float(np.sum(0.5 * (squared[:-1] + squared[1:]) * step))
