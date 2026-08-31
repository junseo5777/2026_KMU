#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planner/tracker.py — 곡률 피드포워드 + 횡오차 피드백 추종기 (3-6절).

Pure Pursuit 단독은 코너를 안쪽으로 자른다. 일반 주행에서는 그게 오히려
자연스럽지만 콘 회피에서는 그 안쪽에 콘이 있다. lattice 가 곡률 프로파일을
이미 주므로 그것을 피드포워드로 쓰고, 피드백은 오차 보정만 담당하게 한다.

    delta = atan(L * kappa_ff) + K_PSI * e_psi + atan(K_E * e_lat / max(v, V_MIN))
            └── 경로가 요구하는 조향 ──┘   └─ 헤딩 오차 ─┘  └── 횡오차 (Stanley 형) ──┘

부호 규약 (params.py 와 동일)
    바디프레임 +x 전방 / +y 좌측, delta > 0 이면 좌회전.
    모터 명령으로 바꾸는 것은 params.steer_rad_to_cmd() 한 곳에서만 한다.
"""

import math

import numpy as np

from . import params as P


def frenet_error(pose, edge):
    """
    경로에 대한 (횡오차, 헤딩오차, 투영 호길이) 를 구한다.

    ``pose`` 는 경로가 정의된 프레임에서의 (x, y, yaw) 다. 계획 직후 같은
    사이클에 쓰면 pose 는 원점이라 오차가 0 이지만, 계획을 여러 사이클에 걸쳐
    유지할 때(3-5절 (a) 상태 연속성)는 그동안 움직인 만큼 오차가 생긴다.

    횡오차 부호: 차량이 경로의 **좌측**에 있으면 양수. 그러면 우조향으로
    되돌려야 하므로 피드백 항의 부호는 음수여야 한다 — 아래 pure_feedback
    참조.
    """
    x, y, yaw = pose
    sample_x, sample_y, sample_yaw, _, sample_s = edge.sample(
        P.CANDIDATE_SAMPLES
    )
    index = int(np.argmin(np.hypot(sample_x - x, sample_y - y)))

    # 경로 접선에 수직인 방향으로의 부호 있는 거리
    tangent = np.array([math.cos(sample_yaw[index]),
                        math.sin(sample_yaw[index])])
    normal = np.array([-tangent[1], tangent[0]])   # 접선을 +90도 회전 = 좌측
    offset = np.array([x - sample_x[index], y - sample_y[index]])
    lateral = float(offset @ normal)

    heading = math.atan2(math.sin(yaw - sample_yaw[index]),
                         math.cos(yaw - sample_yaw[index]))
    return lateral, heading, float(sample_s[index])


def curvature_at(edge, arclength):
    """호길이 기준으로 곡률을 읽는다. 경로 끝을 넘으면 끝값을 준다."""
    _, _, _, kappa, s = edge.sample(P.CANDIDATE_SAMPLES)
    if arclength <= s[0]:
        return float(kappa[0])
    if arclength >= s[-1]:
        return float(kappa[-1])
    return float(np.interp(arclength, s, kappa))


def steer_command(edge, pose=(0.0, 0.0, 0.0), v_mps=0.0):
    """
    경로를 따라가는 조향각(rad, 좌회전 양수)을 낸다.

    ``v_mps`` 는 반드시 **m/s** 다. xycar 명령 단위 속도(9~20)를 그대로 넣으면
    Stanley 항 K_E*e_lat/v 가 20 배 작아져 횡오차 보정이 사실상 사라진다.
    명령 단위 → m/s 환산은 Phase 4 에서 실측한다.
    """
    lateral, heading, arclength = frenet_error(pose, edge)
    kappa_ff = curvature_at(edge, arclength + P.LD_FF_M)

    feedforward = math.atan(P.WHEEL_BASE_M * kappa_ff)
    # 차량이 경로 좌측(lateral>0)이면 우조향(delta<0)으로 되돌려야 한다.
    feedback_lateral = -math.atan(
        P.K_E * lateral / max(abs(v_mps), P.V_MIN_MPS)
    )
    feedback_heading = -P.K_PSI * heading

    delta = feedforward + feedback_heading + feedback_lateral
    return float(np.clip(delta, -P.MAX_STEER_RAD, P.MAX_STEER_RAD))


def steer_cmd_from_path(edge, pose=(0.0, 0.0, 0.0), v_mps=0.0):
    """조향각을 구해 모터 명령 단위까지 한 번에 바꿔준다."""
    return P.steer_rad_to_cmd(steer_command(edge, pose, v_mps))
