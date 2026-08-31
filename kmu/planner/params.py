#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planner/params.py — 플래너가 쓰는 모든 상수의 단일 출처.

매직 넘버를 다른 파일에 박지 않는다. 여기에만 두고, 값마다 단위와 근거를 남긴다.
값이 바뀌면 여기 한 곳만 고치면 된다.

좌표 규약 (플래너 내부 전체에 적용)
  바디프레임: +x 전방, +y 좌측, yaw 는 +x 기준 CCW (ROS 표준, laser frame 과 동일)
  곡률 kappa > 0 이면 좌회전, 조향각 delta > 0 이면 좌회전
  ※ dwa.py 의 프레임(+y 전방, +x 우측)과 다르다. 섞지 말 것.
  ※ /erp_writes · /xycar_motor 의 조향 명령은 반대로 '음수가 좌회전'이다.
     변환은 steer_rad_to_cmd() 한 곳에서만 한다.
"""

import math


# ══════════════════════════════════════════════════════════════════
#  차량 제원 (2026-08-09 실측)
# ══════════════════════════════════════════════════════════════════

WHEEL_BASE_M = 0.35          # 앞뒤 축간 거리
VEHICLE_WIDTH_M = 0.30       # 차폭 (최대폭)
VEHICLE_HALF_WIDTH_M = VEHICLE_WIDTH_M / 2.0

# 앞바퀴 실제 조향각 한계. 명령값이 아니라 바퀴가 꺾이는 각도다.
MAX_STEER_RAD = math.radians(30.0)

# 자전거 모델 최대 곡률과 최소 회전반경.
#   kappa_max = tan(delta_max) / L = tan(30 deg) / 0.35 = 1.650 /m
#   R_min     = 1 / kappa_max                           = 0.606 m
KAPPA_MAX = math.tan(MAX_STEER_RAD) / WHEEL_BASE_M
R_MIN_M = 1.0 / KAPPA_MAX


# ══════════════════════════════════════════════════════════════════
#  조향 명령 변환
# ══════════════════════════════════════════════════════════════════

# /erp_writes(Twist.angular.z) · /xycar_motor([steer, speed]) 의 조향 명령 범위.
# lane_detection3.STEER_HW_LIMIT 과 같은 값이어야 한다.
STEER_CMD_LIMIT = 50.0

# 명령값 부호: steer_calib.py 측정 절차("좌한계는 음수 방향")와
# lane_detection3.compute_steering(lat_err>0 이면 우조향)이 일치한다.
#   → 명령 음수 = 좌회전. 플래너는 delta>0 이 좌회전이므로 부호를 뒤집는다.
STEER_CMD_SIGN = -1.0

# ⚠ 미검증 — Phase 4 에서 실측할 것.
#   명령 ±STEER_CMD_LIMIT 가 바퀴 ±MAX_STEER_RAD 에 대응한다고 가정한 선형 매핑이다.
#   근거는 lane_detection3.py:135 의 "STEER_HW_LIMIT = 50.0 (xycar 스펙 /
#   steer_calib.py 실측)" 주석 하나뿐이고, steer_calib 의 실측 결과
#   (trim b, lim_l, lim_r)가 코드에 반영된 흔적이 없다. 좌우 비대칭이나
#   직진 트림이 있으면 이 선형 가정이 깨진다.
#   Phase 4 결과가 나오면 여기를 좌/우 개별 스케일 + 트림으로 바꾼다.
STEER_CMD_PER_RAD = STEER_CMD_LIMIT / MAX_STEER_RAD


def steer_rad_to_cmd(delta_rad):
    """바퀴 조향각(rad, 좌회전 양수)을 모터 명령값으로 바꾼다."""
    cmd = STEER_CMD_SIGN * delta_rad * STEER_CMD_PER_RAD
    return max(-STEER_CMD_LIMIT, min(STEER_CMD_LIMIT, cmd))


# ══════════════════════════════════════════════════════════════════
#  코스 제원 (실측)
# ══════════════════════════════════════════════════════════════════

LANE_WIDTH_M = 0.40          # 차선 중앙 to 차선 중앙 (주행 가능 폭)

# 차선 안에서 차체가 쓸 수 있는 편측 여유.
#   (0.40 - 0.30) / 2 = 0.05 m 밖에 없다.
LANE_GEOMETRIC_MARGIN_M = (LANE_WIDTH_M - VEHICLE_WIDTH_M) / 2.0

# 대회 규정상 무감점으로 허용되는 차선 이탈량(5~10 cm). 보수적으로 하한을 쓴다.
# 이 여유가 없으면 LANE_KEEP 의 하드 제약이 편측 5 cm 가 되어, 인지·추종 오차가
# 그보다 커지는 순간 모든 후보가 탈락하고 Fallback 으로 영구 퇴각한다.
LANE_DEPART_TOLERANCE_M = 0.05


# ══════════════════════════════════════════════════════════════════
#  라바콘
# ══════════════════════════════════════════════════════════════════

# ★ 충돌 기하와 인지 기하는 서로 다른 값이다. 절대 섞지 말 것 ★
#   콘은 위로 갈수록 좁아진다. 라이다는 장착 높이의 단면(7~12 cm)을 보지만,
#   차가 실제로 부딪히는 것은 밑면(13.7 cm)이다.

CONE_BASE_M = 0.137          # 밑면 정사각형 한 변 (충돌 기하)

# 밑면 정사각형의 반폭. 콘이 측정 방향과 이루는 각 t 에 대해
# 반폭 = (a/2)(|cos t| + |sin t|) 이고, t=0 에서 0.0685, t=45deg 에서 0.0969 다.
CONE_HALF_EXTENT_M = CONE_BASE_M * math.sqrt(2.0) / 2.0

# 마주보는 콘의 '안쪽 면' 사이 최소 거리 (지면 실측). 콘 두께는 이미 빠져 있다.
CONE_GATE_INNER_M = 0.45

# 라이다 장착 높이에서 보이는 콘 단면 폭 (ridar.py 판별 게이트, 실측).
CONE_SCAN_WIDTH_MIN_M = 0.07
CONE_SCAN_WIDTH_MAX_M = 0.12

# ★ 스캔 표면과 충돌 표면의 어긋남 ★
#   플래너는 raw scan 점을 장애물로 쓴다 (클러스터링 지연을 피하려고 DWA 가
#   그렇게 하고 있고, lattice 도 같은 입력을 쓴다). 그런데 라이다가 보는 것은
#   장착 높이의 좁은 단면이고 차가 부딪히는 것은 넓은 밑면이다.
#   → 스캔 점은 실제 충돌 표면보다 이만큼 바깥에 찍힌다. 그만큼을 R_safe 에
#     더하지 않으면 "스캔상으로는 여유 있는데 실제로는 밑면을 치는" 일이 난다.
#   보수적으로 단면이 가장 좁을 때(0.07)를 쓴다.
CONE_SCAN_INSET_M = (CONE_BASE_M - CONE_SCAN_WIDTH_MIN_M) / 2.0


# ══════════════════════════════════════════════════════════════════
#  안전반경 R_safe
# ══════════════════════════════════════════════════════════════════

# ⚠ TRACKING_ERROR_M 은 Phase 4 에서 실측할 때까지 잠정값이다.
#   3-5절: lattice 는 계획한 경로를 추종기가 오차를 갖고 따라가므로,
#   이 항을 빼먹으면 "계획상 통과인데 실제로는 콘을 치는" 현상이 반복된다.
TRACKING_ERROR_M = 0.05      # ⚠ 잠정 — Phase 4 실측으로 교체
PERCEPTION_ERROR_M = 0.03    # ⚠ 잠정 — 클러스터 중심 추정 오차

# raw scan 점까지 확보해야 하는 거리.
#   R_safe = 차량 반폭 + (스캔면-충돌면 오프셋) + 추종오차 + 인지오차
# 콘 '중심'이 아니라 '스캔 점'까지의 거리 기준이다. 클러스터 중심을 쓰던
# 이전 모델(차량반폭 + 콘반폭)과 값이 다르니 섞지 말 것.
R_SAFE_CONE_M = (
    VEHICLE_HALF_WIDTH_M
    + CONE_SCAN_INSET_M
    + TRACKING_ERROR_M
    + PERCEPTION_ERROR_M
)

# 최소 회랑에서 남는 오차 예산.
#   충돌면(밑면 안쪽)이 중앙선에서 +-G/2 이고, 스캔 점은 거기서 inset 만큼
#   더 바깥이다. 통과 조건은
#     차량반폭 + inset + 오차 <= G/2 + inset   →   오차 <= G/2 - 차량반폭
#   inset 이 양변에서 상쇄되므로, 예산은 콘 크기·단면폭과 무관하게 결정된다.
GATE_ERROR_BUDGET_M = CONE_GATE_INNER_M / 2.0 - VEHICLE_HALF_WIDTH_M  # 0.075

# 현재 오차 가정이 예산 안에 들어오는지. 음수면 최소 게이트를 통과할 수 없다.
#   지금 값: 0.075 - (0.05 + 0.03) = -0.005 m  ← 5 mm 초과
# Phase 4 에서 추종오차를 실측해 이 값이 양수가 되는지 반드시 확인할 것.
GATE_ERROR_MARGIN_M = GATE_ERROR_BUDGET_M - (TRACKING_ERROR_M
                                             + PERCEPTION_ERROR_M)


def cone_r_safe(tracking_error_m=None, perception_error_m=None):
    """추종/인지 오차를 바꿔 가며 콘 안전반경을 다시 계산한다 (replay 탐색용)."""
    tracking = (TRACKING_ERROR_M if tracking_error_m is None
                else tracking_error_m)
    perception = (PERCEPTION_ERROR_M if perception_error_m is None
                  else perception_error_m)
    return (VEHICLE_HALF_WIDTH_M + CONE_SCAN_INSET_M
            + tracking + perception)


# 회랑 공칭 폭 — 한쪽 벽만 보일 때 반대쪽을 추정하는 데 쓴다.
# 스캔 점 기준 좌우 벽 사이 거리 = 안쪽 면 사이 + 2 * inset
CORRIDOR_SCAN_WIDTH_M = CONE_GATE_INNER_M + 2.0 * CONE_SCAN_INSET_M


# ══════════════════════════════════════════════════════════════════
#  카메라 BEV 스케일
# ══════════════════════════════════════════════════════════════════

# lane_detection3.LANE_WIDTH_PX 와 같은 값이어야 한다.
# preprocessing.py 의 BEV 파라미터를 바꾸면 lane_width_calib.py 재실행이 선행돼야 한다.
LANE_WIDTH_PX = 350.0
BEV_M_PER_PX = LANE_WIDTH_M / LANE_WIDTH_PX     # 0.0011429 m/px (1.143 mm)


def bev_px_to_m(px):
    """BEV 픽셀 거리를 미터로 바꾼다 (가로 방향)."""
    return px * BEV_M_PER_PX


# BEV 세로(종방향) 스케일 [m/px].
#
# ⚠ 미측정. lane_width_calib.py 가 재는 것은 가로 스케일뿐이고, BEV 워프는
#   원근 변환이라 가로/세로 스케일이 같다는 보장이 전혀 없다.
#   (ROI = 원본 480행 중 336~446행 → BEV 640 x 110 px)
#
# 측정 방법
#   1) 차량을 정지시키고 카메라 ROI 안 지면에 종방향으로 거리를 아는 표식 두 개를
#      놓는다 (예: 0.30 m 간격의 테이프 두 줄).
#   2) preprocessing.lane_masks() 의 bev 출력에서 두 표식의 행(y) 좌표를 읽는다.
#   3) BEV_M_PER_PX_LONG = 실제거리[m] / 행 차이[px]
#
# None 이면 psi 와 kappa_lane 은 NaN 으로 발행된다. d 는 가로 스케일만 쓰므로
# 측정 전에도 유효하다 — Phase 3 의 부호·크기 검증은 d 로 먼저 할 수 있다.
BEV_M_PER_PX_LONG = None


def lane_measurement_si(lat_err_px, heading_px, heading_span_px,
                        curvature_px):
    """
    BEV 픽셀 측정값을 바디프레임 SI 단위로 바꾼다.

    ★ 부호 변환은 여기 한 곳에서만 한다 ★
    BEV 이미지 열(column)은 오른쪽이 +인데 바디프레임 y 는 왼쪽이 +다.
    두 규약이 섞이면 차가 반대로 꺾인다.

    입력 (전부 lane_detection3 의 BEV 픽셀)
      lat_err_px      : 차로중심 - 차량축. + 면 차로중심이 차량 오른쪽
      heading_px      : 먼 지점 - 가까운 지점의 차로중심 x 차이
      heading_span_px : 위 두 지점의 세로(종방향) 간격
      curvature_px    : x = A y^2 + B y + C 의 A 계수

    출력 (바디프레임, +y 좌측)
      d          [m]   차로중심의 횡변위. + 면 차로중심이 차량 왼쪽
      psi        [rad] 차로중심선의 헤딩. + 면 좌회전 방향
      kappa_lane [1/m] 차로중심선의 곡률. + 면 좌회전 방향

    세로 스케일이 없으면 psi 와 kappa_lane 은 NaN 이다.
    """
    if lat_err_px is None:
        return float('nan'), float('nan'), float('nan')

    lateral = -float(lat_err_px) * BEV_M_PER_PX

    if BEV_M_PER_PX_LONG is None or not heading_span_px:
        return lateral, float('nan'), float('nan')

    forward = float(heading_span_px) * BEV_M_PER_PX_LONG
    heading = math.atan2(-float(heading_px) * BEV_M_PER_PX, forward)

    # x_px = A y_px^2 + ... 를 바디프레임으로 옮기면
    #   d^2(y_body)/d(x_body)^2 = -2 A * (가로스케일) / (세로스케일)^2
    # 기울기가 작을 때 이 값이 곧 곡률이다.
    kappa = (-2.0 * float(curvature_px) * BEV_M_PER_PX
             / (BEV_M_PER_PX_LONG * BEV_M_PER_PX_LONG))
    return lateral, heading, kappa


# ══════════════════════════════════════════════════════════════════
#  lattice 후보 생성
# ══════════════════════════════════════════════════════════════════

# 종방향 horizon 레이어.
#   quintic 의 최대 곡률은 대략 5.774*d/X^2 이므로 X >= sqrt(5.774*d/kappa_max).
#   kappa_max=1.650 에서 d=0.4 이면 X>=1.18, d=0.5 이면 X>=1.32 다.
#   1.2 는 미세 보정용(짧아서 큰 d 는 곡률 제약에 걸려 자동 탈락),
#   2.0 이 주력, 3.0 은 완만한 선제 기동용.
#   레이어를 6개로 늘린 이유: 게이트가 0.9 m 간격으로 오는데 레이어가
#   1.2/2.0/3.0 뿐이면 종점을 게이트 근처에 놓을 수 없어, 첫 게이트를
#   비껴가는 후보만 남는다. 실측으로 여유가 45 -> 63 mm 로 늘었다.
HORIZON_LAYERS_M = (0.8, 1.2, 1.6, 2.0, 2.6, 3.0)

# 모든 후보를 '같은' 전방거리까지 평가한다.
#   짧게 잡으면: 짧은 레이어가 자기 horizon 밖 장애물을 못 봐서 통과 판정된다.
#     실제로 지그재그에서 콘 정면으로 직진하는 X=1.2 후보가 선택됐다.
#   길게 잡으면: 10 Hz 로 재계획하는데도 3 m 앞 게이트까지 한 번에 통과하는
#     경로를 요구하게 되어 과제약이 된다. 5차 다항식 하나로는 게이트가
#     좌우로 번갈아 밀리는 슬라럼을 한 번에 꿸 수 없어 전부 탈락한다.
#   → 재계획 주기(10 Hz) 동안 실제로 주행할 거리 + 여유로 잡는다.
#     1 m/s 에서 한 사이클 0.1 m 이므로 2.0 m 면 20 사이클 분의 여유가 있다.
EVAL_HORIZON_M = 2.0

# 횡방향 종점. OVERTAKE 의 상한 +0.50 m 를 덮어야 한다.
# 도로 전체 폭이 0.80 m 뿐이라 3-3절 자리표시자 ±1.2 는 물리적으로 무의미했다.
#   13개(간격 83 mm)로는 이산화 손실이 30 mm 나 됐다. 콘 회랑의 오차 예산이
#   75 mm 뿐이라 그 손실이 통과/실패를 갈랐다. 21개면 간격 50 mm, 손실 12 mm.
LATERAL_TARGETS_M = 21       # 후보 개수
LATERAL_SPAN_M = 0.50        # linspace(-0.50, +0.50, 21) → 간격 50 mm

# 종점 헤딩 후보(rad). 회랑을 비스듬히 통과해야 할 때 필요하다.
TERMINAL_HEADINGS_RAD = (-0.35, 0.0, 0.35)


# ══════════════════════════════════════════════════════════════════
#  코스트 가중치
# ══════════════════════════════════════════════════════════════════

# 곡률 에너지 — 모든 모드 공통. 같은 조건이면 덜 꺾는 쪽을 고른다.
W_CURV = 0.5

# 후보 평가 시 경로를 몇 점으로 이산화할지. 제약 검사와 코스트에 함께 쓴다.
# 3.0 m horizon 을 25점으로 나누면 간격 0.12 m 로, 콘 반폭(0.097 m)보다 촘촘해
# 점 사이로 장애물이 빠져나가지 않는다.
CANDIDATE_SAMPLES = 25


# ══════════════════════════════════════════════════════════════════
#  추종기 (3-6절)
# ══════════════════════════════════════════════════════════════════

# 곡률 피드포워드를 읽을 전방 거리(m). 조향 응답 지연을 앞당겨 보상한다.
LD_FF_M = 0.30

K_PSI = 1.0                  # 헤딩 오차 피드백 (rad 당 rad)
K_E = 1.2                    # 횡오차 피드백 (Stanley 형)
V_MIN_MPS = 0.5              # 저속에서 K_E/v 가 발산하지 않게 하는 하한
