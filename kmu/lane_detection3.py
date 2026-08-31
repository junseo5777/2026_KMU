"""
lane_detection.py — 슬라이딩 윈도우 기반 (실제 트랙용)

설계:
  - 히스토그램 피크 → 좌/우 차선 시작점 → 슬라이딩 윈도우로 픽셀 추적 → polyfit
  - 얇은 흰 점선/노란 중앙선·반사 환경에 강건 (윈도우가 점선도 따라감)
  - 노란 중앙선으로 현재 차로(1/2) 판별 (시뮬 자산 재사용)
  - 시뮬 검증 로직 재사용: 모드 판별, 동적 게인, heading 보정,
    동적 lookahead, 모드별 고정 속도+슬루 제한, 변화율 제한,
    lane_detected 히스테리시스,
    이전 프레임 fit 재사용(margin search)·회복 로직

입력: white_mask, yellow_mask (BEV 이진화), bev_shape
출력: dict(steering_angle, target_speed, driving_mode, lane_detected, ...)

- 루프: 슬라이딩 윈도우로 시작점 검출 -> margin기반(으로 3차 다항식으로 정규화된 fit탐색(fit[0]-> 곡률. fit[2]->가상 차선 평행이동 기준), margin 신회 
    -> margin 방식이 신뢰를 잃는 트리거(LD_LOST_FRAMES 검출 실패 프레임, low_conf_count 저신회 상태 연속 프레임) 달성 시 none으로 리셋하여 다시 슬라이딩 윈도우로 검출


-검토: 한 차선 소실시 사용되는 동적 lane_width lpf적용하나, 누적 오염 시 문제 발생하므로 차선 중앙에 두고 실측한 lane_width로 고정하여 사용해주면 좋을 듯
다만 실측값은 bev파라미터 바꾸면 다시 측정해야함
1. 고정 lane_width만 사용하는 방법(차선 중앙to차선 중앙, measured 값 수십프레임 평균치 사용)
2. 동적 lane_width유지하되, 정적 lane_width 기준으로 값 제한두고, straight 모드일 떄만 측정하는 등 두 값 조합하는 방법
"""

import numpy as np
from collections import deque


# ════════════════════════════════════════════════
#  설정 파라미터
# ════════════════════════════════════════════════
# ── 슬라이딩 윈도우 ──
N_WINDOWS      = 10      # 세로 분할 윈도우 수
WIN_MARGIN     = 44      # 윈도우 가로 반폭 (px)
WIN_MINPIX     = 30      # 윈도우 재중심 최소 픽셀
MIN_LANE_PIX   = 148     # 차선 인정 최소 총 픽셀
HIST_RATIO     = 0.5     # 히스토그램 계산에 쓸 하단 비율
MARGIN_SEARCH  = 59      # 이전 fit 주변 재탐색 폭 (윈도우 생략) (60→80: 곡선에서 밴드 이탈 방지)

# ── 윈도우 재획득 base: W×Y peak pairing ──
PEAK_MIN_MASS = 255.0 * 8    # peak 인정 최소 질량 (마킹 세로 8행 상당).
                             # 15행이면 기울어진 마킹(재획득 시 헤딩오차)이 컬럼에
                             # 분산되며 임계 미달 → pairing 이 단일 강등되는 것을 확인.
                             # 잡음 블롭 방어는 W×Y 룰 + 간격 허용창이 담당하므로
                             # 여기는 스펙클만 거르면 충분.
PEAK_NMS_PX   = 37           # peak 비최대 억제 반경 (같은 마킹의 중복 peak 방지)
PAIR_SEP_LO   = 0.85         # 쌍 간격 허용창 하한 x lane_width (노이즈 여유)
PAIR_SEP_HI   = 1.5          # 상한: 곡선/헤딩 기울기가 수평 간격을 sqrt(1+slope^2)배로
                             # 부풀림 (헤딩 0.5~1.0 에서 1.15~1.41배 시뮬 확인) →
                             # 대칭 ±15% 창은 S자 회복 순간에 진짜 쌍을 거부했음

# ── 폴리핏 ──
POLY_ORDER     = 2

# ── 차선 폭 (BEV px, 고정) ──

LANE_WIDTH_PX  = 260


GEOM_WIDTH_LO  = 0.75    # lane_width 대비 하한
GEOM_WIDTH_HI  = 1.25    # 상한
# 평행성: 진짜 평행한 두 선은 모든 y 에서 수직 폭이 같다. 상단만 오염된 fit 은
# 근거리/원거리 폭이 어긋나므로 이 비율로 잡힌다 (h-1 단독 검사의 사각지대 보완).
GEOM_PARALLEL_TOL = 1.35

# ── 좌우 쌍 붕괴 판정 ──
# 한쪽 margin_search 가 반대쪽 차선으로 드리프트하면 둘 다 같은 차선을 잡아
# real=(True,True) 인데 폭이 0 에 가까워진다. 이 비율×lane_width 미만이면 쌍을 폐기.
# 0.5 로 여유를 크게 둔 이유: 정상 주행 중에는 절대 도달할 수 없는 값이어야 하고,
# 붕괴 시에는 폭이 0 근처로 가므로 중간 어디를 잡아도 판별된다 → 오탐 쪽으로 안전하게.
LANE_MIN_SEP_RATIO = 0.5

# ── 쌍 간격 '상한' — 벌어진 쌍은 한쪽만 쓴다 ──
# LANE_MIN_SEP_RATIO 는 하한(두 윈도우가 같은 선을 탄 쌍 붕괴)만 본다. 반대쪽,
# 즉 '너무 벌어진 쌍' 에는 아무 게이트가 없어서 그대로 중점이 계산됐다.
#   실측 — S자에서 추정 차로중심이 물리적으로 불가능한(|lat_err| > 차로 반폭)
#   576프레임의 구성:
#       단일선 슬롯이 선의 실제 방향과 반대     55.6%   ← F 가 담당
#       양선인데 sep 이상 (중앙 387px, 정상 260) 26.0%   ← 여기
#       단일선, 슬롯은 맞는데 선이 멀리          11.3%
#       양선 정상                                7.1%
#   sep 387px 는 차선폭의 1.49배다. 둘 중 하나는 차선이 아니거나 옆 차로의 선이고,
#   그때 중점은 실제 차로중심에서 최대 (387-260)/2 = 64px 벗어난다.
#
# 임계 1.35: GEOM_WIDTH_HI(1.25) 는 신뢰도만 깎고 fit 은 그대로 쓰므로 그보다
#   위에 둔다. 기울어진 쌍의 수평 간격은 perp x sqrt(1+slope^2) 로 부푸는데,
#   그건 _perp_widths 가 이미 되돌려 재는 값이라 여기서 다시 볼 필요가 없다.
#   다만 이 게이트는 수평 sep 을 그대로 보므로 여유를 둔 것이다.
#
# 왜 '양쪽 폐기' 가 아니라 '한쪽만 남기기' 인가 — 둘 다 재봤다:
#       양쪽 폐기 : S자 지속 11.1% → 8.9%,  단 blind 4.1% → 7.5% (거의 2배)
#       한쪽만    : S자 지속 11.1% → 8.8%,  blind 4.2% (사실상 무변화)   ← 채택
#   폐기는 조향 근거를 통째로 없애 감쇠 경로로 보내지만, 한쪽만 남기면 단일선
#   경로(가상 차선 + 오프셋 추종)가 그대로 이어받는다.
#
# 남길 쪽 선정: 노랑 우선(LAB b* + HSV 3중 게이트라 흰색보다 오검출이 적다),
#   같은 색이면 픽셀이 많은 쪽. F 와 함께 쓰면 S자 지속이 11.1% → 4.5% 가 된다
#   (F 단독 6.9%, G 단독 6.5%).
LANE_MAX_SEP_RATIO = 1.35

WW_RESET_FRAMES = 8

# 비물리적 곡률 거부: BEV 높이 전체에서 차선폭의 몇 배까지 휘어도 되는가.
# |A|*h^2 <= lane_width*MAX_SWEEP_RATIO. h=110, lane_width=350 이면 |A| <= 0.029 로,
# 정상 급커브(실측 A≈0.011)의 2.6배 여유. 이미지2 의 오염 fit(A≈0.07)은 차단된다.
# 기하에서 자동으로 유도되므로 BEV/차선폭을 바꿔도 따라온다.
MAX_SWEEP_RATIO = 1.0

# 화면 밖 외삽 거부 여유 (x lane_width). _reject_bad_fit 3) 참조.
OUT_OF_FRAME_MARGIN = 0.5

# 기하 무효 시 confidence 페널티. LOW_CONF_THRESH(0.3) 보다 확실히 낮아야
# 회복 리셋이 발동한다. 기존 0.3 은 1.0*0.3=0.30 이 임계와 정확히 같아
# `conf < LOW_CONF_THRESH` 가 항상 False → 회복 트리거가 영구 미발동하는 사각지대였음.
GEOMETRY_INVALID_PENALTY = 0.2

# ── 모드 판별 곡률 임계 ──
STRAIGHT_THRESH = 0.00224    # (0.0015→0.002) straight 유지력↑ (노이즈 곡률로 entry 승격 억제)
CURVE_THRESH    = 0.00404    # (0.0025→0.004) curve 조기 승격 억제

# ── bend_ahead 기반 모드 승격 (곡률이 못 잡는 '차선에 대한 큰 틀어짐') ──
# 임계 선정 근거는 _classify_mode 주석의 분포표 참조. 80 은 S자 위험프레임의
# 64.9% 를 승격시키면서 정상 직선 오분류는 2.2% 에 그치는 지점이다.
# 160 은 S자 straight 프레임의 32.6% / 그 외 0.4% 로 더 확실한 급선회 신호.
BEND_ENTRY_PX = 80.0
BEND_CURVE_PX = 160.0

# ── 속도 판정 창 (2-4) ──
# 속도는 '이번 프레임 모드'가 아니라 '최근 창에서 가장 굽었던 모드'로 정한다.
#   게인/상한은 즉시 반응해야 하지만(굽으면 바로 꺾어야 함), 속도는 한 프레임의
#   오판으로 올라가면 안 된다 — 올라간 속도는 되돌리는 데 SPEED_DECEL 만큼
#   시간이 걸리고, 그 사이가 실차에서 이탈이 나던 구간이다.
#   실측(영상 4편, 그 외 구간의 'conf<=0.3 인데 속도>=13' 비율):
#       창 없음 4.11%  →  win4 3.45%  →  win8 2.87%  →  win15 1.94%
#     검출 지표(양선/무검출/단일선 과대오차)는 전 변형에서 완전히 불변 —
#     검출을 안 건드리고 속도만 보수화하는 순수한 변경이다.
#   대가는 평균속도: win8 -3.2% / win15 -5.3%. 효과의 70% 를 손실 60% 로 얻는
#   win8 을 택한다. 속도를 더 양보할 수 있으면 15 로 올리면 된다.
SPEED_MODE_WINDOW = 8

MODE_GAINS = {
    'straight': dict(kp=0.063, k_head=0.167, max_steer=9.0,  speed=40.0),
    'entry':    dict(kp=0.092, k_head=0.470, max_steer=30.0, speed=18.0),
    'curve':    dict(kp=0.289, k_head=0.293, max_steer=50.0, speed=11.0),
}

# MODE_GAINS = {
#     'straight': dict(kp=0.070, k_head=0.185, max_steer=9.0,  speed=20.0),
#     'entry':    dict(kp=0.117, k_head=0.522, max_steer=30.0, speed=12.0),
#     'curve':    dict(kp=0.397, k_head=0.325, max_steer=50.0, speed=9.0),
# }


STEER_HW_LIMIT = 50.0
MODE_RELAX_FRAMES = 4

# ── 조향 게인 ──
KD_LATERAL       = 0.032    # (0.15→0.02) D항 축소: 픽셀 노이즈 미분 증폭 방지
KD_LPF_ALPHA     = 0.3     # d_error 저역통과 계수 (노이즈 미분 억제)
KD_CLAMP         = 300.0   # d_error 클램프 (px/s)
# KFF_CURVE 3000→400→300: 커브 조기 선회 완화 시도. 다만 이것만으로는 안 들었다 —
#   중앙정렬 시 FF 항은 전체의 24% 뿐이고 지배항은 heading 이었다(MODE_GAINS 주석 참조).
#   heading 과 일부 중복된다(중앙정렬 포물선에서 heading ≈ A*k). heading 이 잡지
#   못하는 순수 곡률 성분만 남기는 역할이라 유지하되, k_head 분리가 실차 검증되면
#   제거 후보 1순위.
KFF_CURVE        = 334.0 #334.0
# ── 단일선 프레임의 heading 항 감쇠 ──
#
# ★ 1.0 = 무효. 0.5 를 실차 S자에서 시도했다가 되돌린 자리다. 다시 내리지 말 것.
#
# 왜 0.5 를 시도했었나 (그리고 왜 틀렸나):
#   근거는 "단일선 heading 은 실선 ± lane_width/2 로 세운 **가상** 중심선을 외삽한
#   것이라 부정확하다" 였다. 코드를 보면 성립하지 않는다 —
#       center_far  = polyval(fit, y_eval) + sign*offset
#       center_near = polyval(fit, h-1)    + sign*offset
#       heading     = center_far - center_near
#   두 항에 같은 sign*offset 이 들어가 **정확히 상쇄**된다. offset 을 0 으로 두든
#   999 로 두든 heading 값은 동일하다. 즉 단일선 heading 은 가상선과 무관하고,
#   실측한 그 선 자체의 기울기다. 양선 분기와 같은 y 구간, 같은 실측 fit 이다.
#
#   두 번째 오독: "단일선 heading 평균이 -35.6 (양선 +1.2) 이라 편향" — 이건 편향이
#   아니라 선택 효과다. 한쪽만 보이는 프레임은 차가 크게 틀어졌을 때 생기므로,
#   그때 heading 이 큰 것은 오류가 아니라 신호다.
#
# 실차 결과 (되돌린 이유):
#   영상 재생 지표는 좋아 보였다 — 포화 S자 1차선 21.3→12.7%, 2차선 13.3→6.8%,
#   조향-횡오차 상관 0.49→0.56. 그러나 실차 S자에서는 이탈이 여러 번 났다
#   (기존 코드는 흔들려도 10회 중 2회 수준).
#   개루프 재생은 궤적이 녹화본으로 고정돼 있어 '조향이 약해진 것'과 '불필요한
#   조향이 걷힌 것'을 구분하지 못한다. 포화 감소·|steer| 중앙값 감소(31.4→29.7)는
#   후자로 읽었지만 실제로는 전자였다. heading 은 이 제어기의 유일한 선행(lead)
#   항이라, 절반으로 깎으면 가장 크게 틀어진 순간에 위상 지연이 생겨 오버슈트 →
#   진동 → 이탈로 간다. 개루프 지표는 원리상 위상 지연에 벌점을 못 준다.
SINGLE_HEAD_SCALE = 1.0
# KFF_HEADING_BOTH / KFF_HEADING_SINGLE 은 MODE_GAINS['k_head'] 로 흡수했다.
#   - BOTH: 모드마다 필요한 선제 응량이 다른데 전역 상수 하나로 묶여 있었다.
#   - SINGLE(0.8): 양선(0.5) 과 다른 값을 쓸 근거가 검증된 적이 없고, 단일선은
#     가상 차선을 세운 추정치라 오히려 선제 반응을 키울 이유가 없다. 같은 k_head 사용.

# ── 동적 lookahead ──
LOOKAHEAD_MIN = 0.30       # (0.35→0.30) 선제 조향 완화
LOOKAHEAD_MAX = 0.50 #0.40      # (0.60→0.45) 코너 진입 시 너무 먼 곳을 보는 문제 완화
# lookahead 보간 기준 속도 = 실제로 쓰는 모드 속도 범위 (MODE_GAINS 에서 자동 유도).
# 새 파라미터가 아니라 파생값 — 모드 속도를 올리면 매핑이 따라온다.
SPEED_REF_LO = min(g['speed'] for g in MODE_GAINS.values())
SPEED_REF_HI = max(g['speed'] for g in MODE_GAINS.values())
BRAKE_LOOKAHEAD_RATIO = 0.6  # (0.85→0.6) BEV 최상단(외삽 부정확) 대신 중간 지점으로 bend 추정

# ── 속도: 모드별 고정 목표 + 슬루 제한 ──
# ★ MAX_SPEED 는 '안전 클립'이지 목표가 아니다 ★
#   기존 12.0 은 MODE_GAINS['straight']['speed'] 와 같은 값이라, 모드 속도를
#   15 로 올려도 여기서 12 로 잘려 아무 변화가 없었다(숨은 상한). 모드 속도를
#   단계적으로 올릴 수 있도록 여유를 둔다. 실제 목표는 MODE_GAINS 에서만 정한다.
MAX_SPEED   = 45.0
MIN_SPEED   = 11.0    # 하한 겸 시동 시 prev_speed 초기값
#   → 보호 역할은 DECEL 이 전부 하고, ACCEL 슬루는 복귀만 지연시킨다.
SPEED_ACCEL = 30.0   # 가속 슬루 (speed 단위/s). 사실상 제한 해제 수준이되
                     # 개루프 구동에서 계단 지령이 휠스핀을 내지 않도록 유한값 유지.
                     # 실차 가속 한계를 재면 그 값으로 맞추는 것이 정확하다.
SPEED_DECEL = 60.0    # 감속 슬루 = 모드 글리치 필터. 완전히 풀면 1프레임 오염이
                     # 그대로 속도 계단이 된다(위 표 inf/inf 행).

# ── 조향 변화율 제한 (단위/초 — _limit_rate 에서 dt 를 곱함) ──
# 기존 18/55 는 dt 없이 '프레임당' 적용돼 30fps=540/s, 15fps=270/s 로 프레임레이트에
# 따라 거동이 2배 달라졌음 (dt 를 쓰는 _compute_speed 와 불일치). 초 단위로 정규화하고
# 300/700 으로 조임 (30fps 환산 10/23 per frame, 기존 18/55 의 약 절반).
MAX_STEER_RATE_BASE  = 300.0
MAX_STEER_RATE_CURVE = 500.0

# ── 안정화 ──
SMOOTH_WINDOW     = 6
SMOOTH_N_STRAIGHT = 8
# 4 → 7: curve 만 유독 짧았다(straight/entry 는 8). 짧은 창은 '빠른 응답'이 아니라
#   '오염 프레임이 지령에 그대로 실림' 이었다 — S자에서 lat_err 오추정이 평균
#   7.7프레임(최대 69) 지속하는데, 4프레임 창은 그 램프를 그대로 통과시킨다.
#   실측(영상 4편, H+F+G 위에서 4 → 7):
#       S자   포화 14.5% → 10.7%   급등 1.08% → 0.78%   Δp99 19.45 → 13.84
#       그 외 포화  4.1% →  3.5%   급등 1.31% → 0.91%   Δp99 15.46 → 10.44
#   대가는 위상 지연 약 1.5프레임(0.05초)과 상관 0.46 → 0.42. 응답 곡선은
#   |lat_err|>80px 구간에서 40.4 → 38.9 로, 수정 전 기준선(37.4)보다는 여전히 높다.
#   ※ 개루프 재생은 지연에 벌점을 못 준다 — 이 값만은 실차에서 재확인할 것.
SMOOTH_N_CURVE    = 3
JUMP_REJECT_RATIO = 0.4  # 이전 프레임 대비 x값 변화폭(w*JUMP_REJECT_RATIO) 제한(넘으면 새 프레임 신뢰x, 기존 프레임 유지)
LOST_RESET_FRAMES = 5

# ── 직선 데드존 ──
# 3.0→1.0: max_steer=5.0 시절 데드존 3.0 은 가용범위의 60% 를 차지해 출력이
# '0 아니면 3~5' 인 뱅뱅 제어가 되고, 경계에서 on/off 채터링이 그대로 조향에 나갔음.
# 적용 방식도 계단형 컷 → soft deadzone(|steer|-dz) 으로 바꿔 불연속 제거.
STRAIGHT_DEADZONE_STEER = 1.0

# ── 회복 ──
SUSPECT_RESET_FRAMES = 4
LOW_CONF_THRESH      = 0.3
VIRTUAL_HOLD_FRAMES  = 9   # 한쪽이 real 검출 없이 이 프레임 수를 넘기면 prev 를 버리고
                           # 슬라이딩 윈도우로 강제 복귀 (가상 fit 자기 오염 루프 차단)

# ── W/Y 역할 기반 슬롯 교정 ──
# 트랙 위상상 노란 중앙선은 1차로에서 우경계, 2차로에서 좌경계다. 단일선 추적 중
# 슬롯이 그와 어긋나면 _fill_missing 이 가상 차선을 반대편에 그려 추정 차로중심이
# 정확히 lane_width 만큼 어긋난다(세션 초기에 실측한 -350.5px 오차가 이것).
#   실측 근거 — 노랑 단독 검출 1365프레임에서 현재 슬롯 vs 역할 기대 슬롯:
#       일치   938f (68.7%)  →  단일선 과대오차(>0.6폭) 발생률 12.8%
#       불일치 427f (31.3%)  →                            84.1%   ← 6.6배
#   즉 불일치는 거의 그대로 차로중심 오류다.
#
# 연속 프레임을 요구하는 이유: 교환은 좌/우가 한 프레임에 뒤바뀌는 불연속이라
# 차로중심이 lane_width 만큼 점프한다. 즉시 교환하면 S자 조향 급등이
# 0.95% → 1.25% 로 악화됐다. 연속 3프레임을 요구하면 1.04% 로 완화되면서
# 이득(그 외 단일선 과대오차 3.3% → 1.4%)은 그대로 남는다.
#   연속 1 / 3 / 5 실측 (S자 급등, S자 단일선과대):
#       1 → 1.25%, 6.7%   |   3 → 1.04%, 6.5%   |   5 → 0.98%, 7.1%
ROLE_SWAP_FRAMES = 3

LANE_OBS_DEADBAND  = 0.12   # x lane_width (=31px). 이 안쪽은 판정 보류
LANE_LATCH_FRAMES  = 3      # current_lane 이 None 일 때의 최초 확정
LANE_SWITCH_FRAMES = 3      # 뒤집기 — 상향은 위 근거로 기각, 최초 확정과 동일하게 둠

# ── current_lane 판정 밴드 (BEV 최하단 이 행 수만 사용) ──
# ★ 판정 밴드와 히스토그램 밴드를 분리한다 ★
#   기존엔 노란선 위치를 HIST_RATIO(하단 50% = 55행) 열 히스토그램으로 잡았다.
#   그 밴드는 재획득용 base 를 뽑기엔 맞지만 '노란선이 차량의 어느 쪽인가' 를
#   묻기엔 세로로 너무 길다.
#
#   원인(실차에서 관측된 그대로): S자에서 조향을 주면 차량이 차선에 대해 틀어지고,
#   BEV 에서 차선이 기운다. 그러면 같은 한 줄이 근거리에서는 좌측, 원거리에서는
#   우측에 놓인다 — lane_9 f3000 에서 같은 노란선이 근거리 x≈220 / 원거리 x≈450.
#   55행 밴드는 이 둘을 한 히스토그램에 섞어 넣고, peak 가 어느 쪽에 서느냐가
#   기울기에 좌우된다. 2차로 주행 중 1차로로 판정되는 것이 이 메커니즘이다.
#   가까운 쪽일수록 헤딩 오차의 영향이 작으므로 최하단만 본다.
#
#   실측 — 정답은 lane_9 S자 라벨 3구간(f992-1420=1차로, f2520-3050=2차로,
#   f3975-4100=2차로, 1083프레임). '차로전환' 은 확정값이 뒤집힌 횟수(정답 0회):
#       판정 밴드          관측 커버리지  관측 정확도  확정 정확도  차로전환
#       하단 50% (기존)       60.2%      90.6%      88.6%      15
#       하단 25행             40.3%      95.4%      96.3%       7
#       하단 15행  ← 채택     32.3%      97.4%      98.9%       3
#   관측 커버리지는 절반으로 줄지만 히스테리시스가 받쳐 확정 커버리지는
#   99.5% → 98.1% 로 거의 유지된다.
#
#   ※ 원본 이미지(비-BEV)로 옮기는 안도 재봤으나 이득이 없다 — BEV 는 원본
#     사다리꼴의 호모그래피라 정보량이 같다. 원본 437-458행이 98.5% 로 BEV
#     하단 15행(98.9%) 과 사실상 동률이었다. 공간이 아니라 '어느 행을 보느냐' 가
#     전부였으므로, 별도 BEV/별도 파이프라인을 만들 이유가 없다.
#
#   ★ F(ROLE_SWAP 흰선 확장) 와 반드시 함께 써야 한다 ★ 단독 적용은 직선 구간의
#     차로중심 오추정을 4.9% → 8.9% 로 악화시킨다. current_lane 이 정확해지면
#     W/Y 역할 검사가 성립하는 프레임이 늘어나는데, 검사가 'Y' 만 보고 있으면
#     흰선 쪽 불일치가 교정되지 않은 채 늘어나기 때문이다. F 를 붙이면 5.4%,
#     F+G 까지 붙이면 4.8% 로 기존(4.9%) 아래로 내려간다.
LANE_ID_ROWS = 15

# ── lane_detected 히스테리시스 ──
LD_OK_THRESH     = 0.5
LD_LOST_FRAMES   = 4
LD_REGAIN_FRAMES = 2

# ── dt 강건성 ──
DT_MIN, DT_MAX = 0.02, 0.5



# ════════════════════════════════════════════════
#  무상태 헬퍼
# ════════════════════════════════════════════════

# 검출된 차선픽셀 덩어리 point polyfit하여 1차(직선) or 2차(곡선) x=Ay^2+By+C, 계수 반환
def polyfit_xy(ys, xs, order=POLY_ORDER):
    """BEV에서 차선은 세로로 x에 대응하는 y가 여러개일 수 있음 -> y(세로)를 독립변수로 x = f(y) 피팅"""
    if len(xs) < 5: 
        return None
    order_use = 1 if (ys.max() - ys.min()) < 40 else order # 짧은 세로선에 대해서는 곡률이 오염되지 않도록 1차 피팅하여 곡률 0.0 유지
    fit = np.polyfit(ys, xs, order_use)
    if order_use == 1:
        fit = np.array([0.0, fit[0], fit[1]])
    return fit

# 왼쪽 차선 혹은 오른쪽 차선이 소실된 상황에서 가상의 왼쪽 차선 혹은 오른쪽 차선을 만들어줌
def get_curvature(left_fit, right_fit):
    """곡률 추정. ★ 1차 강등된 fit 은 '측정값 없음'으로 제외한다 ★

    polyfit_xy 는 y 범위가 40px 미만이면 2차를 포기하고 1차로 강등하면서 A 자리에
    리터럴 0.0 을 채운다. 그건 '곡률이 0 이다' 가 아니라 '곡률을 못 쟀다' 인데,
    그대로 모드 판정에 들어가면 급커브에서 straight 로 오판한다 — 낮은 조향
    상한(9.0)과 높은 속도(15.0)가 동시에 걸려 그대로 이탈한다.
      실측(영상 4편): A 가 정확히 0.0 인 프레임이 S자 6.8% / 그 외 1.3%,
      그중 straight 로 판정된 것이 S자 2.5% / 그 외 0.5%.
      (실차 f1114 가 이 케이스 — conf 0.00, curv +0.00000, mode straight, speed 14.5)

    한쪽만 1차 강등이면 나머지 한쪽의 A 를 쓴다. 둘 다면 None(=모름) 을 반환하고,
    호출부가 모드를 동결하거나 bend_ahead 로 대체 판정한다.

    반환: float 또는 None
    """
    cands = [float(f[0]) for f in (left_fit, right_fit)
             if f is not None and f[0] != 0.0]
    if not cands:
        return None
    return sum(cands) / len(cands)

# 히스토그램에서 질량 상위 peak 추출 (질량 게이트 + 비최대 억제)
def _find_peaks(hist, min_mass=PEAK_MIN_MASS, nms=PEAK_NMS_PX, max_peaks=4):
    peaks = []
    h = hist.copy()
    for _ in range(max_peaks):
        x = int(np.argmax(h))
        m = float(h[x])
        if m < min_mass:
            break
        peaks.append((x, m))
        h[max(0, x - nms):x + nms] = 0.0
    return peaks

PEAK_FALLBACK_PIX = 0


def _band_peaks(hist):
    p = _find_peaks(hist)
    if p or PEAK_FALLBACK_PIX <= 0:
        return p
    if float(hist.sum()) >= PEAK_FALLBACK_PIX * 255.0:
        x = int(np.argmax(hist))
        return [(x, float(hist[x]))]
    return []

# BRAKE_LOOKAHEAD_RATIO 기반으로
# bend_ahead <- 곡률보다 먼 지점의 휨을 반환(다가오는 커브를 미리 보는 역할)
def estimate_bend_ahead(left_fit, right_fit, h, far_ratio=BRAKE_LOOKAHEAD_RATIO):
    if left_fit is None or right_fit is None: return 0.0
    y_near = h - 1 # y(x)형태의 fit에서 (y,x)를 얻기 위함()
    y_far  = int(h * (1.0 - far_ratio))
    c_near = (np.polyval(left_fit, y_near) + np.polyval(right_fit, y_near)) / 2 # 가까운 곳 (y값->polyval->x값)으로 차로 중앙 x
    c_far  = (np.polyval(left_fit, y_far)  + np.polyval(right_fit, y_far))  / 2 # 먼 곳 차로 중앙 x
    return float(abs(c_far - c_near)) # 가까운 곳 대비 먼 곳이 얼마나 벗어났는지(x값 차이)


# ════════════════════════════════════════════════
#  검출기 클래스
# ════════════════════════════════════════════════
class LaneDetector:

    def __init__(self, lane_width=LANE_WIDTH_PX):
        self.lane_width = float(lane_width)
        self.prev_left  = None
        self.prev_right = None
        self.current_lane    = None
        self.candidate_lane  = None
        self.candidate_count = 0
        self.prev_error = 0.0
        self._d_err_f   = 0.0   # d_error 저역통과 상태
        self.steer_hist = deque(maxlen=SMOOTH_WINDOW)
        self.prev_steer = 0.0
        self._lost_count = 0
        self._low_conf_count = 0
        self._virt_left  = 0    # 좌/우 각각 real 검출 없이 지난 프레임 수
        self._virt_right = 0
        self._pair_reject = 0   # 좌우 쌍 붕괴로 폐기한 누적 횟수 (진단용)
        self._sep_drop  = 0     # 쌍 간격 상한 초과로 한쪽만 남긴 누적 횟수 (진단용)
        self._ww_count  = 0     # (W,W)+폭위반 누설 카운터
        self._ww_reject = 0     # (W,W) 붕괴로 재획득 강제한 누적 횟수 (진단용)
        self._reacq_skip = 0    # 재획득 불가로 저신뢰 리셋을 보류한 누적 횟수 (진단용)
        self._role_bad  = 0     # W/Y 역할-슬롯 불일치 연속 카운터
        self._role_swap = 0     # 역할 기반 슬롯 교환 누적 (진단용)
        self._geom = None       # 최근 기하 검사 결과 (수직폭/평행비/위반항목)
        self._col_left = self._col_right = None   # 추적 중인 색 ('W'/'Y'/None)
        self.mode = 'straight'
        self.mode_relax_count = 0
        self._mode_hist = deque(maxlen=SPEED_MODE_WINDOW)   # 속도 판정용 (2-4)
        self.prev_speed = MIN_SPEED
        # 기동 직후에는 아직 아무 프레임도 안 봤다. True 로 시작하면 '차선 잡음'을
        # 근거 없이 발행하고, False 가 되려면 나쁜 프레임이 LD_LOST_FRAMES(5) 개
        # 필요해 최대 5프레임 동안 거짓 True 가 나간다. False 로 시작해도
        # LD_REGAIN_FRAMES(2) 프레임이면 올라오므로 정상 인식이 늦어지지 않는다.
        self.lane_detected = False
        self._ld_lost = 0
        self._ld_ok   = 0
        # 검출 픽셀 저장 (디버그/단일선 조향)
        self._left_px  = None
        self._right_px = None
        # 시각화 디버그 (윈도우 사각형, 탐색 모드)
        self._dbg_rects  = []
        self._dbg_search = {'left': None, 'right': None}

    # ------------------------------------------------
    #  슬라이딩 윈도우 차선 검출
    # ------------------------------------------------
    def _sliding_window(self, binary, x_base):
        """x_base 에서 시작해 위로 윈도우 쌓으며 픽셀 수집."""
        h, w = binary.shape
        nz = binary.nonzero()
        nzy = np.array(nz[0]); nzx = np.array(nz[1])
        win_h = h // N_WINDOWS
        x_current = x_base
        lane_inds = []
        for win in range(N_WINDOWS):
            y_lo = h - (win + 1) * win_h
            y_hi = h - win * win_h
            x_lo = int(x_current - WIN_MARGIN)
            x_hi = int(x_current + WIN_MARGIN)
            # 디버그: 윈도우 사각형 기록 (시각화용)
            self._dbg_rects.append((x_lo, y_lo, x_hi, y_hi))
            good = ((nzy >= y_lo) & (nzy < y_hi) &
                    (nzx >= x_lo) & (nzx < x_hi)).nonzero()[0]
            lane_inds.append(good)
            if len(good) > WIN_MINPIX:
                # 윈도우 재중심은 평균이 아니라 중앙값. 윈도우(±44px) 안에 반사가
                # 섞이면 평균은 그쪽으로 끌려가지만 중앙값은 소수 오염을 무시한다.
                # observed_lane 에서 평균→peak 로 바꿔 정답률이 80.7→86.5% 오른 것과
                # 같은 성질이다. 실측(영상 4편): 조향 Δp99 가 S자 22.79→22.44,
                # 그 외 16.96→16.52 로 내려가고 손해는 없었다.
                # (peak(argmax) 도 시험했으나 양선이 60.4→59.6% 로 줄어 중앙값이 나았다)
                x_current = int(np.median(nzx[good]))   # 윈도우 재중심
        if not lane_inds:
            return None, None, None
        lane_inds = np.concatenate(lane_inds)
        xs = nzx[lane_inds]; ys = nzy[lane_inds]
        if len(xs) < MIN_LANE_PIX:
            return None, xs, ys
        return polyfit_xy(ys, xs), xs, ys

    def _margin_search(self, binary, prev_fit):
        """이전 fit 주변만 재탐색 (윈도우 생략, 빠르고 안정)."""
        nz = binary.nonzero()
        nzy = np.array(nz[0]); nzx = np.array(nz[1])
        if len(nzx) == 0:
            return None, None, None
        fit_x = np.polyval(prev_fit, nzy)
        good = (np.abs(nzx - fit_x) < MARGIN_SEARCH).nonzero()[0]
        xs = nzx[good]; ys = nzy[good]
        if len(xs) < MIN_LANE_PIX:
            return None, xs, ys
        return polyfit_xy(ys, xs), xs, ys

    @staticmethod
    def _track_mask(color, white_mask, yellow_mask, combined):
        """추적 중인 색의 채널. 색이 아직 정해지지 않았으면(재획득 직후) 합본."""
        if color == 'Y':
            return yellow_mask
        if color == 'W':
            return white_mask
        return combined

    @staticmethod
    def _classify_color(xs, ys, yellow_mask):
        """수집 픽셀의 노랑 비율로 추적 색 판정 (과반)."""
        if xs is None or len(xs) == 0:
            return None
        frac = float((yellow_mask[ys, xs] > 0).mean())
        return 'Y' if frac > 0.5 else 'W'

    def detect_lane(self, white_mask, yellow_mask):
        """
        노란 중앙선 + 흰 외곽선으로 좌/우 차선 검출.
        노란선 위치로 현재 차로(1/2) 판별.
        반환: left_fit, right_fit, observed_lane, (real_l, real_r),
              (real_left_fit, real_right_fit)
        """
        h, w = white_mask.shape
        vehicle_x = w // 2
        # 디버그 정보 초기화 (시각화용)
        self._dbg_rects = []
        self._dbg_search = {'left': None, 'right': None}
        combined = np.clip(white_mask.astype(np.int16) + yellow_mask, 0, 255).astype(np.uint8)

        # 판정 밴드는 재획득용 히스토그램 밴드와 별개다 (LANE_ID_ROWS 주석 참조)
        lo_band = max(0, h - LANE_ID_ROWS)
        yx = yellow_mask[lo_band:, :].nonzero()[1]
        if len(yx) > MIN_LANE_PIX:
            # ★ 위치를 평균이 아니라 '최강 peak' 로 잡는다 ★
            #   평균은 마스크에 섞인 오염에 그대로 끌려간다. 라바콘이 노란 마스크로
            #   들어오는 것을 실측했고(lane_8 라바콘 구간 노랑 픽셀의 23%),
            #   그때 current_lane 오판이 발생했다. 컬럼 히스토그램의 최강 peak 는
            #   '한 곳에 세로로 쌓인' 것만 고르므로 흩어진 오염에 안 끌린다.
            #   재획득(_histogram_bases)이 쓰는 것과 같은 방식이라 구조도 일관된다.
            #   peak 가 없으면(질량 미달) 기존 평균으로 폴백한다.
            _py = _band_peaks(np.sum(yellow_mask[lo_band:, :], axis=0).astype(np.float64))
            yellow_x = (float(max(_py, key=lambda q: q[1])[0]) if _py
                        else float(np.mean(yx)))
            # 데드밴드: 노란선이 차량축 바로 위면 어느 쪽인지 판정이 성립하지 않는다
            # (LANE_OBS_DEADBAND 주석 참조). 틀린 답 대신 '모름'을 낸다.
            if abs(yellow_x - vehicle_x) < self.lane_width * LANE_OBS_DEADBAND:
                observed_lane = None
            else:
                observed_lane = 1 if yellow_x > vehicle_x else 2   # 노란선이 우측=1차로
        else:
            yellow_x = None
            observed_lane = None

        left_fit = right_fit = None
        lx = ly = rx = ry = None
        if self.prev_left is not None:
            left_fit, lx, ly = self._margin_search(
                self._track_mask(self._col_left, white_mask, yellow_mask, combined),
                self.prev_left)
            self._dbg_search['left'] = 'margin'
        if self.prev_right is not None:
            right_fit, rx, ry = self._margin_search(
                self._track_mask(self._col_right, white_mask, yellow_mask, combined),
                self.prev_right)
            self._dbg_search['right'] = 'margin'

        # 2) 미해결 측만 슬라이딩 윈도우 재획득 — base 는 이때만 계산 (lazy: margin이 도는 대부분의 프레임에서는 pairing 연산 자체가 생략됨)
        if left_fit is None or right_fit is None:
            left_base, right_base = self._histogram_bases(white_mask, yellow_mask, h, vehicle_x)

            y_hist = h - 1 - int(h * HIST_RATIO / 2)      # 히스토그램 밴드 중앙
            near   = self.lane_width * LANE_MIN_SEP_RATIO
            if right_fit is not None and left_base is not None and \
               abs(left_base - np.polyval(right_fit, y_hist)) < near:
                left_base = None
                self._dbg_search['left'] = 'dup'
            if left_fit is not None and right_base is not None and \
               abs(right_base - np.polyval(left_fit, y_hist)) < near:
                right_base = None
                self._dbg_search['right'] = 'dup'

            if left_fit is None:
                if left_base is not None:
                    left_fit, lx, ly = self._sliding_window(combined, left_base)
                    self._dbg_search['left'] = 'window'
                else:
                    self._dbg_search['left'] = 'none'
            if right_fit is None:
                if right_base is not None:
                    right_fit, rx, ry = self._sliding_window(combined, right_base)
                    self._dbg_search['right'] = 'window'
                else:
                    self._dbg_search['right'] = 'none'

        # 프레임 단위 fit 위생 (급변 + 비물리적 곡률)
        left_fit  = self._reject_bad_fit(left_fit,  self.prev_left,  w, h)
        right_fit = self._reject_bad_fit(right_fit, self.prev_right, w, h)

        self._left_px  = (lx, ly) if left_fit  is not None else None
        self._right_px = (rx, ry) if right_fit is not None else None

        # 추적 색 갱신 — 다음 프레임 margin_search 가 볼 채널을 정한다.
        # 매 프레임 다시 판정하므로 한 번 잘못 잡혀도 고착되지 않는다.
        if left_fit is not None:
            self._col_left = self._classify_color(lx, ly, yellow_mask)
        if right_fit is not None:
            self._col_right = self._classify_color(rx, ry, yellow_mask)

        real_left  = left_fit  is not None
        real_right = right_fit is not None

        # ── 좌우 쌍 붕괴 폐기 ──
        # 한쪽 fit 이 반대쪽 차선으로 드리프트해 둘 다 같은 선을 잡은 상태를 잡아낸다.
        # 이 상태는 real=(True,True) 라서 _lost_count 가 안 오르고, conf 도 0.30 으로
        # 고정돼(GEOM_INVALID_PENALTY 주석 참조) 회복 트리거가 영구히 안 걸렸다.
        #
        # 예외 상황 오염 방지 분기:
        #   1) 양쪽 real 일 때만 검사 → 단일선(커브에서 안쪽 선 소실)은 대상 자체가 아님
        #   2) _fill_missing 이전에 검사 → 가상 fit 은 정의상 정확히 lane_width 라 무의미
        #   3) y=h-1(차량 바로 앞)에서만 판정 → y=0 은 2차항 노이즈에 민감
        #      (_reject_jump 가 같은 이유로 h-1 을 쓰는 것과 동일한 근거)
        #   4) 헤딩 틀어짐·커브 기울기는 수평 간격을 perp/cos(theta) 로 '넓히는' 방향이라
        #      최소간격 검사는 원리적으로 오탐이 날 수 없음 (교차는 음수 sep 으로 포착)
        if real_left and real_right:
            sep = float(np.polyval(right_fit, h-1) - np.polyval(left_fit, h-1))
            if sep < self.lane_width * LANE_MIN_SEP_RATIO:
                left_fit = right_fit = None
                self._left_px = self._right_px = None
                # prev 를 즉시 비워 다음 프레임을 히스토그램+슬라이딩 윈도우로 되돌린다
                # (튜너에서 스크럽 시 검출기를 리셋하면 회복되던 것과 같은 효과)
                self.prev_left = self.prev_right = None
                self._col_left = self._col_right = None
                real_left = real_right = False
                self._pair_reject += 1
                self._dbg_search['left'] = self._dbg_search['right'] = 'reject'
            elif sep > self.lane_width * LANE_MAX_SEP_RATIO:
                # 벌어진 쌍 → 더 믿을 만한 한쪽만 남기고 단일선 경로로 넘긴다
                # (LANE_MAX_SEP_RATIO 주석 참조). prev/색도 함께 비워 다음
                # 프레임이 버린 쪽을 margin_search 로 되살리지 않게 한다.
                _nl = 0 if self._left_px  is None else len(self._left_px[1])
                _nr = 0 if self._right_px is None else len(self._right_px[1])
                if self._col_left == 'Y' and self._col_right != 'Y':
                    _keep_left = True
                elif self._col_right == 'Y' and self._col_left != 'Y':
                    _keep_left = False
                else:
                    _keep_left = (_nl >= _nr)
                if _keep_left:
                    right_fit = None
                    real_right = False
                    self._right_px = None
                    self.prev_right = None
                    self._col_right = None
                    self._dbg_search['right'] = 'sep'
                else:
                    left_fit = None
                    real_left = False
                    self._left_px = None
                    self.prev_left = None
                    self._col_left = None
                    self._dbg_search['left'] = 'sep'
                self._sep_drop += 1

        # ★ W/Y 역할이 슬롯을 결정한다 (ROLE_SWAP_FRAMES 주석 참조) ★
        #   self.current_lane 은 아직 이번 프레임 관측이 반영되기 전 값이라
        #   (갱신은 process 에서) 순환 논리가 되지 않는다.
        _bad = 0
        if self.current_lane in (1, 2) and (real_left != real_right):
            _col = self._col_right if real_right else self._col_left
            # 하드룰은 대칭이다: 1차로 → Y=우/W=좌,  2차로 → Y=좌/W=우.
            # 기존엔 'Y' 만 검사해 흰선 단독 추적이 무방비였다 — 실측상 잘못된
            # 슬롯의 62.3% 가 'W + 2차로' 조합이다 (ROLE_SWAP_FRAMES 주석의
            # 표본은 노랑 단독만 세어 이 케이스가 아예 빠져 있었다).
            _want = (None if _col is None else
                     ((self.current_lane == 1) if _col == 'Y' else (self.current_lane == 2)))
            if _want is not None and real_right != _want:
                _bad = 1
        self._role_bad = (self._role_bad + 1) if _bad else 0
        if self._role_bad >= ROLE_SWAP_FRAMES:
            self._role_bad = 0
            self._role_swap += 1
            if real_right:
                left_fit, right_fit = right_fit, None
                real_left, real_right = True, False
                self._col_left, self._col_right = 'Y', None
                self._left_px, self._right_px = self._right_px, None
                self.prev_left, self.prev_right = self.prev_right, None
                self._virt_left, self._virt_right = 0, self._virt_left
            else:
                left_fit, right_fit = None, left_fit
                real_left, real_right = False, True
                self._col_left, self._col_right = None, 'Y'
                self._left_px, self._right_px = None, self._left_px
                self.prev_left, self.prev_right = None, self.prev_left
                self._virt_left, self._virt_right = self._virt_right, 0
            # 차로중심이 lane_width 만큼 이동하므로 D항 상태를 비운다 (스파이크 방지)
            self.prev_error = 0.0
            self._d_err_f = 0.0

        real_left_fit  = left_fit  if real_left  else None
        real_right_fit = right_fit if real_right else None

        # 한쪽만 있으면 반대쪽 가상 생성
        left_fit, right_fit = self._fill_missing(left_fit, right_fit)
        return (left_fit, right_fit, observed_lane, (real_left, real_right),
                (real_left_fit, real_right_fit))

    # ------------------------------------------------
    #  윈도우 재획득 base 선정 — W×Y peak pairing
    # ------------------------------------------------
    def _histogram_bases(self, white_mask, yellow_mask, h, vehicle_x):
        """
        기존 고정 중앙분할(argmax(hist[:w/2]) / argmax(hist[w/2:]))은 횡오차가
        lane_width/2 에 달해 차선 하나가 이미지 중앙에 걸치는 순간, 그 선의 좌/우
        절반이 각각 좌/우 base 로 뽑혀 두 윈도우가 같은 선을 타는 쌍 붕괴를 유발
        (S자 REJ 연속 발생의 근본 원인). 흰/노랑 히스토그램에서 peak 를 따로 뽑아
        간격이 차선폭에 맞는 '흰+노랑' 쌍만 채택한다.

        W×Y 하드룰 근거(트랙 기하): 유효 경계쌍은 항상 [흰 외곽선, 노란 중앙선]
        조합이고 흰 외곽선끼리는 2*lane_width 간격이라 흰+흰 정상 쌍은 존재 불가.
        빛반사는 거의 흰 마스크로 들어오므로(노랑은 LAB b*+HSV 3중 게이트) 반사가
        낀 쌍은 이 제약에서 원천 배제된다.

        반환 (left_base, right_base) — None 인 쪽은 윈도우 생략.
        """
        lo = int(h * (1 - HIST_RATIO))
        pw = _band_peaks(np.sum(white_mask[lo:, :], axis=0).astype(np.float64))
        py = _band_peaks(np.sum(yellow_mask[lo:, :], axis=0).astype(np.float64))

        best_score = 0.0
        pair = None
        for xw, mw in pw:
            for xy, my in py:
                sep = abs(xw - xy)
                if not (self.lane_width * PAIR_SEP_LO < sep < self.lane_width * PAIR_SEP_HI):
                    continue
                # 질량 x 근접 우대: 직선(반사 변별이 필요한 곳)에서는 기대폭에 가까운
                # 쌍이 이기고, 커브에서는 후보가 하나뿐이라 가중치가 결과를 안 바꿈
                score = (mw + my) / (1.0 + abs(sep - self.lane_width) / self.lane_width)
                if score > best_score:
                    best_score = score
                    pair = (min(xw, xy), max(xw, xy))
        if pair is not None:
            return pair

        # 쌍 실패 → 최강 peak 단일선 강등. 반대쪽 base 를 ±lane_width 로 찍어주면
        # 마침 그 자리에 있던 반사를 윈도우가 물어 잘못된 쌍이 부활하므로 일부러
        # 비운다 (이후 _fill_missing 가상 차선 + 단일선 조향 경로가 담당).
        cand = pw + py
        if not cand:
            return None, None
        x0 = max(cand, key=lambda p: p[1])[0]
        return (x0, None) if x0 < vehicle_x else (None, x0)

    def _fill_missing(self, left_fit, right_fit):
        if left_fit is not None and right_fit is None:
            right_fit = left_fit.copy(); right_fit[2] += self.lane_width
        elif right_fit is not None and left_fit is None:
            left_fit = right_fit.copy(); left_fit[2] -= self.lane_width
        return left_fit, right_fit

    def _reject_bad_fit(self, fit, prev_fit, w, h):
        """프레임 단위 fit 위생 — 급변 거부 + 비물리적 곡률 거부."""
        if fit is None:
            return None
        # 1) 급변: 최하단(y=h-1, 차량 바로 앞)에서 판정. y=0(최원거리)은 2차항
        #    노이즈에 민감해 곡선 진입 시 정상 fit 을 오폐기한다.
        if prev_fit is not None:
            jump = abs(np.polyval(fit, h - 1) - np.polyval(prev_fit, h - 1))
            if jump > w * JUMP_REJECT_RATIO:
                return None   # 급변 → 이번 프레임 버림 (prev 유지)
        # 2) 비물리적 곡률: BEV 높이 전체에서 차선폭의 MAX_SWEEP_RATIO 배 넘게
        #    휘는 fit 은 차선이 아니라 잡음이 만든 것. 윈도우가 옆 반사/벽으로
        #    걸어가면 여기서 걸린다. 정상 급커브(A≈0.011)의 2.6배 여유가 있다.
        if abs(fit[0]) * h * h > self.lane_width * MAX_SWEEP_RATIO:
            return None
        # 3) 화면 밖 외삽: 차량이 달리고 있는 차로의 경계는 '차량 바로 앞'(y=h-1)
        #    에서 반드시 화면 안에 있다 (차로 반폭이 130px 인데 BEV 는 640px).
        #    수평에 가까운 성분은 y 범위가 좁아 x=f(y) 외삽이 크게 튀는데,
        #    기존 게이트(급변/곡률)는 이걸 못 잡는다.
        #      lane_9 f2712: 성분이 y 0~81 만 덮는데(bw/bh=3.8) y=109 로 외삽하면
        #      x=-130 → 단일선 분기가 차로중심을 x=-0.5 로 추정 → lat_err -320px
        #      (차선폭의 1.23배) → steer -50 이 여러 프레임 지속.
        #    이탈 중일 수도 있으므로 차선 반폭만큼 여유를 준다.
        x_near = float(np.polyval(fit, h - 1))
        m = self.lane_width * OUT_OF_FRAME_MARGIN
        if not (-m <= x_near <= w + m):
            return None
        return fit

    # ------------------------------------------------
    def _update_lane_state(self, observed_lane):
        if observed_lane is None:
            return
        if observed_lane == self.candidate_lane:
            self.candidate_count += 1
        else:
            self.candidate_lane  = observed_lane
            self.candidate_count = 1
        # 비대칭: 최초 확정은 빠르게, 뒤집기는 어렵게 (LANE_OBS_DEADBAND 주석 참조)
        need = LANE_LATCH_FRAMES if self.current_lane is None else LANE_SWITCH_FRAMES
        if self.candidate_count >= need:
            self.current_lane = self.candidate_lane

    def _classify_mode(self, curvature_abs, bend):
        """곡률 + bend_ahead 로 모드 판정.

        ★ bend 분기 복원 ★ 예전에 '곡률과 중복 + 도달 불가' 로 뺐는데 틀린 판단이었다.
        S자에서 차선은 BEV 에서 '급하게 기운 거의 직선' 으로 보인다 — 곡률 A 는
        0 에 가깝지만 기울기는 크다. 즉 도로가 굽은 게 아니라 차량이 차선에 대해
        크게 틀어진 상태이고, 곡률은 그걸 원리적으로 못 잡는다.
          실측(영상 4편, 현재 mode=straight 인 프레임의 bend_ahead 분포):
              S자   239f  중앙 113.3  p75 175.7  p90 212.6
              그 외 10038f 중앙   6.6  p75  12.8  p90  34.1     ← 중앙값 17배 차이
          임계 80 에서 S자 위험프레임 64.9% 승격 / 정상 직선 오분류 2.2%.
          (실차 f2711 이 이 케이스 — curv -0.00206(straight 임계 미만) 인데 bend 250,
           그래서 mode straight + speed 15.0 으로 커브에 진입했다)

        curvature_abs 는 None(=1차 강등으로 못 잼) 일 수 있다. 그때는 bend 단독으로
        판정하고, bend 도 임계 미만이면 모드를 동결한다 — 근거 없이 straight 로
        내려가지 않게 하는 것이 이 함수의 핵심이다.
        """
        rank = {'straight': 0, 'entry': 1, 'curve': 2}
        tc = None
        if curvature_abs is not None:
            tc = ('curve' if curvature_abs > CURVE_THRESH else
                  'entry' if curvature_abs > STRAIGHT_THRESH else 'straight')
        tb = ('curve' if bend > BEND_CURVE_PX else
              'entry' if bend > BEND_ENTRY_PX else None)
        if tc is None and tb is None:
            self.mode_relax_count = 0
            return self.mode                      # 판정 근거 없음 → 동결
        if tc is None:
            target = tb
        elif tb is None:
            target = tc
        else:
            target = tc if rank[tc] >= rank[tb] else tb   # 높은 쪽 채택
        if rank[target] > rank[self.mode]:
            self.mode = target; self.mode_relax_count = 0
        elif rank[target] < rank[self.mode]:
            self.mode_relax_count += 1
            if self.mode_relax_count >= MODE_RELAX_FRAMES:
                self.mode = target; self.mode_relax_count = 0
        else:
            self.mode_relax_count = 0
        return self.mode

    @staticmethod
    def _perp_widths(rlf, rrf, h):
        """
        근거리/중간/원거리에서의 '수직' 차선폭.

        수평 간격 sep(y) 는 그 지점의 기울기에 따라 perp*sqrt(1+slope^2) 로
        부푼다. 기울기로 나눠 되돌리면, 진짜 평행한 두 선은 곡률·헤딩과 무관하게
        모든 y 에서 같은 값이 된다 → 좁은 게이트와 평행성 검사가 둘 다 성립.
        """
        dl, dr = np.polyder(rlf), np.polyder(rrf)
        out = []
        for y in (h - 1, h // 2, 0):
            sep = float(np.polyval(rrf, y) - np.polyval(rlf, y))
            slope = float((np.polyval(dl, y) + np.polyval(dr, y)) / 2.0)
            out.append(sep / np.sqrt(1.0 + slope * slope))
        return out            # [near, mid, far]

    def _confidence(self, real_detected, real_fits, bev_shape):
        real_left, real_right = real_detected
        rlf, rrf = real_fits
        h, w = bev_shape
        n = int(real_left) + int(real_right)
        self._geom = None
        if n == 0:
            return 0.0
        score = 0.5 if n == 1 else 1.0
        if real_left and real_right:
            pw = self._perp_widths(rlf, rrf, h)
            near, lo, hi = pw[0], min(pw), max(pw)
            # 1) 절대 폭 (근거리 기준) — 벽/반사로 벌어지거나 좁혀진 쌍을 차단
            bad_width = not (self.lane_width * GEOM_WIDTH_LO < near
                             < self.lane_width * GEOM_WIDTH_HI)
            # 2) 평행성 — 상단만 오염된 fit 은 근/원거리 폭이 어긋난다.
            #    h-1 단독 검사가 못 보던 사각지대(오염은 원거리에서 시작)를 덮는다.
            bad_par = (lo <= 0.0) or (hi / lo > GEOM_PARALLEL_TOL)
            center = (np.polyval(rlf, h - 1) + np.polyval(rrf, h - 1)) / 2
            bad_ctr = not (0 < center < w)
            if bad_width: score *= GEOMETRY_INVALID_PENALTY
            if bad_par:   score *= GEOMETRY_INVALID_PENALTY
            if bad_ctr:   score *= GEOMETRY_INVALID_PENALTY
            self._geom = {'perp_near': near, 'perp_far': pw[2],
                          'parallel_ratio': (hi / lo) if lo > 0 else float('inf'),
                          'bad': ''.join(c for c, b in
                                         (('W', bad_width), ('P', bad_par), ('C', bad_ctr)) if b)}
        return float(np.clip(score, 0.0, 1.0))

    def _update_lane_detected(self, confidence):
        if confidence >= LD_OK_THRESH:
            self._ld_ok += 1; self._ld_lost = 0
            if self._ld_ok >= LD_REGAIN_FRAMES: self.lane_detected = True
        else:
            self._ld_lost += 1; self._ld_ok = 0
            if self._ld_lost >= LD_LOST_FRAMES: self.lane_detected = False
        return self.lane_detected

    def _dynamic_lookahead(self, speed):
        # 보간 기준을 MIN/MAX_SPEED 가 아니라 '실제로 쓰는 모드 속도 범위'로 잡는다.
        # MAX_SPEED 는 안전 클립이라 여유를 두는 값인데, 여기에 묶어두면 상한을
        # 올리는 순간 같은 주행속도의 lookahead 가 조용히 짧아진다
        # (12.0 에서 t=1.0 → MAX_SPEED 20 이면 t=0.5, lookahead 0.40→0.35).
        # 모드 속도에서 유도하면 속도를 단계적으로 올려도 매핑이 따라온다.
        t = np.clip((speed - SPEED_REF_LO) / max(1e-6, SPEED_REF_HI - SPEED_REF_LO),
                    0.0, 1.0)
        return LOOKAHEAD_MIN + t * (LOOKAHEAD_MAX - LOOKAHEAD_MIN)

    def _limit_rate(self, steer, mode, dt):
        rate = MAX_STEER_RATE_CURVE if mode == 'curve' else MAX_STEER_RATE_BASE
        lim = rate * dt          # dt 반영: 프레임레이트가 변해도 초당 변화율 일정
        delta = np.clip(steer - self.prev_steer, -lim, lim)
        self.prev_steer = self.prev_steer + delta
        return self.prev_steer

    def _compute_speed(self, mode, dt, blind=False):
        """모드별 고정 목표속도 + 슬루 제한.
        곡률/bend 연속 감속식은 fit 노이즈가 그대로 속도에 실려 직선에서도
        속도가 요동하던 원인이라 제거 (entry/curve 모드가 선제 감속 역할).

        blind = 쓸 fit 이 현재에도 과거에도 없는 상태. 조향을 못 만드는 동안에는
        하한 속도로 내린다. SPEED_DECEL(4/s) 슬루를 그대로 타므로 계단이 아니라
        완만한 감속이고, 회복되면 같은 슬루로 돌아온다."""
        target = MIN_SPEED if blind else float(MODE_GAINS[mode]['speed'])
        speed = float(np.clip(target,
                              self.prev_speed - SPEED_DECEL * dt,
                              self.prev_speed + SPEED_ACCEL * dt))
        speed = float(np.clip(speed, MIN_SPEED, MAX_SPEED))
        self.prev_speed = speed
        return speed

    def _smoothed(self):
        if not self.steer_hist:
            return 0.0
        n = SMOOTH_N_STRAIGHT if self.mode in ('straight', 'entry') else SMOOTH_N_CURVE
        n = min(n, len(self.steer_hist))
        return float(np.mean(list(self.steer_hist)[-n:]))

    # ------------------------------------------------
    def compute_steering(self, left_fit, right_fit, real_detected, bev_shape,
                         curvature, kp, k_head, max_steer, lookahead, dt):
        h, w = bev_shape
        cx = w // 2
        y_eval = int(h * (1.0 - lookahead))
        real_left, real_right = real_detected
        n_real = int(real_left) + int(real_right)

        # 단일선: 오프셋 추종 + heading
        if n_real == 1:
            fit = right_fit if real_right else left_fit
            sign = -1.0 if real_right else 1.0
            offset = self.lane_width / 2.0
            center_far  = np.polyval(fit, y_eval) + sign * offset
            center_near = np.polyval(fit, h-1)   + sign * offset
            # lat_err(위치) / heading(진행방향) 분리 — 양선 분기와 동일한 근거.
            # 기존 error(=center_far-cx) 는 lat_err + heading 이라 두 물리량이 섞여
            # 있었고, kp 를 올리면 선제 조향이 같이 커져 따로 튜닝이 불가능했다.
            lat_err = center_near - cx
            heading = center_far - center_near
            # 단일선 분기에서도 prev_error 를 갱신해야 양쪽 복귀 시 D항이 튀지 않음
            # (기존엔 양쪽 분기에서만 갱신 → 복귀 프레임의 d_raw 가 낡은 값과의 차이로 계산됨)
            self.prev_error = lat_err
            # SINGLE_HEAD_SCALE 은 현재 1.0 (무효). 내리면 실차 이탈 — 주석 참조.
            # heading 에서 sign*offset 은 상쇄되므로 이 값은 실측 fit 의 기울기다.
            steer = (kp * lat_err + KFF_CURVE * fit[0]
                     + k_head * SINGLE_HEAD_SCALE * heading)
            steer = float(np.clip(steer, -max_steer, max_steer))
            self.steer_hist.append(steer)
            return float(self._smoothed())

        # 관측이 하나도 없는 프레임 (conf == 0) 은 새 지령을 만들지 않는다.
        #   기존에는 real=(False,False) 여도 낡은 prev fit 이 남아 있으면 그것으로
        #   lat_err/heading 을 계산해 '측정한 것처럼' 조향을 냈다. 실차 f1114 가
        #   그 상태다 (conf 0.00 인데 steer/speed 가 정상값처럼 나감).
        #   낡은 fit 이라도 있으면 직전 지령 유지, 그마저 없으면 감쇠.
        #   유지는 최대 LOST_RESET_FRAMES(5) 프레임 — 그 뒤 리셋이 prev 를 비워
        #   자동으로 감쇠 경로로 넘어간다.
        if n_real == 0:
            if left_fit is None or right_fit is None:
                return float(self.prev_steer * 0.9)
            return float(self.prev_steer)

        # 양쪽
        if left_fit is not None and right_fit is not None:
            target_far  = (np.polyval(left_fit, y_eval) + np.polyval(right_fit, y_eval)) / 2.0
            center_near = (np.polyval(left_fit, h-1) + np.polyval(right_fit, h-1)) / 2.0
            heading = target_far - center_near
        else:
            # 쓸 fit 이 없다 → 새 지령을 만들 근거가 없다. '마지막으로 실제로 낸
            # 조향'(prev_steer = _limit_rate 의 상태)에서 프레임당 10% 씩 감쇠시킨다.
            #   - steer_hist 를 쓰지 않는 이유: 기존 코드는 hist[-1]*0.9 를 반환만
            #     하고 append 를 안 해 값이 매 프레임 동일했다(감쇠가 아니라 고정).
            #     게다가 리셋이 hist 를 비우면 그 순간 0 으로 튀었다.
            #   - 0 으로 즉시 펴지 않는 이유: 커브에서 몇 프레임 놓쳤을 때 바퀴를
            #     중립으로 되돌리면 호를 벗어난다. 0.9^k 는 약 0.7초에 10% 수준까지
            #     내려가므로, 짧은 소실은 유지하고 긴 소실은 결국 직진으로 수렴한다.
            return float(self.prev_steer * 0.9)

        # 순수 횡오차: 차량축(cx) 대비 '바로 앞' 차로 중앙의 어긋남.
        # 기존 error(=target_far-cx) 는 lat_err + heading 이 항등적으로 성립해
        # 두 물리량이 한 변수에 섞여 있었다 (MODE_GAINS 주석 참조).
        lat_err = center_near - cx
        # D항: 저역통과 + 클램프 (픽셀 노이즈 ±수 px 가 /dt 로 수십 배 증폭되는 것 방지)
        # ★ 미분 대상도 lat_err 로 바꿨다. 기존엔 heading 이 섞인 error 를 미분해
        #   '도로 곡률이 변하는 속도'가 감쇠항에 들어갔다 — D항은 차량의 횡방향
        #   운동만 감쇠해야 한다.
        d_raw = (lat_err - self.prev_error) / dt
        self._d_err_f = (1.0 - KD_LPF_ALPHA) * self._d_err_f + KD_LPF_ALPHA * d_raw
        d_error = float(np.clip(self._d_err_f, -KD_CLAMP, KD_CLAMP))
        self.prev_error = lat_err
        steer = kp * lat_err + KD_LATERAL * d_error + KFF_CURVE * curvature + k_head * heading
        steer = np.clip(steer, -max_steer, max_steer)
        self.steer_hist.append(steer)
        return float(np.clip(self._smoothed(), -max_steer, max_steer))

    # ------------------------------------------------
    #  메인
    # ------------------------------------------------
    def process(self, white_mask, yellow_mask, bev_shape, dt=0.033):
        h, w = bev_shape
        dt = float(np.clip(dt, DT_MIN, DT_MAX))

        (left_fit, right_fit, observed_lane, real_detected,
         real_fits) = self.detect_lane(white_mask, yellow_mask)
        lane_before = self.current_lane
        self._update_lane_state(observed_lane)
        # `lane_before is not None`: None→1/2 는 '차로가 바뀐 것'이 아니라 '처음
        # 확정된 것'이다. 이걸 차선변경으로 오인하면 잘 추적 중이던 prev 를 버린다
        # (노랑이 늦게 보이는 구간을 지나 처음 관측하는 순간에도 발동).
        lane_switched = (lane_before is not None and self.current_lane != lane_before)

        confidence = self._confidence(real_detected, real_fits, bev_shape)

        # 회복: 전체 소실 or 연속 저신뢰 → 리셋
        if real_detected == (False, False):
            self._lost_count += 1
        else:
            self._lost_count = 0
        # `<=`: 페널티 값이 임계와 정확히 같아지는 사각지대 재발 방지
        # (기존 `<` + 페널티 0.3 조합이 conf==0.30 을 영구히 통과시켰음)
        if confidence <= LOW_CONF_THRESH:
            self._low_conf_count += 1
        else:
            self._low_conf_count = 0

        do_reset = self._lost_count >= LOST_RESET_FRAMES
        if not do_reset and self._low_conf_count >= SUSPECT_RESET_FRAMES:
            lb, rb = self._histogram_bases(white_mask, yellow_mask, h, w // 2)
            do_reset = (lb is not None and rb is not None)
            self._low_conf_count = 0
            if not do_reset:
                self._reacq_skip += 1    # 진단: 재획득 불가로 리셋을 보류한 횟수
        
        if do_reset:
            self.prev_error = 0.0
            self._d_err_f   = 0.0
            self._lost_count = 0
            self._low_conf_count = 0
            # ★ 이 프레임의 local fit 을 비워 blind 로 만든다 ★
            #   prev_* 는 건드리지 않는다 — 아래 재커밋 블록이 되돌려놓기 때문이고,
            #   추적 상태 초기화는 재커밋 이후에 따로 한다(위 주석 참조).
            #   비우는 대상은 '이번 프레임의 판단 근거'다:
            #     LOST  경로 — local fit 은 최대 LOST_RESET_FRAMES 만큼 낡은 prev 다.
            #                  그걸로 모드·속도를 정하면 '안 보이는데 직진·고속'이 된다.
            #     SUSPECT 경로 — 기하가 깨졌다고 스스로 판정한 쌍이다. 그 쌍으로
            #                  조향하던 것이 실차에서 관측된 문제였다(f1233/f2719).
            #   blind 가 되면 모드 동결 + 속도 하한 + 조향 감쇠로 넘어간다.
            left_fit = right_fit = None

        if left_fit  is None: left_fit  = self.prev_left
        if right_fit is None: right_fit = self.prev_right

        # prev 에는 real fit 만 저장한다.
        # 기존엔 _fill_missing 을 거친 '가상 fit' 이 prev 로 들어가, 다음 프레임
        # margin_search 가 그 가상 위치 주변을 훑어 다시 real=True 로 잡는 자기 오염
        # 루프가 생겼다. 게다가 한쪽만 검출된 프레임의 confidence 는 0.5 = LD_OK_THRESH
        # 라서 low_conf 리셋도 영원히 안 걸려 루프에 탈출구가 없었다.
        real_left_fit, real_right_fit = real_fits
        if real_left_fit is not None:
            self.prev_left = real_left_fit; self._virt_left = 0
        else:
            self._virt_left += 1
            if self._virt_left >= VIRTUAL_HOLD_FRAMES:
                self.prev_left = None      # 슬라이딩 윈도우로 강제 복귀
                self._col_left = None
        if real_right_fit is not None:
            self.prev_right = real_right_fit; self._virt_right = 0
        else:
            self._virt_right += 1
            if self._virt_right >= VIRTUAL_HOLD_FRAMES:
                self.prev_right = None
                self._col_right = None


        # ★ (W,W) 추적 붕괴 → 재획득 강제 (WW_RESET_FRAMES 주석 참조) ★
        #   margin 추적에는 W×Y 하드룰이 안 걸려 있어, 반사를 문 채 색 래치로
        #   고착된다. 폭 위반이 동반될 때만 발동시켜 '색만 오분류된 정상 쌍'
        #   (lane_7 에서 간격 353 으로 관측)은 건드리지 않는다.
        #   리셋 후에는 _histogram_bases 의 W×Y pairing 이 올바른 쌍을 다시 잡는다
        #   (실측 구간에서 노란선은 계속 보이고 있었으므로 pairing 이 성립한다).
        if (self._col_left == 'W' and self._col_right == 'W'
                and self._geom is not None and 'W' in self._geom['bad']):
            self._ww_count += 1
        else:
            self._ww_count = max(0, self._ww_count - 1)
        ww_reset = self._ww_count >= WW_RESET_FRAMES
        if ww_reset:
            self._ww_count = 0
            self._ww_reject += 1

        # ── 추적 상태 초기화 (세 경로 공통) ──
        #   ★ 반드시 prev 재커밋 '이후' 여야 한다 ★ 앞에 두면 위 재커밋 블록이
        #     같은 프레임의 real fit 으로 되채워 무효가 된다. do_reset 이 실제로
        #     그 상태였다 (do_reset 주석 참조).
        #   세 트리거의 의미는 다르지만 하는 일은 같다 — '지금 물고 있는 것을 놓고
        #   다음 프레임에 히스토그램+슬라이딩 윈도우로 다시 잡아라'.
        #     do_reset      : 전부 소실(LOST) 또는 저신뢰 지속 + 재획득 가능(SUSPECT)
        #     lane_switched : 차로가 바뀌어 좌/우 슬롯의 의미가 뒤집힘
        #     ww_reset      : (W,W) + 폭위반 — 차선이 아닌 것을 물고 있음이 확정
        if do_reset or lane_switched or ww_reset:
            self.prev_left = self.prev_right = None
            self._col_left = self._col_right = None
            self._virt_left = self._virt_right = 0

        # ★ do_reset 프레임은 명시적으로 blind 로 취급한다 ★
        #   위 do_reset 블록이 left_fit/right_fit 을 None 으로 비우지만, 바로 다음
        #   `if left_fit is None: left_fit = self.prev_left` 가 prev 로 되채운다.
        #   그래서 "이 프레임을 blind 로 만든다"는 주석의 의도가 실행되지 않았다 —
        #   실측: do_reset 발동 369회/4298f(S자) 중 99% 가 blind=False 로 빠져나갔고,
        #        그 프레임의 mode/speed 가 stale fit 으로 계산됐다.
        #   prev 되채움 자체는 필요하다 (다음 프레임 margin_search 의 씨앗이자,
        #   추적 상태 초기화는 아래 재커밋 이후 블록이 담당한다).
        #   따라서 fit 을 다시 지우는 대신 blind 플래그만 명시한다:
        #     이번 프레임 관측 = 무효 / 이전 state = 보존 / 제어 = 안전 폴백
        blind = do_reset or (left_fit is None or right_fit is None)

        # None = 양쪽 fit 이 모두 1차 강등이라 곡률을 못 쟀다 (get_curvature 주석)
        curvature     = get_curvature(left_fit, right_fit)
        curv_known    = curvature is not None
        curvature_abs = abs(curvature) if curv_known else None
        # KFF 항에는 '모름' 을 0 으로 넣는다 — 안 잰 값으로 선제 조향하지 않는다.
        curv_ff       = curvature if curv_known else 0.0
        lane_detected = self._update_lane_detected(confidence)

        # bend_ahead 는 더 이상 진단용이 아니라 모드 판정 입력이다 (_classify_mode 주석)
        bend_ahead = estimate_bend_ahead(left_fit, right_fit, h)
        mode       = (self.mode if blind
                      else self._classify_mode(curvature_abs, bend_ahead))
        g          = MODE_GAINS[mode]

        # 속도만 '최근 창 최악 모드' 로 (SPEED_MODE_WINDOW 주석 참조).
        # 게인/상한은 위 mode 를 그대로 쓴다 — 굽음에는 즉시 반응해야 하므로.
        _RANK = {'straight': 0, 'entry': 1, 'curve': 2}
        _INV  = ('straight', 'entry', 'curve')
        self._mode_hist.append(_RANK[mode])
        target_speed = self._compute_speed(_INV[max(self._mode_hist)], dt, blind)
        lookahead    = self._dynamic_lookahead(target_speed)
        kp = g['kp']
        # 모드 정책 상한과 하드웨어 상한 중 작은 쪽. 낼 수 없는 값이 내부 상태
        # (prev_steer / steer_hist)에 남지 않게 여기서 한 번에 묶는다.
        max_steer = min(g['max_steer'], STEER_HW_LIMIT)

        steer = self.compute_steering(
            left_fit, right_fit, real_detected, bev_shape, curv_ff,
            kp=kp, k_head=g['k_head'], max_steer=max_steer,
            lookahead=lookahead, dt=dt)

        if mode == 'straight':
            # soft deadzone: 임계 미만은 0, 초과분만 그대로 통과시켜 경계 불연속 제거
            steer = float(np.sign(steer) * max(0.0, abs(steer) - STRAIGHT_DEADZONE_STEER))
        steer = self._limit_rate(steer, mode, dt)
        steer = float(np.clip(steer, -max_steer, max_steer))

        return {
            'steering_angle': steer, 'target_speed': target_speed,
            'driving_mode': mode, 'current_lane': self.current_lane,
            'lane_confidence': confidence,
            'lane_detected': lane_detected, 'curvature': curv_ff,
            'curv_known': curv_known,           # False = 1차 강등이라 곡률 미측정
            'bend_ahead': bend_ahead, 'lookahead': lookahead,
            'left_fit': left_fit, 'right_fit': right_fit,
            'pair_reject': self._pair_reject,   # 좌우 쌍 붕괴 폐기 누적
            'sep_drop':    self._sep_drop,      # 쌍 간격 상한 초과로 한쪽만 남긴 누적
            'ww_reject':   self._ww_reject,     # (W,W)+폭위반 재획득 강제 누적
            'reacq_skip':  self._reacq_skip,    # 재획득 불가로 리셋 보류한 누적
            'role_swap':   self._role_swap,     # W/Y 역할 기반 슬롯 교환 누적
            'geom': self._geom,                 # 수직폭/평행비/위반항목 (진단)
        }