#!/usr/bin/env python3
"""
yellow_tracer.py — 노란 중앙선 추종 폴백 스택 (라인트레이서)

주 스택(lane_detection3)과 독립적인 최소 구성 폴백. lane_detection3 를 일절
import 하지 않으므로 그쪽 회귀가 이 파일에 전파되지 않는다.

설계 근거:
  주 스택의 검출 실패(좌우 fit 붕괴, 반사 락온, S자 base 오선정)는 전부
  '두 선을 짝짓는' 과정에서 나왔다. 단일선 추종은 짝이 없어 그 실패 모드
  전체가 구조적으로 성립하지 않는다. 게다가 노란 마스크는 LAB b* + HSV
  3중 게이트라 흰색으로 들어오는 빛반사에 원리적으로 둔감하다.

점선 대응 (핵심):
  노란 중앙선은 점선이라 한 프레임 안에서 끊긴다. 슬라이딩 윈도우 대신
  '가로 밴드별 무게중심 → polyfit' 을 쓴다. 노랑이 없는 밴드는 그냥 건너뛰므로
  점선 gap 이 자연스럽게 무시되고, 남은 점들을 지나는 곡선이 보간 역할을 한다.
  대시가 하나만 보여 y 범위가 좁으면 2차는 과적합이라 1차로 강등한다.

소실 리스크 3단 방어:
  TRACK  노랑 fit 성공 → 정상 추종
  COAST  노랑 일시 소실 → 직전 fit 유지 + 조향 감쇠 (점선 gap·순간 가림 흡수)
  WHITE  노랑 장기 소실 → 흰 외곽선에서 차선폭만큼 안쪽(= 노란선 자리)으로 비상 추종
         (짝짓기 아님 — 흰 선 '하나'만 보므로 붕괴 모드가 생기지 않는다)
  전부 실패가 지속되면 정지 (블라인드 주행보다 안전)

추종 정책 (TARGET_OFFSET_PX = 0):
  폴백 중에는 차로 1/2 를 지키지 않고 노란 중앙선 '위' 를 탄다. 실주행 녹화
  (lane_9 f4440~5050)의 부호 있는 실측 중앙값이 -9.5px 로 사실상 0 이었다.
  덕분에 '노란선이 차량 기준 어느 쪽인가'(side) 가 조향에서 빠져, 그 판정이
  뒤집혀 조향 부호가 통째로 반전되는 사고가 원천 제거된다 (상수 주석 참조).
  WHITE 티어만 별도 오프셋(WHITE_OFFSET_PX)과 부호를 쓴다.
  TRACER_ROI 를 넓히면 BEV 픽셀 스케일이 변하므로 오프셋을 다시 재야 한다
  → `-p measure:=true` 모드 제공.

주 스택에서 차용한 것 (근거는 각 상수 주석):
  - fit 위생: 비물리적 곡률 / 화면 밖 외삽 / 급변 거부
  - 조향 항 분리: lat_err(근거리 위치) 와 heading(선 기울기) 를 따로 곱함
  - 모드별 게인(straight/entry/curve) + 곡률 임계 + 강등 히스테리시스
  - MAX_STEER 를 하드웨어 실측 한계(50)에 맞춤
  - 속도 슬루 제한, 기동 시 속도 하한 출발, dt 를 header.stamp 에서 취득

실행:
  python3 yellow_tracer.py
  python3 yellow_tracer.py --ros-args -p camera_topic:=/usb_cam/image_raw/front 
                                      -p yellow_side:=left
  python3 yellow_tracer.py --ros-args -p measure:=true    # 오프셋 실측
"""

import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

try:
    # ros2 run kmu ... — 패키지 모듈로 임포트되는 경우
    from kmu import preprocessing as pp
except ImportError:
    # python3 yellow_tracer.py — 스크립트로 직접 실행하는 경우
    import preprocessing as pp


# ════════════════════════════════════════════════
#  기하 / 캘리브레이션
# ════════════════════════════════════════════════
# None = preprocessing 확정 기하 사용.
# 대시를 더 담고 싶으면 예: (0.58, 0.95) → 이후 measure 모드로 오프셋 재측정 필수.
TRACER_ROI = None

# '노란선 ↔ 차량중심' 목표 수평 거리 (BEV px)
#
# ★ 0 = 노란선 위를 탄다 (라인트레이서) ★
#   정책: 폴백 중에는 차로 1/2 를 지키지 않고 중앙선 자체를 추종한다.
#
#   실측 근거 — lane_9 f4440~5050 (노란 중앙선 추종 주행 녹화, 610프레임):
#       노란선 x - 차량중심 (부호 있음)  중앙 -9.5px  (하단 밴드 기준 -7.6px)
#       선이 오른쪽인 프레임 43% / 왼쪽 57%
#   즉 한쪽으로 치우쳐 따라간 게 아니라 0 근처에서 좌우로 흔들리며 선 위를 갔다.
#   ※ 절대값의 중앙값은 30.5px 인데, 이건 '한쪽으로 30 치우침'이 아니라
#     '0 근처에서 ±30 흔들림'이다. 처음에 이 값을 오프셋으로 잘못 읽었었다.
#
#   0 이 주는 부수 효과가 크다 — used_sign * 0 = 0 이라 '노란선이 차량 기준
#   어느 쪽인가'(side) 가 조향에서 완전히 빠진다. 선 위를 타는 주행에서는 그
#   판정이 원리적으로 불가능했다: 재생 시작 지점 57곳에서 래치 결과가
#   right 22 / left 35 로 갈렸고, 래치가 뒤집히면 조향 부호가 통째로 뒤집힌다.
TARGET_OFFSET_PX = 0.0

# WHITE 티어 전용 오프셋 — 흰 외곽선에서 '노란선이 있어야 할 자리'까지의 거리.
#   차가 노란 중앙선 위를 타므로 흰 외곽선까지는 차선폭 '전체'다 (반폭 아님).
#   TARGET_OFFSET_PX 를 그대로 쓰면 0 → 흰 외곽선 위를 타게 되어 트랙을 벗어난다.
#   부호는 래치하지 않고 매 프레임 흰선 자신의 위치로 정한다 — 외곽선은 차량에서
#   차선폭만큼 떨어져 있어 좌/우가 모호해지지 않는다 (노란선과 다른 점).
WHITE_OFFSET_PX = 260.0

# ── fit 위생 (lane_detection3._reject_bad_fit 에서 차용) ──
#   yellow_tracer 에는 fit 검사가 없어서, 대시가 몇 개만 보이는 프레임의 polyfit 이
#   화면 밖으로 외삽되는 것을 그대로 썼다. 실측: 하단 x 의 std 가 147px (중앙 30.5,
#   p75 56.7 인데 std 147 — 극단값 존재), fit 이 화면 밖인 프레임 1.5%.
#   세 검사를 넣으면 std 147.2 → 43.9, 화면 밖 1.5% → 0% 이고 조향 통계는 불변이다
#   (폐기 19건 = 비물리곡률 17 + 화면밖 2 + 급변 0).
LANE_WIDTH_REF   = 260.0   # 위생 검사 스케일 기준 (주 파이프라인 실측 차선폭)
MAX_SWEEP_RATIO  = 1.0     # |A|*h^2 <= LANE_WIDTH_REF * 이 값
OUT_FRAME_MARGIN = 0.5     # 하단 x 가 [-m, w+m] 밖이면 폐기 (m = 폭 x 이 값)
JUMP_REJECT_RATIO = 0.4    # 직전 fit 대비 하단 x 변화 > w * 이 값 이면 폐기

# ── 점선 보간 (밴드 무게중심) ──
N_BANDS       = 12     # BEV 세로 분할 수 (밴드 하나가 대시보다 얇아야 gap 이 구분됨)
MIN_BAND_PX   = 12     # 밴드를 유효 점으로 인정할 최소 픽셀
MIN_POINTS    = 3      # fit 에 필요한 최소 밴드 점 수
MIN_SPAN_QUAD = 40     # y 범위가 이보다 좁으면 1차로 강등 (대시 1개만 보이는 상황)
TRACK_MARGIN  = 90     # 직전 fit 주변 게이팅 폭 (스퍼리어스 노란 블롭 배제)

# ── 3단 방어 ──
COAST_FRAMES  = 8      # 이 프레임 수까지는 직전 fit 으로 버팀 (점선 gap 흡수)
COAST_DECAY   = 0.92   # COAST 중 조향 감쇠 (블라인드 상태로 계속 꺾지 않도록)
STOP_FRAMES   = 45     # 전부 실패 지속 → 정지
# COAST 만료 즉시 WHITE 로 넘어간다. 그 사이에 '눈 감고 감쇠' 구간을 두면
# 흰선이라는 실제 관측이 있는데도 근거 없이 주행하게 되어 순손해였다.

# ── 제어 (주 스택과 독립된 자체 게인) ──
LOOKAHEAD     = 0.45   # y_eval = h*(1-LOOKAHEAD)
# ★ KP 0.10 → 0.385, K_HEADING 0.35 → 0.20 ★
#   KP 설계기준은 lane_detection3 와 동일하게 잡았다 — '횡오차가 차로 반폭(130px)
#   일 때 max_steer 를 대체로 채운다' → 50/130 = 0.385. 기존 0.10 은 그 1/4 이라
#   헤딩항이 지배했다 (항 분해 실측: 위치 16.4 / 헤딩 15.5, 헤딩 비중 37%).
#   둘은 같은 BEV 픽셀 스케일이라 게인을 직접 비교할 수 있다.
#
#   K_HEADING 은 위치 응답 곡선(|lat_err| 구간별 평균 |steer|)으로 골랐다.
#   lane_9 f4440~5050, offset 30, MAX_STEER 50:
#       KP/K_HEAD      0-20px  20-40  40-65  65-100  100+   단조(4/3)  조향-횡오차 상관
#       0.10 / 0.35      5.8    7.3   18.9   31.1   36.9     1.65        0.59  ← 기존
#       0.385/ 0.325     7.3   13.9   29.5   44.1   43.5     1.49        0.68
#       0.385/ 0.20      5.6   12.8   25.9   41.2   42.5     1.59        0.73  ← 채택
#       0.50 / 0.325     8.2   16.9   33.1   46.2   44.1     1.40        0.68
#   전 조합이 단조(>1)라 lane_detection3 에서 겪은 응답 역전은 없고, 상관이
#   가장 높은 조합을 골랐다. 포화는 전 조합 0%.
#
# ★ 모드 분리 (주 스택 차용) ★
#   위 0.385/0.20 은 '횡오차 = 차로 반폭에서 max_steer' 라는 curve 설계기준으로
#   잡은 값인데, 그걸 전 구간에 하나로 쓰니 직선에서 조향이 과했다.
#   실측(lane_9 f4440~5050, 주 스택 곡률 임계로 분류):
#       모드        프레임   |A| 중앙   |lat| 중앙   |steer| 중앙   p90    >9.0 비율
#       straight   64.1%    0.00034     32.4px      12.7      48.5    61.4%
#       entry       3.8%    0.00358     21.2px      17.9      41.1    78.3%
#       curve      31.8%    0.00682     54.0px      30.1      48.8    89.7%
#     직선 프레임의 61%가 주 스택 straight 상한(9.0)을 넘고 있었다. 항 분해상
#     헤딩 비중은 21.7% 뿐이라 원인은 위치항 — |lat| 32.4px x 0.385 = 12.5.
#   straight/entry 게인은 주 스택 값을 그대로 쓴다 (같은 BEV 픽셀 스케일이라
#   직접 이식된다). curve 만 이 영상의 응답 곡선으로 고른 값을 유지한다.
MODE_GAINS = {
    'straight': dict(kp=0.070, k_head=0.185, max_steer=9.0),
    'entry':    dict(kp=0.117, k_head=0.522, max_steer=30.0),
    'curve':    dict(kp=0.385, k_head=0.200, max_steer=50.0),
}
STRAIGHT_THRESH   = 0.00224   # |A| 임계 (주 스택과 동일 — 같은 BEV 스케일)
CURVE_THRESH      = 0.00404
MODE_RELAX_FRAMES = 4         # 하위 모드로 내려갈 때만 요구 (승격은 즉시)
MAX_STEER     = 50.0   # 하드웨어 상한. 모드 상한과 min() 으로 묶는다
                       # (낼 수 없는 값을 LPF/레이트 상태에 남기지 않는다)
STEER_RATE    = 220.0  # 단위/초 (dt 반영)
DEADZONE      = 1.0
STEER_LPF     = 0.4    # 조향 저역통과 (점선 프레임 간 미세 떨림 억제)

# ── 속도 ──
SPEED_MAX     = 15.0
SPEED_MIN     = 9.0
SPEED_COAST   = 4.0    # COAST/WHITE 단계 감속
STEER_FOR_MIN = 35.0   # 이 조향에서 SPEED_MIN 도달
# 속도 슬루 (단위/초). speed = f(|steer|) 는 조향 노이즈를 그대로 속도에 싣는
# 구조라(주 스택에서 같은 이유로 연속 감속식을 제거했다) 슬루로 한 번 걸러준다.
SPEED_ACCEL   = 20.0
SPEED_DECEL   = 8.0    # COAST/WHITE 진입 시 감속은 빨라야 하므로 주 스택(4.0)보다 큼

DT_MIN, DT_MAX = 0.02, 0.5


class YellowTracer:
    """상태를 가진 순수 로직 (ROS 비의존 — 시뮬/오프라인 검증 가능)."""

    def __init__(self, offset=TARGET_OFFSET_PX, side='auto'):
        self.offset = float(offset)
        self.side = side                # 'left' | 'right' | 'auto'
        self._side_votes = 0
        self.prev_fit = None
        self.lost = 0
        self.prev_steer = 0.0
        self._steer_f = 0.0
        # 기동 시엔 아직 아무 프레임도 안 봤다. SPEED_MAX 로 시작하면 첫 프레임이
        # LOST 여도 슬루가 내려오는 1.4초 동안 눈 감고 최고속으로 달린다.
        # 하한에서 출발해 가속 슬루로 올라오게 한다 (주 스택과 같은 근거).
        self.prev_speed = SPEED_MIN
        self.tier = 'TRACK'
        self.mode = 'straight'
        self._mode_relax = 0
        self._pts = []                  # 디버그: 밴드 점

    # ── 밴드 무게중심 → 점선 보간 fit ──
    def _fit_line(self, mask, prev_fit):
        h, w = mask.shape
        nz = mask.nonzero()
        ys, xs = nz[0], nz[1]
        if len(xs) == 0:
            return None, []
        if prev_fit is not None:      # 직전 fit 주변만 (오검출 블롭 배제)
            keep = np.abs(xs - np.polyval(prev_fit, ys)) < TRACK_MARGIN
            ys, xs = ys[keep], xs[keep]
            if len(xs) == 0:
                return None, []

        band_h = max(1, h // N_BANDS)
        pts = []
        for b in range(N_BANDS):
            lo, hi = b * band_h, min(h, (b + 1) * band_h)
            m = (ys >= lo) & (ys < hi)
            if int(m.sum()) < MIN_BAND_PX:
                continue              # 점선 gap → 그냥 건너뜀 (보간은 fit 이 담당)
            pts.append((float(xs[m].mean()), float(ys[m].mean())))
        if len(pts) < MIN_POINTS:
            return None, pts

        px = np.array([p[0] for p in pts]); py = np.array([p[1] for p in pts])
        order = 2 if (py.max() - py.min()) >= MIN_SPAN_QUAD else 1
        fit = np.polyfit(py, px, order)
        if order == 1:
            fit = np.array([0.0, fit[0], fit[1]])   # [A,B,C] 로 정규화
        return self._hygiene(fit, prev_fit, h, w), pts

    @staticmethod
    def _hygiene(fit, prev_fit, h, w):
        """fit 위생 — lane_detection3._reject_bad_fit 차용 (상단 상수 주석 참조).

        대시가 2~3개만 보이는 프레임의 polyfit 은 y 범위가 좁아 외삽이 크게 튄다.
        차량이 따라가는 선은 '차량 바로 앞'(y=h-1)에서 반드시 화면 안에 있으므로
        그걸 못 지키는 fit 은 보간이 아니라 잡음이다.
        """
        if fit is None:
            return None
        if abs(fit[0]) * h * h > LANE_WIDTH_REF * MAX_SWEEP_RATIO:
            return None                      # 비물리적 곡률
        x_near = float(np.polyval(fit, h - 1))
        m = LANE_WIDTH_REF * OUT_FRAME_MARGIN
        if not (-m <= x_near <= w + m):
            return None                      # 화면 밖 외삽
        if prev_fit is not None:
            if abs(x_near - float(np.polyval(prev_fit, h - 1))) > w * JUMP_REJECT_RATIO:
                return None                  # 급변
        return fit

    def _resolve_side(self, yellow, w):
        """노란선이 차량 기준 어느 쪽인지. auto 는 초기 몇 프레임 투표 후 래치."""
        if self.side in ('left', 'right'):
            return self.side
        xs = yellow.nonzero()[1]
        if len(xs) < 50:
            return None
        self._side_votes += 1 if xs.mean() < w / 2 else -1
        if abs(self._side_votes) >= 5:
            self.side = 'left' if self._side_votes > 0 else 'right'
            return self.side
        return 'left' if self._side_votes > 0 else 'right'

    def process(self, white, yellow, dt=0.033):
        dt = float(np.clip(dt, DT_MIN, DT_MAX))
        h, w = yellow.shape
        cx = w / 2.0
        y_eval = int(h * (1.0 - LOOKAHEAD))

        # side 는 진단 표시용이다 — TARGET_OFFSET_PX=0 이므로 노란선 추종
        # 조향에는 쓰이지 않는다 (상수 주석 참조). WHITE 티어만 부호가 필요한데
        # 그건 아래에서 흰선 자신의 위치로 매 프레임 정한다.
        side = self._resolve_side(yellow, w)

        fit, self._pts = self._fit_line(yellow, self.prev_fit)
        if fit is None and self.prev_fit is not None:
            # 직전 fit 게이팅 때문에 놓쳤을 수 있으니 게이트 없이 1회 재시도
            fit, self._pts = self._fit_line(yellow, None)

        target_off = self.offset        # 노란선 추종: 기본 0 (선 위)
        used_sign = 1.0
        if fit is not None:
            self.prev_fit = fit
            self.lost = 0
            self.tier = 'TRACK'
        else:
            self.lost += 1
            if self.lost <= COAST_FRAMES and self.prev_fit is not None:
                fit = self.prev_fit                 # 점선 gap / 순간 가림 흡수
                self.tier = 'COAST'
            else:
                # 비상: 흰 외곽선 단일 추종 (노랑 반대편이므로 부호 반전).
                # 선을 '하나'만 보므로 짝짓기 붕괴 모드는 여전히 성립하지 않는다.
                wf, self._pts = self._fit_line(white, None)
                if wf is not None and self.lost <= STOP_FRAMES:
                    fit = wf
                    # 흰 외곽선에서 '노란선이 있어야 할 자리'로 차선폭만큼 이동.
                    # 부호는 흰선이 차량 어느 쪽에 있는지로 그 자리에서 정한다
                    # (외곽선은 차선폭만큼 떨어져 있어 좌/우가 모호하지 않다).
                    target_off = WHITE_OFFSET_PX
                    used_sign = 1.0 if float(np.polyval(wf, h - 1)) < cx else -1.0
                    self.tier = 'WHITE'
                else:
                    self.tier = 'LOST'

        if fit is None or self.tier == 'LOST':
            # 근거 없이 꺾지 않는다. 조향은 0 으로 수렴, 장기 소실이면 정지.
            self._steer_f *= COAST_DECAY
            steer = self._rate_limit(self._steer_f, dt)
            if self.lost > STOP_FRAMES:
                self.prev_speed = 0.0       # 비상 정지는 슬루를 거치지 않는다
                speed = 0.0
            else:
                speed = self._limit_speed(SPEED_COAST, dt)
            return self._out(steer, speed, fit, side)

        line_far  = float(np.polyval(fit, y_eval))
        line_near = float(np.polyval(fit, h - 1))
        # ★ 위치(lat_err) / 진행방향(heading) 분리 ★
        #   기존 error 는 '먼 지점' 기준(target_far - cx)이라 lat_err + heading 이
        #   항등적으로 성립했다. 즉 KP 를 올리면 선제 조향이 같이 커져 두 게인을
        #   따로 튜닝할 수 없었다 (주 스택에서 고친 것과 같은 구조).
        #   실측: 분리하니 위치항 평균 16.4 → 13.8 (중복분이 빠짐),
        #         |steer| p90 58.9 → 53.1.
        lat_err = (line_near + used_sign * target_off) - cx
        heading = line_far - line_near      # 선 자체의 기울기 (선제 조향)

        g = MODE_GAINS[self._classify_mode(abs(float(fit[0])))]
        max_steer = min(g['max_steer'], MAX_STEER)
        steer = g['kp'] * lat_err + g['k_head'] * heading
        if self.tier in ('COAST', 'WHITE'):
            steer *= COAST_DECAY            # 근거가 약한 단계는 보수적으로
        steer = float(np.clip(steer, -max_steer, max_steer))

        self._steer_f = (1 - STEER_LPF) * self._steer_f + STEER_LPF * steer
        s = self._steer_f
        if abs(s) < DEADZONE:               # soft deadzone (경계 불연속 제거)
            s = 0.0
        else:
            s = np.sign(s) * (abs(s) - DEADZONE)
        steer = self._rate_limit(s, dt)

        t = min(1.0, abs(steer) / STEER_FOR_MIN)
        speed = SPEED_MAX + (SPEED_MIN - SPEED_MAX) * t
        if self.tier in ('COAST', 'WHITE'):
            speed = min(speed, SPEED_COAST)
        return self._out(steer, self._limit_speed(speed, dt), fit, side)

    def _classify_mode(self, curvature_abs):
        """곡률 → 모드. 승격은 즉시, 강등은 MODE_RELAX_FRAMES 연속 요구 (채터 방지)."""
        if curvature_abs > CURVE_THRESH:
            target = 'curve'
        elif curvature_abs > STRAIGHT_THRESH:
            target = 'entry'
        else:
            target = 'straight'
        rank = {'straight': 0, 'entry': 1, 'curve': 2}
        if rank[target] > rank[self.mode]:
            self.mode, self._mode_relax = target, 0
        elif rank[target] < rank[self.mode]:
            self._mode_relax += 1
            if self._mode_relax >= MODE_RELAX_FRAMES:
                self.mode, self._mode_relax = target, 0
        else:
            self._mode_relax = 0
        return self.mode

    def _rate_limit(self, steer, dt):
        lim = STEER_RATE * dt
        self.prev_steer += float(np.clip(steer - self.prev_steer, -lim, lim))
        return self.prev_steer

    def _limit_speed(self, target, dt):
        """속도 슬루 — speed=f(|steer|) 라 조향 노이즈가 그대로 속도에 실리는 것 억제."""
        self.prev_speed = float(np.clip(target,
                                        self.prev_speed - SPEED_DECEL * dt,
                                        self.prev_speed + SPEED_ACCEL * dt))
        return self.prev_speed

    def _out(self, steer, speed, fit, side):
        return {'steering_angle': float(steer), 'target_speed': float(speed),
                'tier': self.tier, 'mode': self.mode, 'lost': self.lost, 'fit': fit,
                'side': side, 'points': self._pts}


# ════════════════════════════════════════════════
#  ROS 노드
# ════════════════════════════════════════════════
class YellowTracerNode(Node):
    def __init__(self):
        super().__init__('yellow_tracer')
        self.bridge = CvBridge()

        self.declare_parameter('camera_topic', '/image_raw')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('yellow_side', 'auto')     # auto | left | right
        self.declare_parameter('debug_view', True)
        self.declare_parameter('measure', False)          # 오프셋 실측 모드

        topic = self.get_parameter('camera_topic').value
        self.debug_view = self.get_parameter('debug_view').value
        self.measure = self.get_parameter('measure').value
        self.tracer = YellowTracer(side=self.get_parameter('yellow_side').value)
        self.samples = []

        self.cmd_pub = self.create_publisher(
            Twist, self.get_parameter('cmd_vel_topic').value, 10)
        qos = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            durability=rclpy.qos.QoSDurabilityPolicy.VOLATILE)
        self.create_subscription(Image, topic, self.image_cb, qos)

        self.prev_t = None      # monotonic 폴백용
        self._t_img_prev = None  # header.stamp 주 소스
        self.get_logger().info(
            f"yellow_tracer 시작 (topic={topic}, side={self.tracer.side}, "
            f"offset={self.tracer.offset:.0f}px"
            + (", MEASURE 모드 — 제어 미발행)" if self.measure else ")"))
        if self.measure:
            self.get_logger().info("차체를 차로 정중앙·차선과 평행하게 두세요. q 종료")

    def image_cb(self, msg):
        # dt 는 header.stamp(촬영 시각) 우선, monotonic 은 폴백.
        #   monotonic 은 '콜백 호출 간격'이라 스케줄링이 밀리면 영상 간격이 일정한데도
        #   dt 가 부풀려지고, 그 dt 가 조향 레이트리밋·속도 슬루에 그대로 곱해진다.
        #   (camera_node 와 같은 근거 — 시뮬/일부 드라이버는 스탬프를 0 으로 채우므로
        #    폴백은 유지한다.)
        now = time.monotonic()
        st = msg.header.stamp
        t_img = st.sec + st.nanosec * 1e-9
        dt = None
        if t_img > 0.0 and self._t_img_prev is not None:
            d_img = t_img - self._t_img_prev
            if 0.0 < d_img < 1.0:          # 역행(bag 루프)·정지·비정상 스탬프 배제
                dt = d_img
        if dt is None:
            dt = 0.033 if self.prev_t is None else (now - self.prev_t)
        self._t_img_prev = t_img if t_img > 0.0 else None
        self.prev_t = now
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        white, yellow, bev, _span = pp.lane_masks(frame, roi=TRACER_ROI)
        res = self.tracer.process(white, yellow, dt=dt)

        if self.measure:
            self._measure(yellow, res)
        else:
            tw = Twist()
            tw.linear.x = float(res['target_speed'])
            tw.angular.z = float(-res['steering_angle'] * 0.01)
            self.cmd_pub.publish(tw)

        if self.debug_view:
            self._show(bev, white, yellow, res)

    def _measure(self, yellow, res):
        """차로 정중앙 정렬 상태에서 노란선↔차량중심 수평거리 수집."""
        fit = res['fit']
        if fit is None or res['tier'] != 'TRACK':
            return
        h, w = yellow.shape
        d = abs(float(np.polyval(fit, h - 1)) - w / 2.0)
        self.samples.append(d)
        if len(self.samples) % 25 == 0:
            a = np.array(self.samples)
            self.get_logger().info(
                f"[measure] n={len(a)} median={np.median(a):.1f}px "
                f"std={a.std():.1f} → TARGET_OFFSET_PX = {round(float(np.median(a)))}")

    def _show(self, bev, white, yellow, res):
        h, w = yellow.shape
        canvas = cv2.cvtColor(np.clip(white.astype(np.int16) + yellow, 0, 255).astype(np.uint8),
                              cv2.COLOR_GRAY2BGR)
        canvas[yellow > 0] = (0, 255, 255)
        # 밴드 무게중심 (점선 보간의 근거점)
        for (px, py) in res['points']:
            cv2.circle(canvas, (int(px), int(py)), 3, (0, 0, 255), -1)
        fit = res['fit']
        if fit is not None:
            pts = [(int(np.polyval(fit, y)), y) for y in range(0, h, 4)]
            for i in range(len(pts) - 1):
                if 0 <= pts[i][0] < w:
                    cv2.line(canvas, pts[i], pts[i + 1], (255, 200, 0), 2)
            y_eval = int(h * (1.0 - LOOKAHEAD))
            sign = 1.0 if res['side'] != 'right' else -1.0
            if res['tier'] == 'WHITE':
                sign = -sign
            tx = int(np.polyval(fit, y_eval) + sign * self.tracer.offset)
            cv2.circle(canvas, (tx, y_eval), 6, (0, 0, 255), -1)
            cv2.line(canvas, (tx, y_eval), (w // 2, h - 1), (0, 0, 255), 1)
        cv2.line(canvas, (w // 2, 0), (w // 2, h), (120, 120, 120), 1)

        col = {'TRACK': (0, 255, 0), 'COAST': (0, 200, 255),
               'WHITE': (255, 160, 0), 'LOST': (0, 0, 255)}[res['tier']]
        txt = (f"{res['tier']} lost{res['lost']} side:{res['side']}  "
               f"steer {res['steering_angle']:+5.1f} spd {res['target_speed']:.1f}")
        cv2.putText(canvas, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(canvas, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.imshow('yellow_tracer', canvas)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YellowTracerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.measure and node.samples:
            a = np.array(node.samples)
            print("\n" + "=" * 46)
            print(f"TARGET_OFFSET_PX = {round(float(np.median(a)))}   "
                  f"(n={len(a)}, std={a.std():.1f}px)")
            print("=" * 46)
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()