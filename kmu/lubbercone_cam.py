"""
lubbercone_cam.py — 라바콘 구간 카메라 확인 게이트

역할은 '검출'이 아니라 '거부권'이다. 라이다가 라바콘 구간이라고 판단했을 때
카메라가 주황색 근거를 못 대면 DWA 모드로 넘어가지 않게 막는 용도다.
따라서 재현율보다 오탐 억제가 우선이고, 로직은 의도적으로 얇게 유지한다.

    state 측 사용:  dwa_mode = lidar_says_cone AND cam.zone

── 왜 '색 비율' 단독으로는 안 되는가 ──────────────────────────────
주행 코스(체육관)의 바닥이 나무색이라 화면 하단 전체가 주황 계열이다.
느슨한 주황 범위(H 3~22, S>=110)로는 바닥만으로 화면의 20% 이상이 잡힌다.
실측 HSV 분포 (라바콘 = lane_8 콘구간 171만px / 바닥 = lane_10 체육관 92만px):

        H  중앙 :  콘 16  /  바닥 18      ← 겹친다. 색상은 판별력이 없다
        S  중앙 :  콘 157 /  바닥 97      ← 여기서 갈린다
        V  중앙 :  콘 109 /  바닥 116     ← 겹친다

    S 임계별 통과율:
        S>=140   콘 69.2%  바닥 10.8%   (6.4배)
        S>=190   콘 27.8%  바닥  0.8%   (33배)
        S>=210   콘 25.1%  바닥  0.3%   (84배)
    → 판별은 채도가 한다. S>=190 을 쓴다. 콘 픽셀의 28% 만 남지만 비율
      임계를 넘기에는 충분하고(콘 구간 실측 중앙 14.3%), 바닥은 사실상 0 이 된다.

── 왜 가로 분산이 필요한가 ────────────────────────────────────
두 가지를 동시에 막는다.

1) 코스 밖 콘 1개 / 벽의 보관 더미 (lane_10 f1230 이 비율 9.9% 로 진짜 콘
   구간 8~24% 와 겹친다). 실제 대회장에는 없다고 확인됐지만 연습 영상에는 있다.
2) ★ 지름길의 갈색 바닥. S>=190 을 통과하는 바닥 픽셀은 니스 광택 반사라
   몇 개 열에만 뭉친다. 그래서 열 조건이 바닥도 함께 막고 있다 —
   COLS_ON 을 4→3 으로 풀면 진입은 15f 빨라지지만 조명 허용폭이
   0.80~1.20배 → 0.80~1.00배로 좁아지며 지름길이 뚫린다.

        ROI 를 8열로 나눴을 때 채워진 열 수 (중앙값)
            진짜 콘 구간   7 / 8
            단독·보관 콘   2 / 8
            그 외 전부     0 / 8

── 확정 동작점 (lane_6/7/8/9/10) ─────────────────────────────
    S>=200, 열>=3 (열당 충전율 2% 이상), 비율>=1.0%, ON 5연속 / OFF 8연속

        진입 선행     lane_7 +52f / lane_8 +30f   (구간 시작 전에 켜짐)
        콘 구간 ON    86.6%
        지름길 ON      0.0%      ← 반드시 지켜야 하는 조건
        그 외 ON       3.85%     ← 라이다 AND 가 흡수하는 몫

★ 이 동작점에서 판별은 '열 수'가 한다. RATIO_ON=1.0 은 사실상 순수 노이즈
  하한이고, 비율은 구속조건이 아니다. 튜닝할 때 비율을 만지면 아무 일도
  안 일어난다 — COLS_ON 과 COL_FILL, 그리고 S_MIN 이 실제 손잡이다.

  선행 30f 는 공짜가 아니다. 이전 동작점(S190/열4/비율4.0/ON10)은 선행이
  -5f/-1f (구간 시작과 동시) 였지만 조명 허용폭이 0.80~1.20배로 더 넓었다.
  선행을 벌면서 허용폭이 0.90~1.10배로 좁아졌다. 아래 표 참조.

── 비용 ──────────────────────────────────────────────────────
    0.144 ms/프레임 (preprocessing.process 2.06ms 의 7%, 30fps 에서 코어 0.43%)
    ROI 를 먼저 잘라 160x48 로 줄인 뒤 HSV 변환하므로 원본 해상도를 안 건드린다.

── 지름길 갈색 바닥 검증 (요구사항: 이것만은 반드시 구분) ──────────
lane_10 의 지름길 6구간 1,954프레임에서 zone ON = 0.0%.
채도를 0.90~1.10배로 흔들어도 0.0% 유지.

── 진입 시점 검증 ────────────────────────────────────────────
    lane_7  목표창 f346~373  →  f294 ON  (창 시작 52f 전)
    lane_8  목표창 f235~250  →  f205 ON  (창 시작 30f 전)

── ★ S_MIN 을 낮추지 말 것 ──────────────────────────────────
'널널하게' 가 안전 쪽이 아니다. S 를 낮추면 허용폭이 넓어지는 게 아니라
아래로 옮겨가면서 지름길이 뚫린다. 채도 배율을 흔들어 잰 '깨끗한 구간'
(진입 성공 AND 지름길 0%), 현재 동작점(열3/ON5) 기준:

        S_MIN 190   0.80~1.00   ← 1.00배(현재 조명)가 밝은 쪽 끝. 위험
        S_MIN 200   0.90~1.10   ← 현재. 1.0 이 중앙
        S_MIN 210   0.90~1.10   (선행이 26f 로 줄어듦)

  열4/비율4.0/ON10 이던 이전 동작점에서는 S190 이 0.80~1.20 으로 가장
  넓었다. 즉 최적 S_MIN 은 열 조건과 묶여 있다 — 둘 중 하나만 바꾸면 안 된다.

대회장 조명이 다르면:
    콘이 안 잡힌다(어둡다)  → S_MIN 을 190 으로. 단, 지름길 재검증 필수
    지름길이 뚫린다(밝다)   → S_MIN 을 210 으로 (선행 30f→26f)
    둘 다면 조명이 허용폭을 벗어난 것이므로 S_MIN 대신 노출을 손봐야 한다.
hsv_picker 통계를 받으면 콘/바닥 S 중앙값 차이로 위 표를 다시 그릴 수 있다.
"""

import cv2
import numpy as np


# ── ROI: 화면 하단 밴드 ──
#   차 바로 앞 노면. 위로 더 열면 벽에 쌓아둔 보관 콘이 들어온다
#   (lane_10 f7470 의 적/황/청 콘 더미가 y 120~270 에 걸쳐 있다).
ROI_TOP = 0.55
ROI_BOT = 0.95

# ── 축소 해상도 ── 8열로 나눠 떨어지게 160 (열당 20px)
DS_W, DS_H = 160, 48

# ── 주황 게이트 ──
#   H 상한 25 로 콘 고채도 픽셀의 95.2% 를 덮는다. 순적색(H 170~179)에 3.6%
#   가 더 있지만 거부권 용도에 inRange 를 한 번 더 도는 값은 아니다.
H_MIN, H_MAX = 0, 25
S_MIN = 200      # ★ 판별의 핵심. 나무바닥(S 중앙 97) 과 콘(157) 을 가른다
V_MIN = 80       # 그림자·검은 타이어 배제용 하한. 판별력은 없다

# ── 가로 분산 ──
N_COLS = 8
COL_FILL = 0.02          # 한 열이 '채워졌다'고 볼 최소 충전율

# ── 프레임 판정 ──
#   RATIO_ON 은 순수 노이즈 하한일 뿐 구속조건이 아니다 (docstring 동작점 참조).
#   실제 판별은 COLS_ON 이 한다.
RATIO_ON = 1.0           # ROI 내 주황 비율 (%)
COLS_ON = 3              # 채워진 열 수

# ── 시간 히스테리시스 ──
LATCH_ON = 2             # 연속 충족 프레임 → ON
# OFF 는 이탈 타이밍용이 아니다 — 이 게이트는 '진입 시점에만' 읽힌다.
# 짧게 잡을수록 오탐이 켜져 있는 창이 줄어 라이다 오탐과 겹칠 확률이 낮아진다.
# 20→8 로 줄여도 진입까지 끊김 0f (선행 52/30f 불변), 오탐 지속 중앙 40f→21f.
LATCH_OFF = 8            # 연속 미충족 프레임 → OFF


class ConeCamGate:
    """라바콘 구간 카메라 확인 게이트. 프레임당 update(bgr) 한 번."""

    def __init__(self, logger=None):
        self.log = logger
        self.zone = False
        self._on = 0
        self._off = 0
        self._lo = np.array([H_MIN, S_MIN, V_MIN], np.uint8)
        self._hi = np.array([H_MAX, 255, 255], np.uint8)

    def update(self, bgr):
        """returns dict(zone, ratio, cols, raw)

        zone  : 히스테리시스 통과한 최종 판정 (state 에서 라이다와 AND)
        ratio : ROI 내 주황 비율 % (튜닝·로그용)
        cols  : 채워진 열 수 0~8
        raw   : 이번 프레임 단독 판정 (래치 전)
        """
        if bgr is None or bgr.size == 0:
            return self._out(0.0, 0, False)

        h = bgr.shape[0]
        roi = bgr[int(h * ROI_TOP):int(h * ROI_BOT)]
        if roi.size == 0:
            return self._out(0.0, 0, False)

        # 자르고 나서 줄이고, 줄인 뒤에 HSV — 순서를 바꾸면 비용이 몇 배가 된다
        small = cv2.resize(roi, (DS_W, DS_H), interpolation=cv2.INTER_AREA)
        mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV),
                           self._lo, self._hi)

        ratio = 100.0 * float(mask.mean()) / 255.0
        # (48, 8, 20) 로 접어 열 단위 평균 — reshape 은 복사가 없다
        col_fill = mask.reshape(DS_H, N_COLS, DS_W // N_COLS).mean(axis=(0, 2))
        cols = int((col_fill > COL_FILL * 255.0).sum())

        raw = (ratio >= RATIO_ON) and (cols >= COLS_ON)
        if raw:
            self._on += 1
            self._off = 0
        else:
            self._off += 1
            self._on = 0

        if not self.zone and self._on >= LATCH_ON:
            self.zone = True
            self._say(f"라바콘 구간 ON (비율 {ratio:.1f}% / 열 {cols})")
        elif self.zone and self._off >= LATCH_OFF:
            self.zone = False
            self._say("라바콘 구간 OFF")

        return self._out(ratio, cols, raw)

    def reset(self):
        self.zone = False
        self._on = self._off = 0

    # ──
    def _out(self, ratio, cols, raw):
        return {'zone': self.zone, 'ratio': ratio, 'cols': cols, 'raw': raw}

    def _say(self, m):
        if self.log is not None:
            self.log.info(f"[cone] {m}")