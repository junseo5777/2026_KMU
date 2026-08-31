"""
detectors/school_zone.py

어린이 보호구역 판단 모듈.
노면의 노란 글씨(보호구역 표시)를 형태로 감지해 진입/해제를 판정,
보호구역 내부 여부(active)를 상태로 유지.

설계:
  - entering/exiting 은 "표지를 본 순간" 만 True (edge)
  - active 는 내부에서 상태 유지 → 표지를 놓치는 프레임이 있어도
    플래닝이 상태를 잃지 않음
  - camera_node 는 update(bgr) 한 번 호출로 'school_zone_active' bool 획득

사용:
  from track_drive.school_zone import SchoolZoneTracker
  sz = SchoolZoneTracker()
  out = sz.update(bgr)
  out['school_zone_active']   # bool → CamData.school_zone_active
"""

import numpy as np
import cv2


# ════════════════════════════════════════════════
#  설정 파라미터 (정상 작동 확인된 값)
# ════════════════════════════════════════════════
# 노면 ROI (하단 = 차량 가까운 노면)
ROI_TOP_RATIO    = 0.60 # 0.55 -> 0.60
ROI_BOTTOM_RATIO = 1.00

# 노란색 HSV
YELLOW_LOWER = np.array([27, 200, 170])
YELLOW_UPPER = np.array([35, 255, 255])

# 글씨 형태 판정
TEXT_MIN_ASPECT = 2.0     # 가로/세로 비 — 글씨는 가로로 넓음
TEXT_MIN_AREA   = 1000     # 개별 덩어리 최소 면적
TEXT_AREA_THRESH= 10000   # 글씨 형태 픽셀 합 임계 (넘으면 보호구역 표지)

# 상태 안정화 (연속 N프레임 일치 시 진입/해제 확정)
CONFIRM_FRAMES = 1


# ════════════════════════════════════════════════
#  무상태 헬퍼
# ════════════════════════════════════════════════

def is_school_zone_text(yellow_mask):
    """
    노란색 영역이 '차선'인지 '보호구역 글씨'인지 형태로 구분.
    - 차선 점선  : 세로로 길쭉한 얇은 선 (aspect 작음)
    - 보호구역 글씨: 가로로 넓거나 덩어리진 형태 (aspect 큼, area 큼)
    Returns: (is_text: bool, text_area: int)
    """
    contours, _ = cv2.findContours(
        yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_like_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / (h + 1e-6)
        area   = cv2.contourArea(cnt)
        if aspect > TEXT_MIN_ASPECT and area > TEXT_MIN_AREA:
            text_like_area += area

    return text_like_area > TEXT_AREA_THRESH, int(text_like_area)


# ════════════════════════════════════════════════
#  상태를 갖는 트래커
# ════════════════════════════════════════════════

class SchoolZoneTracker:
    """
    보호구역 진입/해제 판정 + active 상태 유지.
    이미지(bgr)를 받아 내부에서 글씨 감지 → 상태 갱신까지 처리.
    """

    def __init__(self):
        self.active = False        # 현재 보호구역 내부 여부

        # 글씨 감지 안정화
        self._text_seen_cnt = 0    # 글씨 연속 감지 프레임
        self._text_gone_cnt = 0    # 글씨 연속 미감지 프레임
        self._prev_text = False    # 직전 프레임 글씨 여부 (edge 판정)

    def update(self, bgr):
        """
        bgr: 원본 BGR 이미지
        Returns dict {
          school_zone_active,   # ★ bool (CamData 로 전달)
          entering,             # 진입 표지 본 순간 (edge, 디버깅)
          exiting,              # 해제 표지 본 순간 (edge, 디버깅)
          text_area,            # 글씨 형태 면적 (디버깅)
        }
        """
        # ── ROI 노란 마스크 ──
        h = bgr.shape[0]
        roi = bgr[int(h*ROI_TOP_RATIO):int(h*ROI_BOTTOM_RATIO), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

        # ── 글씨 감지 ──
        text_now, text_area = is_school_zone_text(yellow)

        # ── 안정화 (연속 프레임) ──
        if text_now:
            self._text_seen_cnt += 1
            self._text_gone_cnt = 0
        else:
            self._text_gone_cnt += 1
            self._text_seen_cnt = 0

        text_stable = self._text_seen_cnt >= CONFIRM_FRAMES   # 글씨 확정 감지
        text_absent = self._text_gone_cnt >= CONFIRM_FRAMES   # 글씨 확정 부재

        # ── 진입/해제 edge 판정 ──
        #   글씨를 새로 보기 시작 → 진입/해제 표지 통과
        #   보호구역 안이면 표지 = 해제, 밖이면 표지 = 진입
        entering = False
        exiting  = False
        if text_stable and not self._prev_text:
            # 글씨 표지를 새로 만남 (edge)
            if self.active:
                exiting = True
                self.active = False
            else:
                entering = True
                self.active = True

        # 직전 글씨 상태 갱신 (edge 검출용)
        if text_stable:
            self._prev_text = True
        elif text_absent:
            self._prev_text = False

        return {
            #'school_zone_active': bool(self.active),
            'school_zone_entering': bool(entering),
            'school_zone_exiting': bool(exiting),
            #'text_area':          text_area,
        }

    def reset(self):
        self.active = False
        self._text_seen_cnt = 0
        self._text_gone_cnt = 0
        self._prev_text = False