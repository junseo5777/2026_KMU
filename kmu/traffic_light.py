"""
detectors/traffic_light.py

신호등 색상/좌회전 화살표 감지.
- 3구 신호등 (빨-주-초)  : 시작 지점에서 1회 판단 (정적)
- 4구 신호등 (빨-주-좌-초): 매 트랙 판단 (좌회전 트리거)

방식 (배경 오검출 방지):
  ① 상단 ROI 크롭
  ② 검은색 신호등 본체(가로 직사각형) 검출 → 신호등 on/off
  ③ 박스 내부에서만 HSV 색 판정 → 배경 나무/지형 초록·주황 완전 배제
  카메라는 "무슨 색인지"만 보고, 좌회전 진입 결정은 planning 이
  경찰차(라이다)·정지선과 결합해 최종 판단.

사용:
  from my_cam.detectors.traffic_light import TrafficLightDetector
  tl = TrafficLightDetector()
  out = tl.update(bgr)   # dict 반환
"""

import numpy as np
import cv2


# ════════════════════════════════════════════════
#  설정
# ════════════════════════════════════════════════
# 상단 ROI (도로/배경 1차 배제)
ROI_TOP_RATIO    = 0.0
ROI_BOTTOM_RATIO = 0.45     # 0.40→0.55 (신호등이 ROI 경계에 잘리지 않게)

# ── 신호등 본체(검은 박스) 검출 ──
BLACK_LOWER = np.array([0, 0, 0])
BLACK_UPPER = np.array([180, 90, 90])   # 저채도·저명도 = 검정/짙은회색 (튜닝)
BOX_MIN_AREA      = 13000     # 신호등 본체 최소 면적 (멀면 작음 → 가까워야 인식)
BOX_MIN_ASPECT    = 1.8      # 가로/세로 비 최소 (가로로 긴 직사각형)
BOX_MAX_ASPECT    = 6.0      # 가로/세로 비 최대
BOX_MIN_ASPECT_CUT= 1.2      # 경계에 잘렸을 때 완화된 하한 (세로 잘려 더 납작)

# ── 신호 색 HSV (박스 내부에서만 적용) ──
RED_LOWER1  = np.array([0,   100, 100]);  RED_UPPER1  = np.array([5,   255, 255])
RED_LOWER2  = np.array([160, 100, 100]);  RED_UPPER2  = np.array([180, 200, 255])
GREEN_LOWER = np.array([50,  100, 100]);  GREEN_UPPER = np.array([70,  220, 255])
YELLOW_LOWER= np.array([10,  100, 140]);  YELLOW_UPPER= np.array([20,  220, 255])

# 색 검출 최소 픽셀 (박스 내부 기준이라 기존보다 작아도 됨)
MIN_PIXELS       = 2000
ARROW_MIN_PIXELS = 1300

CONFIRM_FRAMES   = 3


# ════════════════════════════════════════════════
#  무상태 헬퍼
# ════════════════════════════════════════════════

def crop_top_roi(bgr):
    h = bgr.shape[0]
    return bgr[int(h*ROI_TOP_RATIO):int(h*ROI_BOTTOM_RATIO), :]


def find_light_box(roi_bgr):
    """
    검은색 신호등 본체(가로로 긴 직사각형) 검출.
    Returns: (x, y, w, h) or None
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    black = cv2.inRange(hsv, BLACK_LOWER, BLACK_UPPER)

    # 박스를 메우기 위해 약간 팽창 (등 사이 분리 방지)
    black = cv2.morphologyEx(black, cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(black)
    roi_h, roi_w = roi_bgr.shape[:2]
    best = None
    best_area = 0
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        bx   = stats[i, cv2.CC_STAT_LEFT]
        by   = stats[i, cv2.CC_STAT_TOP]
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        bh   = stats[i, cv2.CC_STAT_HEIGHT]
        if area < BOX_MIN_AREA or bh == 0:
            continue

        # ★ 박스가 ROI/화면 경계에 닿았는지 (잘림 가능성)
        touches_edge = (bx <= 1 or by <= 1 or
                        bx + bw >= roi_w - 1 or by + bh >= roi_h - 1)

        aspect = bw / bh
        # 잘리면 세로가 줄어 aspect 가 커지므로, 경계 접촉 시 하한만 적용
        if touches_edge:
            ok = aspect >= BOX_MIN_ASPECT_CUT     # 상한 무시 (잘려서 더 납작해 보임)
        else:
            ok = BOX_MIN_ASPECT <= aspect <= BOX_MAX_ASPECT
        if not ok:
            continue

        if area > best_area:
            best_area = area
            best = (bx, by, bw, bh)
    return best


def color_masks_in_box(roi_bgr, box):
    """박스 영역 내부에서만 red/green/yellow 마스크 생성"""
    x, y, w, h = box
    sub = roi_bgr[y:y+h, x:x+w]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)

    red = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
        cv2.inRange(hsv, RED_LOWER2, RED_UPPER2))
    green  = cv2.inRange(hsv, GREEN_LOWER,  GREEN_UPPER)
    yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    return red, green, yellow


def largest_blob_centroid(mask):
    """가장 큰 색 덩어리의 (면적, cx, cy). 없으면 (0, None, None)"""
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num <= 1:
        return 0, None, None
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (int(stats[idx, cv2.CC_STAT_AREA]),
            float(centroids[idx][0]), float(centroids[idx][1]))


# ════════════════════════════════════════════════
#  검출기 클래스
# ════════════════════════════════════════════════

class TrafficLightDetector:
    def __init__(self):
        self.confirmed_traffic = "NONE"
        self._cand_color = "NONE"
        self._cand_count = 0
        self._last_box   = None    # 디버깅/시각화용

    def update(self, bgr):
        """
        Returns dict {
          traffic_light_detected, traffic_light_color,
          left_arrow_active, raw_color, box
        }
        """
        roi = crop_top_roi(bgr)

        # ① 신호등 본체 검출 (= on/off 판단)
        box = find_light_box(roi)
        self._last_box = box

        if box is None:
            # 신호등 박스 없음 → 신호 없음
            raw_color = "NONE"
        else:
            # ② 박스 내부에서만 색 판정
            red, green, yellow = color_masks_in_box(roi, box)
            r_area, r_cx, _ = largest_blob_centroid(red)
            g_area, g_cx, _ = largest_blob_centroid(green)
            y_area, _,   _  = largest_blob_centroid(yellow)
            raw_color = self._classify(r_area, g_area, y_area, r_cx, g_cx)

        # ③ 안정화
        if raw_color == self._cand_color:
            self._cand_count += 1
        else:
            self._cand_color = raw_color
            self._cand_count = 1
        if self._cand_count >= CONFIRM_FRAMES:
            self.confirmed_traffic = raw_color

        return {'traffic_light_color': self.confirmed_traffic}

    def _classify(self, r_area, g_area, y_area, r_cx, g_cx):
        red_on    = r_area >= MIN_PIXELS
        green_on  = g_area >= MIN_PIXELS
        arrow_on  = g_area >= ARROW_MIN_PIXELS
        yellow_on = y_area >= MIN_PIXELS

        # 빨강 + 초록 동시 → 좌회전
        if red_on and arrow_on:
            return "LEFT"
        # 초록만 → 직진
        if green_on and not red_on:
            return "GREEN"
        # 빨강만 → 정지
        if red_on and not green_on:
            return "RED"
        # 주황
        if yellow_on:
            return "YELLOW"
        return "NONE"

    def reset(self):
        self.confirmed_traffic = "NONE"
        self._cand_color = "NONE"
        self._cand_count = 0
        self._last_box = None