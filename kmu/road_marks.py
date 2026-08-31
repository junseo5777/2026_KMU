"""
detectors/road_marks.py

노면 마크 감지 (BEV 하단 ROI 기반):
- StopLineDetector       : 정지선(흰색 가로 띠) → bool
- LeftTurnMarkerDetector : 좌회전 트리거(노란 T자 가로막대) → bool
- RoadMarkDetector       : 위 둘을 묶은 통합 래퍼 (한 번 호출로 둘 다)

세 클래스 모두 BEV 하단 ROI 에서 색 마스크 → 분포 판정 → bool + 안정화.
개별 클래스로 따로 테스트하거나, 통합 래퍼로 camera_node 에서 한 번에 사용.

사용:
  from my_cam.detectors.road_marks import RoadMarkDetector
  rm = RoadMarkDetector()
  out = rm.update(bev_bgr)
  out['stop_line_detected'], out['left_turn_trigger']   # 둘 다 bool

  # 개별 사용도 가능
  from my_cam.detectors.road_marks import StopLineDetector
  sl = StopLineDetector()
  sl.update(bev_bgr)['stop_line_detected']
"""

import numpy as np
import cv2


# ════════════════════════════════════════════════
#  공통 설정
# ════════════════════════════════════════════════
ROI_TOP_RATIO    = 0.05     # 하단 ROI (차량 가까운 전방 / 노면 마크 영역)
STOP_ROI_TOP_RATIO = 0.55 # 0.80
ROI_BOTTOM_RATIO = 0.65
STOP_ROI_BOTTOM_RATIO = 0.70 # 0.85

# 색 HSV (시뮬레이터 기준, 튜닝 필요)
WHITE_LOWER  = np.array([0,   0,   180]);  WHITE_UPPER  = np.array([180, 60,  255])
YELLOW_LOWER = np.array([27, 200, 170]);   YELLOW_UPPER = np.array([35, 255, 255])

# 정지선 판정
STOP_WHITE_RATIO = 0.62    # ROI 내 흰색 비율 임계

# 좌회전 T자 판정
BAR_ROW_WIDTH_RATIO = 0.40  # 한 행 노란색 가로 퍼짐 비율 임계
BAR_MIN_ROWS        = 6      # 가로막대 행 연속 최소 개수
MIN_YELLOW_RATIO    = 0.05   # ROI 최소 노란 비율 (노이즈 컷)

# CONFIRM_FRAMES = 1           # 연속 N프레임 일치 시 확정


# ════════════════════════════════════════════════
#  공통 헬퍼
# ════════════════════════════════════════════════

def _crop_roi(img):
    h = img.shape[0]
    return img[int(h*ROI_TOP_RATIO):int(h*ROI_BOTTOM_RATIO), :]

def _crop_roi_stop(img):
    h = img.shape[0]
    return img[int(h*STOP_ROI_TOP_RATIO):int(h*STOP_ROI_BOTTOM_RATIO), :]


def _mask(roi, lower, upper):
    """BGR 이면 HSV 마스크, 단일채널(이진)이면 그대로"""
    if roi.ndim == 2:
        return roi
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper)


def _max_consecutive(mask_bool):
    b = np.asarray(mask_bool, bool)
    if not b.any(): return 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0],b.view(np.int8),[0]))))
    return int((idx[1::2]-idx[0::2]).max())


# ════════════════════════════════════════════════
#  정지선
# ════════════════════════════════════════════════
class StopLineDetector:
    """정지선(흰색 가로 띠) — ROI 흰색 비율 임계 + 안정화"""

    def __init__(self):
        self.detected = False
        self._cnt = 0

    def update(self, img):
        roi   = _crop_roi_stop(img)
        white = _mask(roi, WHITE_LOWER, WHITE_UPPER)
        ratio = float(np.count_nonzero(white)) / white.size

        if ratio >= STOP_WHITE_RATIO:
            self._cnt += 1
        else:
            self._cnt = 0
        self.detected = bool(self._cnt)

        return {'stop_line_detected': self.detected}

    def reset(self):
        self.detected = False
        self._cnt = 0


# ════════════════════════════════════════════════
#  좌회전 T자 마커
# ════════════════════════════════════════════════
class LeftTurnMarkerDetector:
    """좌회전 트리거(노란 T자 가로막대) — 가로로 넓은 노란 행 연속 + 안정화"""

    def __init__(self):
        self.detected = False
        self._cnt = 0

    def update(self, img):
        roi    = _crop_roi(img)
        yellow = _mask(roi, YELLOW_LOWER, YELLOW_UPPER)
        rh, rw = yellow.shape
        yellow_ratio = float(np.count_nonzero(yellow)) / (rh * rw)

        # 행별 노란 가로 퍼짐 → 가로막대 행
        row_yellow = yellow.sum(axis=1) / (255.0 * rw)
        bar_rows   = _max_consecutive(row_yellow >= BAR_ROW_WIDTH_RATIO)

        is_trigger = (bar_rows >= BAR_MIN_ROWS) and (yellow_ratio >= MIN_YELLOW_RATIO)
        if is_trigger:
            self._cnt += 1
        else:
            self._cnt = 0
        self.detected = bool(self._cnt)

        return {'left_turn_trigger': self.detected}

    def reset(self):
        self.detected = False
        self._cnt = 0


# ════════════════════════════════════════════════
#  통합 래퍼
# ════════════════════════════════════════════════
class RoadMarkDetector:
    # 정지선 + 좌회전 마커를 한 번에 (camera_node 용)

    def __init__(self):
        self.stop_line  = StopLineDetector()
        self.left_turn  = LeftTurnMarkerDetector()

    def update(self, img):
        
        # img: BEV BGR 이미지
        # Returns dict (두 검출 결과 합침)
        s = self.stop_line.update(img)
        l = self.left_turn.update(img)
        return {
            'stop_line_detected': s['stop_line_detected'],
            'left_turn_trigger':  l['left_turn_trigger'],
        }

    def reset(self):
        self.stop_line.reset()
        self.left_turn.reset()