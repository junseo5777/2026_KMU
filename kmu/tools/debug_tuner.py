# debug_tuner.py
import cv2
import numpy as np
import os
import sys
import glob

# ════════════════════════════════════════════════
#  경로 설정
# ════════════════════════════════════════════════
INPUT_DIR   = os.path.expanduser("~/Videos/lane_frame")
OUTPUT_DIR = os.path.expanduser("~/Pictures/kmued")
os.makedirs(OUTPUT_DIR, exist_ok=True)
FRAME_LIST = sorted(glob.glob(os.path.join(INPUT_DIR, '트랙_직선주행.jpg')))

# ════════════════════════════════════════════════
#  ★ morphology 커널 (clean 함수용) — 여기서 조절
#  MORPH_OPEN_K : 작을수록 약하게 (작은 반사만 제거)
#  MORPH_CLOSE_K: (가로, 세로). 세로를 크게 → 점선 세로로 잇기
#
#  [중요] CLOSE 세로값은 "점선의 빈틈"보다 커야 이어짐!
#    예) BEV에서 점선 간격이 22px 이면 세로 25~35 필요
#    debug_06 결과에서 점선이 안 이어지면 세로값을 키울 것
#    (너무 크면 다른 것까지 뭉개지니 빈틈보다 약간 크게)
# ════════════════════════════════════════════════
MORPH_OPEN_K  = (3, 3)     # 작은 반사/노이즈 제거
MORPH_CLOSE_K = (5, 25)    # 점선 → 연속선 (세로로 길게, 빈틈보다 크게)

for FRAME_PATH in FRAME_LIST:
    image_raw = cv2.imread(FRAME_PATH)
    if image_raw is None:
        print(f"파일 없음: {FRAME_PATH}")
        sys.exit(1)

    # ── Step 0: 토픽 해상도로 resize ──
    SIM_H = 480
    SIM_W = 640
    image = cv2.resize(image_raw, (SIM_W, SIM_H))
    h, w  = image.shape[:2]
    print(f"원본 shape : {image_raw.shape[0]} x {image_raw.shape[1]}")
    print(f"리사이즈 후: {h} x {w}  (SIM_H={SIM_H}, SIM_W={SIM_W})")


# ════════════════════════════════════════════════
#  Step 1: ROI ratio 수평선 위치 확인
# ════════════════════════════════════════════════
def save_roi_preview(img, ratios=[0.40, 0.50, 0.60, 0.65, 0.70, 0.93]):
    vis    = img.copy()
    colors = [(0,255,0),(0,200,255),(0,165,255),(0,80,255),(0,0,255),(255,255,0)]
    for ratio, color in zip(ratios, colors):
        y = int(img.shape[0] * ratio)
        cv2.line(vis, (0, y), (img.shape[1], y), color, 2)
        cv2.putText(vis, f"ratio={ratio:.2f}", (8, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    out = os.path.join(OUTPUT_DIR, 'debug_01_roi_ratio.png')
    cv2.imwrite(out, vis)
    print(f"[Step1] 저장: {out}")

save_roi_preview(image)

ROI_TOP_RATIO    = 0.70
ROI_BOTTOM_RATIO = 0.93


# ════════════════════════════════════════════════
#  Step 2: 사다리꼴 ROI 마스킹
# ════════════════════════════════════════════════
ROI_PTS_RATIO = [
    (0.18, 0.93),   # 좌하단
    (0.355, 0.70),  # 좌상단
    (0.665, 0.70),   # 우상단
    (0.855, 0.93),   # 우하단
]

def apply_roi_mask(img, pts_ratio):
    h, w  = img.shape[:2]
    pts   = np.array([(int(x*w), int(y*h)) for x, y in pts_ratio], dtype=np.int32)
    mask  = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(img, img, mask=mask)
    roi_top    = int(h * ROI_TOP_RATIO)
    roi_bottom = int(h * ROI_BOTTOM_RATIO)
    cropped = masked[roi_top:roi_bottom, 0:w]   # ★ top~bottom 띠만 크롭
    return cropped, roi_top, roi_bottom, pts

def save_trapezoid_preview(img, pts_ratio):
    masked_crop, roi_top, roi_bottom, pts = apply_roi_mask(img, pts_ratio)
    h, w = img.shape[:2]
    vis_orig = img.copy()
    cv2.polylines(vis_orig, [pts], isClosed=True, color=(255,0,0), thickness=2)
    cv2.line(vis_orig, (0, roi_top),    (w, roi_top),    (0,255,0),   1)  # top
    cv2.line(vis_orig, (0, roi_bottom), (w, roi_bottom), (0,255,255), 1)  # bottom
    for i, p in enumerate(pts):
        cv2.circle(vis_orig, tuple(p), 6, (0,0,255), -1)
        label = ['LL','LH','RH','RL'][i]
        ratio = pts_ratio[i]
        cv2.putText(vis_orig, f"{label}({ratio[0]:.2f},{ratio[1]:.2f})",
                    (p[0]+6, p[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
    # 크롭 띠를 원래 세로 위치에 맞춰 위/아래로 패딩 (top~bottom 사이에 배치)
    pad_top    = roi_top
    pad_bottom = h - roi_bottom
    padded = cv2.copyMakeBorder(masked_crop, pad_top, pad_bottom, 0, 0,
                                cv2.BORDER_CONSTANT, value=0)
    compare = np.hstack([vis_orig, padded])
    out = os.path.join(OUTPUT_DIR, 'debug_02_trapezoid.png')
    cv2.imwrite(out, compare)
    print(f"[Step2] 저장: {out}")
    print(f"        ROI 크롭 shape: {masked_crop.shape[0]} x {masked_crop.shape[1]}"
          f"  (top={roi_top}, bottom={roi_bottom})")
    return masked_crop

roi_image = save_trapezoid_preview(image, ROI_PTS_RATIO)


# ════════════════════════════════════════════════
#  Step 3: BEV 변환
# ════════════════════════════════════════════════
BEV_SRC_RATIO = [
    (0.20, 0.92),   # 좌하
    (0.37, 0.01),  # 좌상
    (0.65, 0.01),   # 우상
    (0.83, 0.92),   # 우하
]

def save_bev_preview(roi_img, src_ratio):
    h, w  = roi_img.shape[:2]
    src   = np.float32([(x*w, y*h) for x, y in src_ratio])
    MARGIN = 0.20
    dst   = np.float32([
        [w * MARGIN,     h], [w * MARGIN,     0],
        [w * (1-MARGIN), 0], [w * (1-MARGIN), h],
    ])
    M   = cv2.getPerspectiveTransform(src, dst)
    bev = cv2.warpPerspective(roi_img, M, (w, h))
    bev_vis = bev.copy()
    cv2.line(bev_vis, (w//2, 0), (w//2, h), (0, 255, 0), 1)
    roi_vis = roi_img.copy()
    labels  = ['LL','LH','RH','RL']
    for i, (px, py) in enumerate(src):
        cv2.circle(roi_vis, (int(px), int(py)), 6, (0,0,255), -1)
        cv2.putText(roi_vis, f"{labels[i]}", (int(px)+5, int(py)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
    cv2.polylines(roi_vis, [src.astype(np.int32).reshape(-1,1,2)], True, (255,0,0), 2)
    compare = np.hstack([roi_vis, bev_vis])
    out = os.path.join(OUTPUT_DIR, 'debug_03_bev.png')
    cv2.imwrite(out, compare)
    print(f"[Step3] 저장: {out}")
    print(f"        BEV shape: {bev.shape[0]} x {bev.shape[1]}")
    return bev

bev_image = save_bev_preview(roi_image, BEV_SRC_RATIO)


# ════════════════════════════════════════════════
#  Step 4: HLS 이진화 → combined 마스크 반환
# ════════════════════════════════════════════════
def save_hls_preview(bev_img):
    hls = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
    mask_w = cv2.inRange(hls, np.array([0,   150, 0]), np.array([180, 255, 140]))   # 흰색
    mask_y = cv2.inRange(hsv, np.array([15,  80,  80]), np.array([35, 255, 255]))  # 노란색
    combined = cv2.bitwise_or(mask_w, mask_y)
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    morph = cv2.morphologyEx(combined_bgr, cv2.MORPH_OPEN, kernel=np.ones((5,5),np.uint8),iterations=1)
    morph = cv2.morphologyEx(combined_bgr, cv2.MORPH_CLOSE, kernel=np.ones((5,5),np.uint8),iterations=2)
    compare = np.hstack([bev_img, morph])
    out = os.path.join(OUTPUT_DIR, 'debug_04_hls.png')
    cv2.imwrite(out, compare)
    print(f"[Step4] 저장: {out}")
    return combined   # ★ Step6(clean)에 전달

hls_mask = save_hls_preview(bev_image)


# ════════════════════════════════════════════════
#  Step 5: Canny (참고용)
# ════════════════════════════════════════════════
def save_canny_preview(bev_img):
    gray_bev = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
    blurred_bev = cv2.GaussianBlur(bev_img, (5,5), 0)
    v = np.median(blurred_bev)
    sigma = 0.33
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    canny_bev_all = cv2.Canny(blurred_bev, lower_thresh, upper_thresh)
    canny = cv2.morphologyEx(canny_bev_all, cv2.MORPH_OPEN, kernel=np.ones((5,5),np.uint8))
    # canny = cv2.morphologyEx(canny1, cv2.MORPH_CLOSE, kernel=np.ones((5,5), np.uint8))
    out = os.path.join(OUTPUT_DIR, 'debug_05_canny.png')
    cv2.imwrite(out, canny)
    print(f"[Step5] 저장: {out}")

save_canny_preview(bev_image)


# ════════════════════════════════════════════════
#  Step 6: clean (morphology) — before/after 비교 저장
#  open(작은 반사 제거) → close(점선 잇기) 효과 체감
# ════════════════════════════════════════════════
def clean(mask):
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_OPEN_K)
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_CLOSE_K)
    opened = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  ko)   # 작은 반사 제거
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kc)   # 점선 → 연속
    return opened, closed

def save_clean_preview(mask):
    opened, closed = clean(mask)
    # 4단계 나란히: 원본마스크 | open만 | open+close(최종)
    before = cv2.cvtColor(mask,   cv2.COLOR_GRAY2BGR)
    mid    = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    after  = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    # 라벨
    cv2.putText(before, "1.HLS mask",   (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(mid,    "2.OPEN",       (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
    cv2.putText(after,  "3.OPEN+CLOSE", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,80,255), 2)
    compare = np.hstack([before, mid, after])
    out = os.path.join(OUTPUT_DIR, 'debug_06_clean.png')
    cv2.imwrite(out, compare)
    print(f"[Step6] 저장: {out}")
    print(f"        OPEN_K={MORPH_OPEN_K} CLOSE_K={MORPH_CLOSE_K}")
    print(f"        마스크 픽셀: HLS={int((mask>0).sum())} "
          f"→ OPEN={int((opened>0).sum())} → CLOSE={int((closed>0).sum())}")
    return closed

final_mask = save_clean_preview(hls_mask)

# ════════════════════════════════════════════════
#  Step 7: HSV 이진화 → combined 마스크 반환
# ════════════════════════════════════════════════
def save_hls_preview(bev_img):
    hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
    mask_w = cv2.inRange(hsv, np.array([0,   0, 160]), np.array([180, 40, 255]))   # 흰색
    mask_y = cv2.inRange(hsv, np.array([15,  80,  80]), np.array([35, 255, 255]))  # 노란색
    combined = cv2.bitwise_or(mask_w, mask_y)
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    compare = np.hstack([bev_img, combined_bgr])
    out = os.path.join(OUTPUT_DIR, 'debug_07_hsv.png')
    cv2.imwrite(out, compare)
    print(f"[Step7] 저장: {out}")

hsv_mask = save_hls_preview(bev_image)


print("\n전체 완료. 결과 파일:")
for f in ['debug_01_roi_ratio.png','debug_02_trapezoid.png',
          'debug_03_bev.png','debug_04_hls.png','debug_05_canny.png','debug_06_clean.png', 'debug_07_hsv.png']:
    print(f"   {os.path.join(OUTPUT_DIR, f)}")
print(f"\n탐색기 접근: \\\\wsl$\\Ubuntu-22.04{OUTPUT_DIR.replace('/', chr(92))}")