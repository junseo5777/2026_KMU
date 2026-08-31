# debug_tuner_video.py
# ════════════════════════════════════════════════════════════
#  촬영한 주행 영상(mp4)을 프레임 추출 없이 바로 열어
#  ROI / BEV / HLS(흰색) / HSV(노랑) / morphology(빛반사·점선) 를
#  트랙바로 실시간 튜닝하는 도구.
#
#  기존 debug_tuner.py 는 개별 사진 1장에 대해 debug_01~07 을 저장했다면,
#  이 스크립트는 영상 전체를 스크럽하며 파이프라인 결과를 라이브로 본다.
#  → 1레인/2레인/중앙 여러 영상에서 파라미터가 견고한지 바로 확인 가능.
#
#  [실행]
#    python3 debug_tuner_video.py --all                       # ~/Videos/lane 전체
#    python3 debug_tuner_video.py --video ~/Videos/lane/xxx.mp4
#
#  [조작키]
#    space : 재생 / 일시정지
#    d / a : 다음 / 이전 프레임 (일시정지 상태에서 한 칸씩)
#    n     : 다음 영상 (--all 일 때)
#    s     : 현재 트랙바 값 콘솔 출력 + params.txt 저장 (복붙용)
#    r     : 트랙바 기본값으로 리셋
#    q/ESC : 종료
#
#  [주의] cv2 GUI(창)가 떠야 함. 자이카 데스크톱 환경에서 실행.
# ════════════════════════════════════════════════════════════
import cv2
import numpy as np
import os
import glob
import argparse

# ── 경로 ──
INPUT_DIR  = os.path.expanduser("~/Videos/lane")
OUTPUT_DIR = os.path.expanduser("~/Pictures/kmued")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIM_W, SIM_H = 640, 480     # 토픽 해상도(파이프라인 기준)

# ════════════════════════════════════════════════════════════
#  기하 파라미터(상수) — 카메라 장착이 바뀌면 여기서 수정
#  (색/모폴로지는 아래 트랙바로 실시간 조절)
# ════════════════════════════════════════════════════════════
# 사다리꼴 ROI 꼭짓점 (전체 프레임 기준 비율)  LL, LH, RH, RL
ROI_PTS_RATIO = [
    (0.18, 0.93),   # 좌하단
    (0.355, 0.70),  # 좌상단
    (0.665, 0.70),   # 우상단
    (0.855, 0.93),   # 우하단
]
# BEV 원근변환 소스점 (크롭된 ROI 기준 비율)  LL, LH, RH, RL
BEV_SRC_RATIO = [
    (0.20, 0.92),   # 좌하
    (0.37, 0.01),  # 좌상
    (0.65, 0.01),   # 우상
    (0.83, 0.92),   # 우하
]

BEV_MARGIN = 0.20

# ════════════════════════════════════════════════════════════
#  트랙바 기본값  (debug_tuner.py 기본값과 동일하게 시작)
# ════════════════════════════════════════════════════════════
DEFAULTS = {
    "roi_top(%)":  65,     # ROI 상단 비율 x100
    "roi_bot(%)":  90,     # ROI 하단 비율 x100
    # 흰색(HLS): L>=Lmin 이고 S<=Smax
    "W_L_min":    150,
    "W_S_max":    140,
    # 노랑(HSV): H범위 + S>=Smin + V>=Vmin
    "Y_H_min":     15,
    "Y_H_max":     35,
    "Y_S_min":     80,
    "Y_V_min":     80,
    # morphology
    "open_k":       3,     # OPEN 커널(정사각) — 작은 반사 제거
    "close_w":      5,     # CLOSE 가로
    "close_h":     25,     # CLOSE 세로 — 점선 잇기(빈틈보다 크게)
}
# 트랙바 최댓값
MAXV = {
    "roi_top(%)": 100, "roi_bot(%)": 100,
    "W_L_min": 255, "W_S_max": 255,
    "Y_H_min": 179, "Y_H_max": 179, "Y_S_min": 255, "Y_V_min": 255,
    "open_k": 15, "close_w": 25, "close_h": 51,
}

CTRL_WIN = "controls"
VIEW_WIN = "lane tuner (space=play, s=save, n=next, q=quit)"


def _nothing(x):
    pass


def build_controls():
    cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CTRL_WIN, 460, 520)
    for name, val in DEFAULTS.items():
        cv2.createTrackbar(name, CTRL_WIN, val, MAXV[name], _nothing)


def reset_controls():
    for name, val in DEFAULTS.items():
        cv2.setTrackbarPos(name, CTRL_WIN, val)


def get_params():
    p = {name: cv2.getTrackbarPos(name, CTRL_WIN) for name in DEFAULTS}
    # 커널은 최소 1
    p["open_k"]  = max(1, p["open_k"])
    p["close_w"] = max(1, p["close_w"])
    p["close_h"] = max(1, p["close_h"])
    # top < bot 보장
    if p["roi_top(%)"] >= p["roi_bot(%)"]:
        p["roi_bot(%)"] = min(100, p["roi_top(%)"] + 1)
    return p


# ════════════════════════════════════════════════════════════
#  파이프라인 (debug_tuner.py 와 동일 로직, 파라미터만 인자화)
# ════════════════════════════════════════════════════════════
def apply_roi(image, top_r, bot_r):
    h, w = image.shape[:2]
    pts = np.array([(int(x * w), int(y * h)) for x, y in ROI_PTS_RATIO], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    masked = cv2.bitwise_and(image, image, mask=mask)
    roi_top = int(h * top_r)
    roi_bot = int(h * bot_r)
    cropped = masked[roi_top:roi_bot, 0:w]
    return cropped, roi_top, roi_bot, pts


def to_bev(roi_img):
    h, w = roi_img.shape[:2]
    if h < 2 or w < 2:
        return roi_img.copy()
    src = np.float32([(x * w, y * h) for x, y in BEV_SRC_RATIO])
    dst = np.float32([
        [w * BEV_MARGIN,       h], [w * BEV_MARGIN,       0],
        [w * (1 - BEV_MARGIN), 0], [w * (1 - BEV_MARGIN), h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(roi_img, M, (w, h))


def color_mask(bev, p):
    hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
    mask_w = cv2.inRange(hls,
                         np.array([0,   p["W_L_min"], 0]),
                         np.array([180, 255, p["W_S_max"]]))
    mask_y = cv2.inRange(hsv,
                         np.array([p["Y_H_min"], p["Y_S_min"], p["Y_V_min"]]),
                         np.array([p["Y_H_max"], 255, 255]))
    return cv2.bitwise_or(mask_w, mask_y), mask_w, mask_y


def clean(mask, p):
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (p["open_k"], p["open_k"]))
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (p["close_w"], p["close_h"]))
    opened = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  ko)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kc)
    return closed


# ── 표시용 유틸 ──
def label(img, text, color=(0, 255, 255)):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def panel(img, size=(480, 360)):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, size)


def render(frame_bgr, p):
    image = cv2.resize(frame_bgr, (SIM_W, SIM_H))

    # 1) 원본 + ROI/사다리꼴 오버레이
    roi_crop, roi_top, roi_bot, pts = apply_roi(image, p["roi_top(%)"]/100.0,
                                                p["roi_bot(%)"]/100.0)
    vis = image.copy()
    cv2.polylines(vis, [pts], True, (255, 0, 0), 2)
    cv2.line(vis, (0, roi_top), (SIM_W, roi_top), (0, 255, 0), 1)
    cv2.line(vis, (0, roi_bot), (SIM_W, roi_bot), (0, 255, 255), 1)

    # 2) BEV
    bev = to_bev(roi_crop)

    # 3) 색 마스크(흰+노랑)
    combined, mask_w, mask_y = color_mask(bev, p)

    # 4) morphology 최종
    final = clean(combined, p)

    # 2x2 그리드
    top_row = np.hstack([panel(label(vis.copy(),  "1.ORIG+ROI")),
                         panel(label(bev.copy(),  "2.BEV"))])
    bot_row = np.hstack([panel(label(cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR), "3.HLS+HSV mask")),
                         panel(label(cv2.cvtColor(final, cv2.COLOR_GRAY2BGR),    "4.CLEAN(final)"))])
    grid = np.vstack([top_row, bot_row])

    # 픽셀 수 표시(빛반사 과검출 판단용)
    info = f"mask px: raw={int((combined>0).sum())}  final={int((final>0).sum())}"
    cv2.putText(grid, info, (8, grid.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return grid


def save_params(p, video_name):
    lines = [f"# tuned params for: {video_name}"]
    lines.append(f"ROI_TOP_RATIO    = {p['roi_top(%)']/100.0:.2f}")
    lines.append(f"ROI_BOTTOM_RATIO = {p['roi_bot(%)']/100.0:.2f}")
    lines.append(f"# 흰색(HLS): inRange(hls, [0,{p['W_L_min']},0], [180,255,{p['W_S_max']}])")
    lines.append(f"# 노랑(HSV): inRange(hsv, [{p['Y_H_min']},{p['Y_S_min']},{p['Y_V_min']}], [{p['Y_H_max']},255,255])")
    lines.append(f"MORPH_OPEN_K  = ({p['open_k']}, {p['open_k']})")
    lines.append(f"MORPH_CLOSE_K = ({p['close_w']}, {p['close_h']})")
    text = "\n".join(lines)
    out = os.path.join(OUTPUT_DIR, "params.txt")
    with open(out, "a") as f:
        f.write(text + "\n\n")
    print("\n" + text)
    print(f"[저장] {out}\n")


def tune_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[!] 열기 실패: {path}")
        return "skip"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name = os.path.basename(path)
    print(f"[i] {name}  ({total}프레임)  space=재생 d/a=한칸 n=다음영상 s=저장 q=종료")

    cv2.namedWindow(VIEW_WIN, cv2.WINDOW_NORMAL)
    # 프레임 스크럽 트랙바(뷰 창에)
    cv2.createTrackbar("frame", VIEW_WIN, 0, max(1, total - 1), _nothing)

    playing = False
    idx = 0
    last_bar = 0

    while True:
        # 프레임 트랙바로 수동 이동 감지
        bar = cv2.getTrackbarPos("frame", VIEW_WIN)
        if bar != last_bar:
            idx = bar
        last_bar = bar

        idx = max(0, min(idx, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            playing = False
            idx = max(0, idx - 1)
            continue

        p = get_params()
        grid = render(frame, p)
        cv2.putText(grid, f"{name}  frame {idx}/{total-1}  {'PLAY' if playing else 'PAUSE'}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(VIEW_WIN, grid)

        # 재생 중이면 다음 프레임으로
        if playing:
            idx += 1
            if idx >= total:
                idx = total - 1
                playing = False
            cv2.setTrackbarPos("frame", VIEW_WIN, idx)
            last_bar = idx

        key = cv2.waitKey(20 if playing else 30) & 0xFF
        if key in (ord('q'), 27):
            cap.release()
            cv2.destroyWindow(VIEW_WIN)
            return "quit"
        elif key == ord('n'):
            cap.release()
            cv2.destroyWindow(VIEW_WIN)
            return "next"
        elif key == ord(' '):
            playing = not playing
        elif key == ord('d'):
            playing = False
            idx = min(total - 1, idx + 1)
            cv2.setTrackbarPos("frame", VIEW_WIN, idx); last_bar = idx
        elif key == ord('a'):
            playing = False
            idx = max(0, idx - 1)
            cv2.setTrackbarPos("frame", VIEW_WIN, idx); last_bar = idx
        elif key == ord('s'):
            save_params(p, name)
        elif key == ord('r'):
            reset_controls()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default=INPUT_DIR, help="영상 폴더 (기본 ~/Videos/lane)")
    ap.add_argument("--all",   action="store_true", help="폴더 안 모든 영상 순회")
    ap.add_argument("--video", type=str, default=None, help="특정 영상 하나")
    args = ap.parse_args()

    if args.video:
        videos = [os.path.expanduser(args.video)]
    else:
        indir = os.path.expanduser(args.indir)
        videos = sorted(glob.glob(os.path.join(indir, "*.mp4")) +
                        glob.glob(os.path.join(indir, "*.avi")))
        if not videos:
            print(f"[!] {indir} 에 영상이 없음")
            return
        if not args.all:
            videos = videos[:1]   # 지정 없으면 첫 영상만

    build_controls()

    for v in videos:
        res = tune_video(v)
        if res == "quit":
            break

    cv2.destroyAllWindows()
    print("[i] 종료. 저장한 값은 ~/Pictures/kmued/params.txt")


if __name__ == "__main__":
    main()