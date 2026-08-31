# -*- coding: utf-8 -*-
"""
USB 카메라 → 차선(lane) 데이터 영상 녹화 (Ubuntu 22.04)
====================================================
전략: YUYV 무손실 캡처 → mp4(H.264 고화질) 녹화 → 나중에 extract_frames.py 로 프레임 추출.
      고화질 녹화 시 색검출/YOLO 정확도 차이는 미미하고, 원본 보관·재추출·낮은 부하의
      이점을 얻는다.

[사용 카메라]
  Bus 001 Device 002: ID 32e4:9230 HD USB Camera
  → /dev/videoN 으로 잡히는 인덱스를 --cam 으로 지정 (기본 0)
    확인:  v4l2-ctl --list-devices

특징
- YUYV 기본(무손실 캡처). 색공간(HSV/HLS) 검출에 유리. --mjpg 로 전환 가능.
- H.264(avc1) 우선 인코딩, 안 되면 mp4v 로 자동 폴백.
- 강제 종료(창 닫기/Ctrl+C)에도 try/finally 로 파일 손상 방지.
- 카메라 워밍업 후 fps 실측 → 영상 길이 정확. 종료 시 실제 fps 검증.

[설치]
  pip install opencv-python

[폴더]  ~/Videos/lane/<날짜시각>.mp4     (--outdir 로 변경 가능)

[실행]
  python3 capture_lane.py --cam 0
  python3 capture_lane.py --cam 0 --fps 30       # fps 고정(측정 생략, 권장)
  python3 capture_lane.py --cam 0 --mjpg         # YUYV 대신 MJPG

[종료]  q 또는 ESC  (강제 종료해도 파일은 안전하게 닫힘)

[카메라 설정]  촬영 전/직후 cam_setup.sh 로 노출·WB·초점 등을 수동 고정할 것.

[전체]
cd ~/opencv_pkg
source opencv_venv/bin/activate
./cam_setup.sh
python3 capture_lane.py --cam 0 --fps 30

"""

import cv2
import time
import argparse
import signal
from datetime import datetime
from pathlib import Path


def setup_camera(index, width, height, req_fps, use_mjpg):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    # ★ FOURCC(포맷)를 해상도보다 먼저 설정해야 적용됨 ★
    if use_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    else:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          req_fps)
    try:
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)
    except Exception:
        pass
    return cap


def fourcc_str(v):
    v = int(v)
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4))


def make_writer(path, rec_fps, size):
    """H.264(avc1/mp4v) 순으로 시도. 무손실 캡처라도 컨테이너는 mp4."""
    for tag in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*tag), rec_fps, size)
        if w.isOpened():
            return w, tag
        w.release()
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam",    type=int, default=0, help="카메라 인덱스(/dev/videoN)")
    ap.add_argument("--outdir", type=str, default=str(Path.home() / "Videos" / "lane"),
                    help="영상 저장 폴더 (기본 ~/Videos/lane)")
    ap.add_argument("--width",  type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps",    type=float, default=0,
                    help="기록 fps. 0이면 워밍업 후 실측. 카메라가 30 보고하면 --fps 30 권장")
    ap.add_argument("--mjpg",   action="store_true", help="YUYV 대신 MJPG로 캡처")
    args = ap.parse_args()

    videos_dir = Path(args.outdir).expanduser()
    videos_dir.mkdir(parents=True, exist_ok=True)

    use_mjpg = args.mjpg
    cap = setup_camera(args.cam, args.width, args.height, args.fps or 30, use_mjpg)
    if not cap.isOpened():
        print(f"[!] 카메라 {args.cam} 열기 실패. v4l2-ctl --list-devices 로 확인.")
        return

    aw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fmt = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[i] 요청 {args.width}x{args.height} {'MJPG' if use_mjpg else 'YUYV'} / "
          f"실제 {aw}x{ah} {fmt} 보고fps={cam_fps:.1f}")
    if (aw, ah) != (args.width, args.height):
        print(f"[!] 해상도 불일치. v4l2-ctl -d /dev/video{args.cam} --list-formats-ext 확인.")

    # ── 워밍업(첫 프레임 지연 제거) ──
    print("[i] 워밍업...")
    for _ in range(30):
        cap.read()

    # ── 기록 fps 결정 ──
    if args.fps and args.fps > 0:
        rec_fps = float(args.fps)
        print(f"[i] 기록 fps = {rec_fps:.1f} (지정값)")
    else:
        print("[i] 캡처 속도 측정 중...(약 2초)")
        n, t0 = 0, time.time()
        while time.time() - t0 < 2.0:
            if cap.read()[0]:
                n += 1
        measured = n / (time.time() - t0)
        if measured < 5:
            rec_fps = cam_fps if cam_fps >= 5 else 30.0
            print(f"[!] 실측 {measured:.1f} 비정상 → {rec_fps:.1f} 사용")
        else:
            rec_fps = measured
            print(f"[i] 기록 fps = {rec_fps:.1f} (실측)")

    # ── writer (H.264 우선) ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    vpath = videos_dir / f"{ts}.mp4"
    out, codec = make_writer(vpath, rec_fps, (aw, ah))
    if out is None:
        print("[!] VideoWriter 실패 — H.264/mp4v 모두 불가. opencv 코덱 확인 필요.")
        cap.release()
        return
    print(f"[i] 녹화 시작 ({codec}) → {vpath}   (q/ESC=종료)")

    # ── 종료 시그널 안전 처리 ──
    stop = {"flag": False}
    signal.signal(signal.SIGINT,  lambda *a: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))

    written = 0
    disp_fps = 0.0
    fps_cnt = 0
    last_t = time.time()
    t_start = time.time()
    try:
        while not stop["flag"]:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if (frame.shape[1], frame.shape[0]) != (aw, ah):
                frame = cv2.resize(frame, (aw, ah))

            out.write(frame)
            written += 1

            fps_cnt += 1
            now = time.time()
            if now - last_t >= 1.0:
                disp_fps = fps_cnt / (now - last_t)
                fps_cnt = 0
                last_t = now

            view = frame.copy()
            cv2.putText(view, f"REC {disp_fps:.1f}fps  {fmt}  (q/ESC=stop)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("lane record (q=quit)", view)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        elapsed = time.time() - t_start
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        real = written / elapsed if elapsed > 0 else 0
        print(f"[OK] {written}프레임 / {elapsed:.1f}초 (실제 {real:.1f}fps) → {vpath}")
        if abs(real - rec_fps) > 3:
            print(f"[!] 기록fps({rec_fps:.0f})와 실제({real:.1f}) 차이 큼 → 다음엔 --fps {real:.0f}")
        print("[i] 프레임 추출:  python3 extract_frames.py --all --every 10")


if __name__ == "__main__":
    main()