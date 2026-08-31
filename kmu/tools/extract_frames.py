# -*- coding: utf-8 -*-
"""
영상 프레임 추출 (Ubuntu 22.04)
====================================================
~/Videos/lane 의 mp4 영상들을 훑어 --every N 간격으로 프레임을 뽑아
~/Videos/lane_frame 에 이미지로 저장한다. (차선/신호등 데이터셋용)

특징
- --all 로 폴더 전체 일괄 처리, 또는 --video 로 특정 파일 하나만 처리.
- --every N : N프레임마다 1장 저장(기본 10). --every 1 이면 전 프레임.
- 파일명 = <영상이름>_<프레임번호6자리>.jpg  → 여러 영상 섞여도 안 겹침.
- --subdir : 영상별 하위폴더로 분리 저장(lane_frame/<영상이름>/...).
- jpg(기본, --quality) / png 선택 가능.

[설치]
  pip install opencv-python

[실행]
  python3 extract_frames.py --all --every 10
  python3 extract_frames.py --video ~/Videos/lane/20260710_143022.mp4 --every 5
  python3 extract_frames.py --all --every 10 --subdir      # 영상별 폴더 분리
  python3 extract_frames.py --all --every 10 --ext png     # png로 저장
"""

import cv2
import argparse
from pathlib import Path


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def list_videos(indir):
    return sorted(p for p in indir.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def extract_one(vpath, outdir, every, ext, quality, use_subdir):
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"[!] 열기 실패: {vpath}")
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    stem  = vpath.stem

    # 저장 폴더 결정
    dst = outdir / stem if use_subdir else outdir
    dst.mkdir(parents=True, exist_ok=True)

    print(f"[i] {vpath.name}  (총 {total}프레임 / {fps:.1f}fps) → 매 {every}프레임 저장")

    # 저장 이미지 인코딩 파라미터
    if ext == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif ext == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        params = []

    idx = 0     # 읽은 프레임 번호
    saved = 0   # 저장한 장수
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every == 0:
            # 하위폴더 안 쓸 땐 영상이름 prefix로 충돌 방지
            name = f"{idx:06d}.{ext}" if use_subdir else f"{stem}_{idx:06d}.{ext}"
            cv2.imwrite(str(dst / name), frame, params)
            saved += 1
        idx += 1

    cap.release()
    print(f"[OK] {vpath.name}: {saved}장 저장 → {dst}")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir",  type=str, default=str(Path.home() / "Videos" / "lane"),
                    help="영상 폴더 (기본 ~/Videos/lane)")
    ap.add_argument("--outdir", type=str, default=str(Path.home() / "Videos" / "lane_frame"),
                    help="이미지 저장 폴더 (기본 ~/Videos/lane_frame)")
    ap.add_argument("--all",    action="store_true", help="indir 안의 모든 영상 처리")
    ap.add_argument("--video",  type=str, default=None, help="특정 영상 파일 하나만 처리")
    ap.add_argument("--every",  type=int, default=10, help="N프레임마다 1장 저장 (기본 10)")
    ap.add_argument("--ext",    type=str, default="jpg", choices=["jpg", "png"],
                    help="저장 이미지 포맷 (기본 jpg)")
    ap.add_argument("--quality", type=int, default=95, help="jpg 품질 0~100 (기본 95)")
    ap.add_argument("--subdir", action="store_true", help="영상별 하위폴더로 분리 저장")
    args = ap.parse_args()

    if args.every < 1:
        print("[!] --every 는 1 이상이어야 함")
        return

    indir  = Path(args.indir).expanduser()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    # 처리 대상 영상 목록 만들기
    if args.video:
        vpath = Path(args.video).expanduser()
        if not vpath.exists():
            print(f"[!] 영상 없음: {vpath}")
            return
        videos = [vpath]
    elif args.all:
        if not indir.exists():
            print(f"[!] 폴더 없음: {indir}")
            return
        videos = list_videos(indir)
        if not videos:
            print(f"[!] {indir} 에 영상 파일이 없음")
            return
    else:
        print("[!] --all (폴더 전체) 또는 --video <파일> 중 하나를 지정하세요.")
        return

    print(f"[i] 대상 {len(videos)}개 영상 / --every {args.every} / 저장 {outdir}")
    total_saved = 0
    for v in videos:
        total_saved += extract_one(v, outdir, args.every, args.ext,
                                   args.quality, args.subdir)

    print(f"[완료] 총 {total_saved}장 저장 → {outdir}")


if __name__ == "__main__":
    main()