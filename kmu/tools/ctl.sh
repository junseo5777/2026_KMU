#!/bin/bash

DEV=/dev/video0

# 화이트밸런스 자동
v4l2-ctl -d $DEV -c white_balance_automatic=1

# 노출 수동
v4l2-ctl -d $DEV -c auto_exposure=1
v4l2-ctl -d $DEV -c exposure_time_absolute=100

# 채도
v4l2-ctl -d $DEV -c saturation=90

echo "[OK] 화이트밸런스 자동 / 노출 수동 설정 완료"

v4l2-ctl -d $DEV --list-ctrls | \
grep -E "white_balance|auto_exposure|exposure_time|saturation"
