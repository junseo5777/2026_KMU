"""
checkerboard_cam.py — 바닥 체커보드 구간 검출 (원본 프레임 기반)

    camera_node 에서 이렇게 쓴다 (원본 BGR 을 넘긴다):

        from .checkerboard_cam import CheckerCamGate
        self.ckb = CheckerCamGate(logger=self.get_logger())     # 노드 생성 시 1회
        ...
        c = self.ckb.update(bgr)
        out.checkerboard_detected = bool(c['on'])

용도는 **차선 마스크를 고치는 것이 아니라 조향을 묶는 것**이다. state 에서
on 인 동안 max_steer 를 작은 값으로 클램프하면, 체커보드가 fit 을 오염시켜도
차가 물리적으로 크게 꺾이지 못해 옆 구조물(랩카운트 센서·신호등 기둥)과
충돌하지 않는다.

── 왜 BEV 가 아니라 원본인가 ────────────────────────────────────
lane BEV 는 ROI 0.70~0.93 의 얇은 띠라 체커보드가 1~2줄만 담긴다. 격자 구조가
안 보이면 아래 '교차 밀도' 가 성립하지 않는다. 원본 하단이면 격자 전체가 들어온다.

── 왜 형상·색상이 아니라 '명암 교차 밀도' 인가 ────────────────────
★ 형상 게이트는 쓸 수 없다 (실측으로 기각) ★
    BEV 성분의 bh/bw 중앙값:  체커보드 0.53  vs  **S자 차선 0.48**
    둘이 같다. bh/bw < 0.7 로 거르면 체커보드는 80% 지워지지만
    **S자 차선도 88% 가 함께 지워진다.** 형상으로는 원리적으로 못 가른다.

    색상도 안 된다 — 체커보드는 흑/백이라 노면·차선과 색 대비가 없다.

남는 것은 **구조**다. 체커보드만의 성질은 '밝은 칸 바로 옆에 어두운 칸이
있다' 이고, 차선 옆에는 노면(중간 밝기)이 있다. 그래서:

    지면 ROI 를 320px 폭으로 축소 → 회색조 → 노면 중앙값 med 산출
    brt = g > med+40,  drk = g < med-40          (조명에 둔감한 상대 임계)
    교차 = (brt 이면서 drk 에 인접) 또는 (drk 이면서 brt 에 인접)   커널 9x9
    score = 교차 픽셀 비율 %

── 실측 (lane_10, 8,616프레임) ─────────────────────────────────
    score 중앙 0.82   p95 6.26   p99 10.02
    상위 프레임 육안 검증:
        score >= 13   전부 진짜 체커보드
        score 8.7~9.0 콘 보관대 오탐 (임계 12 아래라 안 걸림)

    임계 12 · 3프레임 이상 · 60f 이내 병합 → **8개 구간**
        f730~737  f1860~1864  f2741~2748  f4054~4057
        f4939~4944  f5785~5788  f6974~6985  f8498~8509
    코스에 체커보드가 2곳이므로 4랩 분량과 일치한다.

── ★ 한계 — 반드시 알고 쓸 것 ★ ────────────────────────────────
1) 정면의 체커보드는 잡지만, **화면 가장자리로 비껴 있는 것은 약하다.**
   신호등 옆(2차선 쪽) 체커보드가 그 경우다 (f2831 에서 score 5.4 로 미달).
   → HOLD_FRAMES 로 검출 후 일정 시간 플래그를 유지해 통과 구간을 덮는다.
2) 검출 구간 자체가 짧다(4~12프레임). 차가 실제로 체커보드 위를 지나는 것은
   검출 시점보다 뒤다. HOLD_FRAMES 는 그 시차를 메우는 값이지 여유가 아니다.
3) 라벨이 lane_10 한 편뿐이다. 대회장에서 score 로그를 먼저 확인하고
   RATIO_ON 을 조정할 것 — 아래 '조정 순서' 참조.

조정 순서:
    체커보드를 놓친다   → RATIO_ON 12 → 10 (콘 보관대 오탐이 9 근처이니 10 이 하한)
    엉뚱한 곳에서 뜬다   → RATIO_ON 12 → 14
    통과 중에 꺼진다     → HOLD_FRAMES 상향
"""

import cv2
import numpy as np


# ── 지면 ROI (원본 프레임 세로 비율) ──
ROI_TOP, ROI_BOT = 0.55, 0.95

# ── 축소 폭 (세로는 비율 유지) ──
DS_W = 320

# ── 명암 분리 (노면 중앙값 기준 상대값 — 조명에 둔감) ──
DELTA = 40
NEIGH_K = 9                  # 인접 판정 커널. 체커보드 칸 피치에 맞춘 값

# ── 프레임 판정 (슈미트) ──
RATIO_ON = 12.0              # 이 이상이면 체커보드 (육안 검증: 13+ 는 전부 진짜)
RATIO_OFF = 7.0              # 이 미만이면 해제

LATCH_ON = 2                 # 구간이 4~12프레임뿐이라 짧게 잡는다
HOLD_FRAMES = 45             # 검출 후 플래그 유지 (실제 통과 시점을 덮기 위함)


class CheckerCamGate:
    """바닥 체커보드 구간 검출. 프레임당 update(bgr) 한 번."""

    def __init__(self, logger=None):
        self.log = logger
        self.on = False
        self._on = 0
        self._hold = 0
        self._k = cv2.getStructuringElement(cv2.MORPH_RECT, (NEIGH_K, NEIGH_K))

    def update(self, bgr):
        """returns dict(on, ratio, raw, hold)

        on    : CamData.checkerboard_detected 로 보낼 값.
                state 는 이 값이 True 인 동안 max_steer 를 클램프한다.
        ratio : 명암 교차 밀도 % (튜닝·로그용 — 대회장에서 이 값을 먼저 볼 것)
        raw   : 이번 프레임 단독 판정
        hold  : 남은 유지 프레임
        """
        if bgr is None or bgr.size == 0:
            return self._out(0.0, False)

        h0 = bgr.shape[0]
        roi = bgr[int(h0 * ROI_TOP):int(h0 * ROI_BOT)]
        if roi.size == 0:
            return self._out(0.0, False)

        hh = max(8, int(DS_W * roi.shape[0] / roi.shape[1]))
        g = cv2.cvtColor(cv2.resize(roi, (DS_W, hh), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        med = float(np.median(g))
        brt = (g > med + DELTA).astype(np.uint8)
        drk = (g < med - DELTA).astype(np.uint8)
        cross = ((brt & cv2.dilate(drk, self._k))
                 | (drk & cv2.dilate(brt, self._k)))
        ratio = float(cross.mean()) * 100.0

        raw = ratio >= RATIO_ON
        self._on = self._on + 1 if raw else 0

        if self._on >= LATCH_ON:
            if not self.on:
                self._say(f"체커보드 구간 (교차밀도 {ratio:.1f}%)")
            self.on = True
            self._hold = HOLD_FRAMES
        elif self.on:
            # 해제는 RATIO_OFF 미만일 때만 카운트다운 (중간값에서는 유지)
            if ratio < RATIO_OFF:
                self._hold -= 1
                if self._hold <= 0:
                    self.on = False
                    self._say("체커보드 구간 종료")
            else:
                self._hold = HOLD_FRAMES

        return self._out(ratio, raw)

    def reset(self):
        self.on = False
        self._on = self._hold = 0

    # ──
    def _out(self, ratio, raw):
        return {'on': self.on, 'ratio': ratio, 'raw': raw, 'hold': self._hold}

    def _say(self, m):
        if self.log is not None:
            self.log.info(f"[checker] {m}")