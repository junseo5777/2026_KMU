"""
shortcut_cam.py — 지름길(갈색 나무바닥) 판정. **탈출 시점 검출이 목적이다.**

    camera_node 에서 이렇게 쓴다 (BEV 를 넘긴다):

        from .shortcut_cam import ShortcutCamGate
        self.scut = ShortcutCamGate(logger=self.get_logger())    # 노드 생성 시 1회
        ...
        pre = pp.process(bgr)
        s = self.scut.update(pre['bev'])
        out.shortcut_detected = bool(s['on'])

★ state 는 shortcut_detected 의 **True→False 하강 에지**를 좌회전 보조의
  트리거로 쓴다. 따라서 이 모듈의 성능 지표는 '구간을 몇 % 켜고 있나' 가 아니라
  **'하강 에지가 실제 탈출과 1:1 대응하나'** 다. 구간 내부에서 한 번이라도
  꺼지면 잘못된 조향이 나가 차선을 벗어난다.

── v1 이 실패한 이유 (S 하한이 범인) ────────────────────────────
초판(S 75~130, V 100~165, 비율 25%, OFF 15f)은 구간 내부에서 최장 61프레임까지
꺼졌다. LATCH_OFF 15 로는 원리적으로 못 막는 길이다.
원인은 V 가 아니라 **S** 였다. 실측(구간 내 raw 성공 1629f vs 실패 381f):

                        성공 중앙   실패 중앙
        중앙 S             92        68      ← 실패의 70% 가 S<75
        중앙 V            128       126      ← V 상한 초과는 0%
        S<75 & V>100 픽셀  13.7%     49.7%   ← 이것이 실체
        H 10~24 픽셀       87.6%     79.3%   ← 색상은 살아있다

    조명 빛반사가 **채도만 씻어낸다**. 밝기와 색상은 유지된다. 즉 반사된
    바닥 픽셀은 '흰색에 가까운 갈색' 이 되어 S 하한에 걸려 통째로 탈락했다.
    → S 하한을 20 으로 내려 반사분까지 포괄하고, 대신 비율 임계를 25→45 로
      올려 균형을 맞춘다.

── 왜 슈미트 트리거인가 ────────────────────────────────────────
단일 임계 + 시간 래치는 '드롭아웃을 못 막거나(OFF 짧음) / 탈출이 늦거나(OFF 김)'
둘 중 하나를 고를 수밖에 없다. 비율 자체에 히스테리시스를 걸면 둘 다 해결된다:
    ON  : 비율 >= RATIO_ON(45)  이 LATCH_ON(8) 프레임 연속
    OFF : 비율 <  RATIO_OFF(24) 가 LATCH_OFF(5) 프레임 연속
중간(24~45)에서 흔들리는 것은 상태를 안 바꾼다. 그래서 OFF 확인을 5프레임까지
줄여도 드롭이 안 나고, 탈출 지연이 +19f → **+8f** 로 줄었다.

── MIN_ON_FRAMES — 짧은 깜빡임은 '탈출' 이 아니다 ──────────────
구간 밖에서 잠깐 켜졌다 꺼지면 그것도 하강 에지다. 그래서 **ON 이 충분히
지속된 뒤에만 on 을 True 로 올린다.** 짧은 깜빡임은 아예 on 을 못 올리므로
하강 에지 자체가 생기지 않는다.
    실제 ON 지속 최단 204프레임 vs MIN_ON 80 → 여유 2.5배.
    (구간을 2배 빠르게 통과해도 102f > 80f 로 안전)

── 확정 동작점 검증 (lane_6/7/8/9/10, 26,103프레임 · 실제 탈출 9회) ──
                              정답 에지   허위 에지   구간내 드롭
        v1 (S75~130 th25)        9         5        9건 / 최장 61f
        현재                     9         0        1건 / 최장  1f
    탈출 에지 지연  중앙 +8프레임 (+0.27s)

    조명 내성 — 채도 x0.85~1.15 × 밝기 x0.85~1.15 **15칸 전부 9/0**.
    (v1 은 밝기 ±10% 만 흔들려도 허위 에지가 났다)

── BEV vs 원본 ROI ─────────────────────────────────────────────
같은 게이트로 원본 ROI(0.70~0.93)와 비교한 결과 **탈출 에지 성적은 동일**
(둘 다 9/0, 지연 +8f). 구간 내 최장 드롭만 BEV 1f / 원본 3f 로 BEV 가 낫다.
BEV 는 pp.process 가 이미 계산해 둔 값이라 워프 비용이 0 이므로 BEV 를 쓴다.

대회장 조명이 다르면 만질 순서:
    구간 안에서 꺼진다      → S_MIN 을 20→10, 그다음 RATIO_ON 을 45→38
    구간 밖에서 켜진다      → RATIO_ON 을 45→55 (S 는 건드리지 말 것)
    탈출이 늦다             → RATIO_OFF 를 24→30, LATCH_OFF 를 5→3
"""

import cv2
import numpy as np


# ── 갈색 나무바닥 게이트 ──
#   S 하한 20 은 빛반사로 채도가 씻긴 바닥을 포괄하기 위한 값이다 (docstring 참조).
#   상한 150 은 라바콘 게이트(S>=200)와 겹치지 않도록 유지한다.
H_MIN, H_MAX = 10, 24
S_MIN, S_MAX = 20, 150
V_MIN, V_MAX = 88, 185

# ── 슈미트 트리거 (비율 히스테리시스) ──
RATIO_ON = 45.0              # 이 이상이 LATCH_ON 연속 → 내부 ON
RATIO_OFF = 24.0             # 이 미만이 LATCH_OFF 연속 → OFF (= 탈출)
LATCH_ON = 8
LATCH_OFF = 5

# ── 탈출로 인정할 최소 지속 ──
#   이 프레임 수를 못 채운 ON 은 바깥으로 True 를 내보내지 않는다 →
#   하강 에지가 생기지 않는다.
MIN_ON_FRAMES = 80


class ShortcutCamGate:
    """지름길 갈색 바닥 판정. 프레임당 update(bev) 한 번."""

    def __init__(self, logger=None):
        self.log = logger
        self.on = False          # 바깥에 내보내는 값 (MIN_ON 을 채운 뒤에만 True)
        self._st = False         # 내부 슈미트 상태
        self._on = 0
        self._off = 0
        self._dur = 0
        self._lo = np.array([H_MIN, S_MIN, V_MIN], np.uint8)
        self._hi = np.array([H_MAX, S_MAX, V_MAX], np.uint8)

    def update(self, bev):
        """returns dict(on, exit, ratio, raw_on, dur)

        bev    : pp.process(bgr)['bev'] — 컬러 BEV (BGR). 원본 프레임이 아니다.
        on     : CamData.shortcut_detected 로 보낼 값. 이 값의 True→False 가 탈출.
        exit   : 이번 프레임에 자격 있는 탈출이 일어났으면 True (1프레임 펄스).
                 on 의 하강 에지와 동일하다 — 둘 중 편한 쪽을 쓰면 된다.
        ratio  : BEV 내 갈색 비율 % (튜닝·로그용)
        raw_on : MIN_ON 을 아직 못 채운 내부 상태 (진단용)
        dur    : 현재 ON 이 지속된 프레임 수
        """
        if bev is None or bev.size == 0:
            return self._out(0.0, False)

        mask = cv2.inRange(cv2.cvtColor(bev, cv2.COLOR_BGR2HSV), self._lo, self._hi)
        ratio = 100.0 * float(mask.mean()) / 255.0

        self._on = self._on + 1 if ratio >= RATIO_ON else 0
        self._off = self._off + 1 if ratio < RATIO_OFF else 0

        exited = False
        if not self._st and self._on >= LATCH_ON:
            self._st = True
            self._dur = 0
        elif self._st and self._off >= LATCH_OFF:
            self._st = False
            if self.on:                       # 자격을 갖춘 ON 이었을 때만 탈출
                exited = True
                self.on = False
                self._say(f"지름길 탈출 (지속 {self._dur}f)")
            self._dur = 0

        if self._st:
            self._dur += 1
            if not self.on and self._dur >= MIN_ON_FRAMES:
                self.on = True
                self._say(f"지름길 확정 (갈색 {ratio:.0f}%)")

        return self._out(ratio, exited)

    def reset(self):
        self.on = self._st = False
        self._on = self._off = self._dur = 0

    # ──
    def _out(self, ratio, exited):
        return {'on': self.on, 'exit': exited, 'ratio': ratio,
                'raw_on': self._st, 'dur': self._dur}

    def _say(self, m):
        if self.log is not None:
            self.log.info(f"[shortcut] {m}")