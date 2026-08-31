#!/usr/bin/env python3
"""
lane_detection4.py — 1차로 강조향 부스트 변형

lane_detection3 을 상속한다. 검출·모드 판별·게인·평활·속도·출력 규약을 전부
그대로 물려받고, 딱 한 지점만 바꾼다:

    current_lane == 1 이고 이번 프레임에 계산된 조향이 -20 보다 작으면(= 좌측으로
    세게 꺾는 상황) 그 조향에 일괄 1.1 배를 곱한다.

즉 검출은 손대지 않았으므로 L3 대비 차이는 전적으로 '1차로 강조향 구간의 출력
이득' 하나에서 온다.

── 부호 규약 (헷갈리기 쉬움) ──────────────────────────────
  lane_detection3.compute_steering: lat_err = center_near - cx 이므로
  steer > 0 = 우조향, steer < 0 = 좌조향 (planner/params.py:47 과 동일 규약).
  observed_lane: 노란 중앙선이 차량 우측이면 1차로.
  → '1차로 + steer < -20' 은 노란 중앙선을 오른쪽에 두고 좌측으로 크게 꺾는
    상황, 즉 1차로(안쪽 차로)에서의 좌커브다. 안쪽 차로는 같은 코너를 더 작은
    반경으로 돌아야 하므로 L3 게인이 부족해질 수 있는 구간이 여기다.
  ★ 우조향(+) 쪽과 2차로에는 아무 영향이 없다. 완전히 비대칭인 변경이다.

── 적용 지점: compute_steering 반환 직후 ──────────────────
  process() 의 파이프라인은
      compute_steering → (straight 데드존) → _limit_rate → clip(±max_steer)
  이고, 부스트는 맨 앞 단계의 반환값에 곱한다. 따라서 뒤따르는 안전장치가
  전부 그대로 살아 있다:
    · clip(±max_steer)  — 모드 상한을 넘겨 나가지 않는다.
    · _limit_rate       — 변화율 제한을 우회하지 않는다.
    · steer_hist        — 이동평균 창에는 부스트 '이전' 값이 쌓인다(기저 클래스가
                          _smoothed 직전에 append 하므로). 부스트가 평활에
                          재귀적으로 누적되지 않는다 — 매 프레임 1.1 배 그대로다.

  모드별 실효 구간 (부스트가 실제로 값을 바꾸는 범위):
      straight  max_steer 9.0  → |steer| 이 20 에 도달 불가. 부스트 미발동.
      entry     max_steer 30.0 → -20.0 ~ -27.3 구간만 값이 바뀌고,
                                 -27.3 아래는 1.1 배 해도 -30 으로 잘린다.
      curve     max_steer 50.0 → -20.0 ~ -45.5 구간, 그 아래는 -50 으로 잘림.
  즉 실제 효과는 'entry/curve 에서 상한에 아직 안 붙은 강조향을 상한 쪽으로
  더 빨리 밀어붙이는 것'이다. 이미 포화된 프레임은 아무 변화가 없다.

── 감쇠(blind) 프레임은 제외한다 ──────────────────────────
  ★ 이 예외가 없으면 차선 소실 시 조향이 안 풀린다 ★
  기저 compute_steering 은 쓸 fit 이 없을 때 새 지령을 만들지 않고
  prev_steer * 0.9 (프레임당 10% 감쇠) 를 반환한다. 여기에까지 1.1 을 곱하면
  0.9 * 1.1 = 0.99 — 감쇠가 프레임당 1% 로 죽어 사실상 '강한 좌조향 유지'가
  된다. 차선을 놓친 상태에서 -40 조향을 계속 물고 있는 것이 정확히 이탈 시나리오라,
  부스트는 '측정 근거가 있는 프레임'에서만 건다:
      단일선 : n_real == 1
      양선   : n_real == 2 이고 좌/우 fit 이 둘 다 살아 있음
  이 둘이 기저 클래스가 실제로 lat_err/heading 을 계산하는 분기와 정확히 일치한다.
  (LANE1_BOOST_REQUIRE_REAL = False 로 두면 감쇠 프레임까지 포함시킬 수 있으나
   위 이유로 권장하지 않는다.)

── 알려진 성질: 임계에서의 불연속 ─────────────────────────
  일괄 배수라 -19.9 → -19.9, -20.1 → -22.1 로 임계에서 2.0 만큼 계단이 생긴다.
  조향이 -20 근처를 오가면 이 계단이 그대로 출력에 실린다(_limit_rate 의
  프레임당 허용치는 30fps 에서 base 10.0 / curve 16.5 라 계단을 막지 못한다).
  실차에서 채터링이 보이면 소프트 니(knee)로 바꾸는 것이 한 줄이다:
      g = 1.0 + (GAIN-1.0) * min(1.0, (abs(steer)-abs(THRESH)) / KNEE_PX)
  다만 요구가 '일괄 1.1 배' 이므로 기본 구현은 계단 그대로 둔다.

사용: import 한 줄만 바꾸면 된다.
  from .lane_detection3 import LaneDetector   →   from .lane_detection4 import LaneDetector
"""

# 패키지(camera_node.py) / 단일 디렉토리 스크립트(debug_tuner.py, camera_test_node.py)
# 양쪽에서 import 되므로 상대·절대를 모두 받는다.
try:
    from .lane_detection3 import LaneDetector as _Base, MARGIN_SEARCH
except ImportError:                                    # pragma: no cover
    from lane_detection3 import LaneDetector as _Base, MARGIN_SEARCH

__all__ = ['LaneDetector', 'MARGIN_SEARCH',
           'LANE1_BOOST_LANE', 'LANE1_BOOST_THRESH', 'LANE1_BOOST_GAIN',
           'LANE1_BOOST_REQUIRE_REAL']


# ════════════════════════════════════════════════
#  1차로 강조향 부스트 파라미터
# ════════════════════════════════════════════════
LANE1_BOOST_LANE   = 1        # 이 차로에서만 적용 (current_lane 확정값)
LANE1_BOOST_THRESH = -20.0    # 이 값보다 '작은'(= 더 센 좌조향) 지령이 대상
LANE1_BOOST_GAIN   = 1.2      # 일괄 배수
LANE1_BOOST_REQUIRE_REAL = True   # 감쇠(blind) 프레임 제외 — 상단 주석 참조


class LaneDetector(_Base):
    """제어 출력만 바꾼 변형. 검출·출력 규약은 lane_detection3 과 100% 동일."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lane1_boost = 0    # 부스트가 실제로 발동한 누적 프레임 수 (진단용)

    # ------------------------------------------------
    #  조향 계산 → 1차로 강조향 부스트
    # ------------------------------------------------
    def compute_steering(self, left_fit, right_fit, real_detected, bev_shape,
                         curvature, kp, k_head, max_steer, lookahead, dt):
        steer = super().compute_steering(
            left_fit, right_fit, real_detected, bev_shape, curvature,
            kp=kp, k_head=k_head, max_steer=max_steer,
            lookahead=lookahead, dt=dt)

        if self.current_lane != LANE1_BOOST_LANE or steer >= LANE1_BOOST_THRESH:
            return steer

        if LANE1_BOOST_REQUIRE_REAL:
            # 기저 클래스가 실제로 lat_err/heading 을 계산한 분기인지 확인.
            # 나머지 경로의 반환값은 prev_steer 감쇠라 부스트 대상이 아니다.
            n_real = int(real_detected[0]) + int(real_detected[1])
            measured = (n_real == 1 or
                        (n_real == 2 and left_fit is not None and right_fit is not None))
            if not measured:
                return steer

        self._lane1_boost += 1
        # 상한은 process() 말미의 clip(±max_steer) 이 잡는다. 여기서 미리 자르면
        # _limit_rate 이전/이후 클립이 이중으로 걸려 기저 파이프라인과 어긋난다.
        return float(steer * LANE1_BOOST_GAIN)

    # ------------------------------------------------
    #  진단 카운터 노출
    # ------------------------------------------------
    def process(self, *args, **kwargs):
        out = super().process(*args, **kwargs)
        out['lane1_boost'] = self._lane1_boost   # 1차로 강조향 부스트 발동 누적
        return out
