#!/usr/bin/env python3
"""
steer_calib.py — 조향 좌우 실한계 실측 노드 (/xycar_motor 조향각만 발행)

목적
  실차에서 (1) 직진 트림 b, (2) 좌측 실한계 lim_l, (3) 우측 실한계 lim_r 를 재서
  track_drive 의 조향 지령 → 모터 매핑을 만든다.

  현재는 publish 직전에 `steer - 8.0` 오프셋을 빼는 방식인데, 이러면 좌측 지령이
  모터 범위를 넘어가 그대로 잘려 좌측 조향 권한이 손실된다. 예를 들어
  모터 실범위가 [-45, +60] 이고 b=-8 이면:

      컨트롤러 -60  →  -68  →  클립 -45   (23 만큼 손실)
      컨트롤러 +60  →  +52  →       +52   ( 8 만큼 손실)

  양측을 각각 스케일하면 두 끝이 정확히 모터 한계에 닿는다 (아래 'p' 출력 참조).

안전
  - 구동 속도는 기본 0.0 (제자리에서 조향만). 차량을 들어올리거나 스탠드에 올린
    상태에서 측정할 것.
  - 종료(q / Ctrl-C) 시 angle=0, speed=0 을 여러 번 발행하고 나간다.
  - CMD_ABS_LIMIT 로 발행 절대값을 제한한다 (기본 100).

발행 경로 (mode 파라미터)
  motor (기본) : /xycar_motor 로 Float32MultiArray [steer, speed] 직접 발행.
                 camera_test_node.py / my_motor/go.py 와 동일한 구동부 인터페이스다.
                 total.py 를 거치지 않으므로 거기 박혀 있는 `steer - 8.0` 오프셋이
                 섞이지 않는다 — 캘리브레이션은 이 경로로 재야 한다.
  erp          : track_drive.py 처럼 /erp_writes 로 Twist(linear.x=speed,
                 angular.z=angle) 발행. total.py 가 AUTO 모드일 때만 중계하고
                 -8.0 오프셋이 더해지므로, 측정값이 아니라 최종 경로 점검용이다.

  ※ motor 모드로 잴 때는 total.py 를 내리거나 MANUAL 로 두어야 한다.
    같은 /xycar_motor 에 두 노드가 동시에 쓰면 값이 번갈아 들어간다.

사용법
  ros2 run kmu steer_test
  ros2 run kmu steer_test --ros-args -p mode:=erp -p speed:=0.0
  python3 steer_calib.py --ros-args -p topic:=/xycar_motor -p speed:=0.0

측정 순서
  1) [트림]  0 부근에서 a/d 로 미세 조정하며 앞바퀴가 정확히 직진이 되는 값을 찾고
             t 를 눌러 기록한다.
  2) [좌한계] a 또는 A 로 음수 방향으로 키우다가, 더 눌러도 바퀴가 더 안 꺾이는
             지점에서 [ 를 눌러 기록한다. (s 자동 스윕을 쓰면 관찰이 쉽다)
  3) [우한계] 같은 방식으로 우측을 재고 ] 를 누른다.
  4) p 로 결과와 붙여넣을 코드를 출력한다.

키
  a / d      조향 -1 / +1          (--step)
  A / D      조향 -5 / +5          (--coarse)
  0          조향 0
  z          조향 = 기록된 트림
  s          자동 스윕 시작/정지 (0 → 좌한계 → 0 → 우한계 → 0 반복)
  t          현재 값을 '직진 트림 b' 로 기록
  [ / ]      현재 값을 '좌 한계' / '우 한계' 로 기록
  p          결과 요약 + 매핑 코드 출력
  r          기록 초기화
  q          종료 (0 발행 후)
"""

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


# 제어 명령용 QoS. total.py 가 /xycar_motor · /erp_writes 양쪽에서 쓰는 설정과
# 동일하게 맞춘다 (RELIABLE, depth 10). 여기서 어긋나면 구독이 붙지 않는다.
QOS_CMD = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

# 발행 경로별 기본 토픽
DEFAULT_TOPIC = {'motor': '/xycar_motor', 'erp': '/erp_writes'}

CMD_ABS_LIMIT = 100.0    # 발행 절대값 상한 (안전)
SWEEP_RATE    = 12.0     # 자동 스윕 속도 (단위/초) — 바퀴가 멈추는 지점을 눈으로
                         # 확인할 수 있을 만큼 느리게
SWEEP_SPAN    = 100.0    # 기록된 한계가 없을 때 스윕 진폭

# 컨트롤러 출력 범위. lane_detection3.MODE_GAINS 의 최대 max_steer (curve=100.0).
# 매핑은 이 범위를 모터 실범위로 늘려주는 역할이다.
CTRL_RANGE = 100.0


class SteerCalibNode(Node):

    def __init__(self):
        super().__init__('steer_calib')

        self.declare_parameter('mode', 'motor')   # 'motor' | 'erp'
        self.declare_parameter('topic', '')       # 비우면 mode 기본 토픽
        self.declare_parameter('speed', 0.0)
        self.declare_parameter('rate', 20.0)
        self.declare_parameter('step', 1.0)
        self.declare_parameter('coarse', 5.0)

        self.mode   = str(self.get_parameter('mode').value).lower()
        self.speed  = float(self.get_parameter('speed').value)
        rate        = float(self.get_parameter('rate').value)
        self.step   = float(self.get_parameter('step').value)
        self.coarse = float(self.get_parameter('coarse').value)

        if self.mode not in DEFAULT_TOPIC:
            self.get_logger().warn(
                f"알 수 없는 mode '{self.mode}' — 'motor' 로 대체합니다.")
            self.mode = 'motor'
        topic = self.get_parameter('topic').value or DEFAULT_TOPIC[self.mode]

        if self.mode == 'erp':
            # track_drive.drive() 와 동일: linear.x=speed, angular.z=angle
            self.pub = self.create_publisher(Twist, topic, QOS_CMD)
            self.msg = Twist()
        else:
            # camera_test_node / my_motor 와 동일: data=[steer, speed]
            self.pub = self.create_publisher(Float32MultiArray, topic, QOS_CMD)
            self.msg = Float32MultiArray()

        self.angle   = 0.0
        self.trim    = None      # 직진이 되는 지령값 b
        self.lim_l   = None      # 좌측 실한계
        self.lim_r   = None      # 우측 실한계
        self.sweep   = False
        self._sw_dir = -1.0      # 스윕 진행 방향
        self._sw_leg = 'L'       # 현재 왕복 구간 ('L' 좌측 / 'R' 우측)

        self.dt = 1.0 / rate
        self.create_timer(self.dt, self._tick)

        # 원시 터미널 (SSH 에서도 동작하도록 cv2 창 대신 stdin 사용)
        self._fd  = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

        msg_name = 'Twist' if self.mode == 'erp' else 'Float32MultiArray'
        print(__doc__)
        print("=" * 66)
        print(f"  mode={self.mode}  topic={topic} ({msg_name})")
        print(f"  speed={self.speed}  rate={rate}Hz  step={self.step}/{self.coarse}")
        if self.mode == 'erp':
            print("  ⚠ total.py 가 AUTO 모드여야 중계되고, 거기서 -8.0 이 더 빠집니다.")
        print("=" * 66)

    # ── 터미널 ──
    def restore_terminal(self):
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        except Exception:
            pass

    def _getch(self):
        """논블로킹 1문자 읽기. 없으면 None."""
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
        return None

    # ── 발행 ──
    def _publish(self, angle, speed):
        a = max(-CMD_ABS_LIMIT, min(CMD_ABS_LIMIT, float(angle)))
        if self.mode == 'erp':
            self.msg.linear.x  = float(speed)
            self.msg.angular.z = a
        else:
            self.msg.data = [a, float(speed)]
        self.pub.publish(self.msg)
        return a

    def stop(self, n=5):
        """중립 발행 (종료용)."""
        for _ in range(n):
            self._publish(0.0, 0.0)

    # ── 스윕 ──
    def _sweep_step(self):
        lo = self.lim_l if self.lim_l is not None else -SWEEP_SPAN
        hi = self.lim_r if self.lim_r is not None else SWEEP_SPAN
        self.angle += self._sw_dir * SWEEP_RATE * self.dt
        if self._sw_leg == 'L':
            if self.angle <= lo:
                self.angle = lo; self._sw_dir = 1.0
            elif self.angle >= 0.0 and self._sw_dir > 0:
                self.angle = 0.0; self._sw_leg = 'R'
        else:
            if self.angle >= hi:
                self.angle = hi; self._sw_dir = -1.0
            elif self.angle <= 0.0 and self._sw_dir < 0:
                self.angle = 0.0; self._sw_leg = 'L'; self._sw_dir = -1.0

    # ── 결과 ──
    def report(self):
        print("\n" + "=" * 66)
        print("  측정 결과")
        print("=" * 66)
        b  = self.trim
        ll = self.lim_l
        lr = self.lim_r
        print(f"   직진 트림   b      = {'미측정' if b  is None else f'{b:+.1f}'}")
        print(f"   좌측 한계   lim_l  = {'미측정' if ll is None else f'{ll:+.1f}'}")
        print(f"   우측 한계   lim_r  = {'미측정' if lr is None else f'{lr:+.1f}'}")
        if None in (b, ll, lr):
            print("\n   → 셋 다 측정해야 매핑을 만들 수 있습니다 (t / [ / ]).")
            print("=" * 66)
            return
        if not (ll < b < lr):
            print("\n   ⚠ lim_l < b < lim_r 이 아닙니다. 기록을 다시 확인하세요.")
            print("=" * 66)
            return

        left_span, right_span = b - ll, lr - b
        print(f"\n   가용 폭:  좌 {left_span:.1f}   우 {right_span:.1f}   "
              f"비대칭 {max(left_span, right_span) / max(1e-6, min(left_span, right_span)):.2f}배")

        # 현재(오프셋) 방식에서 잘려나가는 양
        cut_l = max(0.0, (-CTRL_RANGE + b) - ll)
        cut_r = max(0.0, lr - (CTRL_RANGE + b))
        print(f"   지금의 '지령 + {b:+.1f}' 방식이면 컨트롤러 ±{CTRL_RANGE:.0f} 기준")
        print(f"     좌측 {cut_l:.1f} 만큼 클립 손실 / 우측 {cut_r:.1f} 만큼 미도달")

        print("\n" + "-" * 66)
        print("  total.py send_to_motor() 의 `float(steer)-8.` 자리에 아래를 적용하세요")
        print("  (camera_test_node.py 의 xycar_steer_offset 도 같은 보정입니다)")
        print("-" * 66)
        print(f"""
# 조향 지령 → 모터 매핑 (steer_calib.py 실측)
#   b     : 앞바퀴가 직진이 되는 지령값 (기계적 중립 오프셋)
#   lim_l : 좌측으로 더 지령해도 바퀴가 안 꺾이는 지점
#   lim_r : 우측 동일
# 좌/우를 각각 스케일해 두 끝이 정확히 모터 한계에 닿게 한다.
# 단순히 b 를 더하는 방식은 좌측이 범위를 넘어 잘려 조향 권한을 잃는다.
STEER_TRIM  = {b:.1f}
STEER_LIM_L = {ll:.1f}
STEER_LIM_R = {lr:.1f}
STEER_CTRL_RANGE = {CTRL_RANGE:.1f}   # lane_detection3 MODE_GAINS 의 최대 max_steer

def to_motor(steer):
    s = max(-STEER_CTRL_RANGE, min(STEER_CTRL_RANGE, float(steer)))
    span = (STEER_LIM_R - STEER_TRIM) if s >= 0 else (STEER_TRIM - STEER_LIM_L)
    return STEER_TRIM + s * span / STEER_CTRL_RANGE
""")
        print("-" * 66)
        print("  매핑 검산")
        for s in (-CTRL_RANGE, -CTRL_RANGE / 2, 0.0, CTRL_RANGE / 2, CTRL_RANGE):
            span = (lr - b) if s >= 0 else (b - ll)
            print(f"    지령 {s:+7.1f}  →  모터 {b + s * span / CTRL_RANGE:+7.1f}")
        print("=" * 66)

    # ── 메인 루프 ──
    def _tick(self):
        c = self._getch()
        if c is not None:
            if   c == 'a': self.sweep = False; self.angle -= self.step
            elif c == 'd': self.sweep = False; self.angle += self.step
            elif c == 'A': self.sweep = False; self.angle -= self.coarse
            elif c == 'D': self.sweep = False; self.angle += self.coarse
            elif c == '0': self.sweep = False; self.angle = 0.0
            elif c == 'z':
                self.sweep = False
                if self.trim is not None: self.angle = self.trim
            elif c == 's':
                self.sweep = not self.sweep
                print(f"\n[스윕 {'시작' if self.sweep else '정지'}]")
            elif c == 't':
                self.trim = self.angle
                print(f"\n[기록] 직진 트림 b = {self.trim:+.1f}")
            elif c == '[':
                self.lim_l = self.angle
                print(f"\n[기록] 좌측 한계 lim_l = {self.lim_l:+.1f}")
            elif c == ']':
                self.lim_r = self.angle
                print(f"\n[기록] 우측 한계 lim_r = {self.lim_r:+.1f}")
            elif c == 'r':
                self.trim = self.lim_l = self.lim_r = None
                print("\n[초기화] 기록 삭제")
            elif c == 'p':
                self.report()
            elif c == 'q':
                self.stop()
                raise SystemExit

        if self.sweep:
            self._sweep_step()

        self.angle = max(-CMD_ABS_LIMIT, min(CMD_ABS_LIMIT, self.angle))
        sent = self._publish(self.angle, self.speed)

        def f(v):
            return ' --- ' if v is None else f'{v:+5.1f}'
        sys.stdout.write(
            f"\r  angle {sent:+7.1f}  spd {self.speed:4.1f}  "
            f"| b {f(self.trim)}  L {f(self.lim_l)}  R {f(self.lim_r)}  "
            f"| {'SWEEP' if self.sweep else '     '}   ")
        sys.stdout.flush()


def main(args=None):
    rclpy.init(args=args)
    node = SteerCalibNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.stop()
        node.restore_terminal()
        node.report()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()