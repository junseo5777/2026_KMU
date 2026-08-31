#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 차량/콘 검출 캘리브레이션 노드 (x-band ratio 방식)
#
# 새 검출 파이프라인:
#   넓은 ROI → 클러스터링 → width 게이트 → perp_spread → x-band ratio → 프레임 투표
#   (트랙/approach/x_drift/len(clusters) 게이트 전부 제거)
#
# 캘리브레이션 목적상 ratio / perp / width 는 게이트 통과 여부와 무관하게
# 모든 클러스터에 대해 무조건 계산·출력한다. (차 분포 vs 벽 분포 비교용)
#
# 실행:  python3 lidar_calib_ratio.py 2>&1 | tee log.txt
# bag :  ros2 bag play <bagfile>   (다른 터미널)

import rclpy
import numpy as np
from collections import deque
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data


class LidarCalibNode(Node):
    def __init__(self):
        super().__init__('lidar_calib_ratio')

        # ── track_drive 와 동일한 좌표 변환 ──
        self.angle_offset = np.deg2rad(-90.0)
        self.max_range = 3.0

        # ── 넓은 ROI [0]=좌우, [1]=전방 ──
        # 좁은 ROI로 자르면 벽 chord 가 잘려서 width 게이트가 무력화되므로
        # 클러스터 형상을 온전히 보기 위해 넉넉하게 잡는다.
        self.roi_half_width = 1.00
        self.roi_x_min = 0.02
        self.roi_x_max = 2.50

        # ── 클러스터링 ──
        self.cluster_threshold = 0.16
        self.min_cluster_points = 3

        # ── 콘 판정 ──
        self.min_cone_width = 0.06
        self.max_cone_width = 0.10
        self.min_cone_count = 3
        self.history_len = 10
        self.min_positive_frames = 4
        self.cone_count_history = deque(maxlen=self.history_len)

        # ── 차량 판정 ──
        self.min_car_width = 0.12
        self.max_car_width = 0.24
        self.perp_spread_threshold = 0.015
        self.car_min_positive_frames = 5
        self.car_count_history = deque(maxlen=self.history_len)

        # ── x band (내 차선) ──
        self.x_min = -0.25
        self.x_max = 0.25
        self.ratio_threshold = 0.8

        self.angle_increment = None
        self.frame_id = 0

        # ── 분포 수집 (종료 시 요약 출력) ──
        self.stat_ratio_carwidth = []   # width 통과한 클러스터의 ratio
        self.stat_ratio_other = []      # width 미통과 클러스터의 ratio
        self.stat_perp_carwidth = []

        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.get_logger().info('lidar_calib_ratio node started — waiting for /scan ...')

    # ------------------------------------------------------------------
    def scan_callback(self, msg):
        self.angle_increment = msg.angle_increment
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        angles = (
            msg.angle_min
            + np.arange(len(ranges), dtype=np.float32) * msg.angle_increment
            + self.angle_offset
        )

        valid = (
            np.isfinite(ranges)
            & (ranges >= msg.range_min)
            & (ranges <= min(msg.range_max, self.max_range))
        )
        if not np.any(valid):
            cone_detect, cone_pos, car_detect, car_pos = self.vote(0, 0)
            print(f"[f{self.frame_id}] no valid points  "
                  f"cone_pos={cone_pos}/{self.history_len} cone_detect={cone_detect or '-'} "
                  f"car_pos={car_pos}/{self.history_len} car_detect={car_detect or '-'}")
            self.frame_id += 1
            return

        ranges = ranges[valid]
        angles = angles[valid]
        points = np.column_stack((-ranges * np.cos(angles),
                                  -ranges * np.sin(angles)))

        # 축 확인
        nearest = points[np.argmin(np.linalg.norm(points, axis=1))]
        print(f"[f{self.frame_id}][axis] nearest=({nearest[0]:+.2f},{nearest[1]:+.2f})")

        # 넓은 ROI
        roi = (
            (np.abs(points[:, 0]) < self.roi_half_width)
            & (points[:, 1] > self.roi_x_min)
            & (points[:, 1] < self.roi_x_max)
        )
        points = points[roi]

        clusters = self.clusterize(points)
        self.classfi(clusters)
        self.frame_id += 1

    # ------------------------------------------------------------------
    def clusterize(self, points):
        if len(points) < 2:
            return []
        gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        break_idx = np.where(gaps > self.cluster_threshold)[0] + 1

        # ── 끊긴 지점의 gap 크기 로깅 ──
        for bi in break_idx:
            g = gaps[bi - 1]
            # 끊긴 경계 양쪽 점의 좌표도 같이
            p_before = points[bi - 1]
            p_after  = points[bi]
            print(f"[f{self.frame_id}][gap] cut={g:.3f} "
                  f"between=({p_before[0]:+.2f},{p_before[1]:+.2f})|"
                  f"({p_after[0]:+.2f},{p_after[1]:+.2f})")

        clusters = np.split(points, break_idx)
        return [c for c in clusters if len(c) >= self.min_cluster_points]

    # ------------------------------------------------------------------
    def classfi(self, clusters):
        cone_points = []
        car_points = []

        centers = [np.mean(c, axis=0) for c in clusters]

        for i, cluster in enumerate(clusters):
            cluster_center = centers[i]
            d = np.linalg.norm(cluster_center)

            cluster_width = (
                np.linalg.norm(cluster[0] - cluster[-1]) + d * self.angle_increment
            )

            eigvals = np.linalg.eigh(np.cov(cluster.T))[0]
            perp_spread = np.sqrt(max(eigvals[0], 0.0))

            is_cone   = self.min_cone_width < cluster_width < self.max_cone_width
            car_width = self.min_car_width  < cluster_width < self.max_car_width

            # ── ratio 는 게이트와 무관하게 항상 계산 (캘리브레이션 목적) ──
            in_band = (cluster[:, 0] >= self.x_min) & (cluster[:, 0] <= self.x_max)
            n_in = int(np.count_nonzero(in_band))
            ratio = n_in / len(cluster)

            x_lo = float(np.min(cluster[:, 0]))
            x_hi = float(np.max(cluster[:, 0]))

            # 밴드를 완전히 가로지르는지 (벽의 전형적 패턴)
            crosses = (x_lo < self.x_min) and (x_hi > self.x_max)

            if car_width:
                self.stat_ratio_carwidth.append(ratio)
                self.stat_perp_carwidth.append(perp_spread)
            else:
                self.stat_ratio_other.append(ratio)

            if is_cone:
                cone_points.append(cluster_center)

            # ── 실제 검출 조건 ──
            perp_ok  = perp_spread <= self.perp_spread_threshold
            ratio_ok = ratio >= self.ratio_threshold
            if car_width and perp_ok and ratio_ok:
                car_points.append(cluster_center)
                verdict = "CAR"
            elif car_width and perp_ok:
                verdict = "drop:ratio"
            elif car_width:
                verdict = "drop:perp"
            elif is_cone:
                verdict = "cone"
            else:
                verdict = "-"

            print(
                f"[f{self.frame_id}][cls] "
                f"c=({cluster_center[0]:+.2f},{cluster_center[1]:+.2f}) "
                f"w={cluster_width:.3f} n={len(cluster)} perp={perp_spread:.3f} "
                f"x=[{x_lo:+.2f},{x_hi:+.2f}] in={n_in} ratio={ratio:.2f} "
                f"cross={int(crosses)} cone={int(is_cone)} carw={int(car_width)} "
                f"→ {verdict}"
            )

        cone_detect, cone_pos, car_detect, car_pos = self.vote(
            len(cone_points), len(car_points)
        )

        cx = f"{car_points[0][0]:+.2f}" if car_points else "None"
        cy = f"{car_points[0][1]:+.2f}" if car_points else "None"
        print(
            f"--- [f{self.frame_id}] clusters={len(clusters)} "
            f"cones={len(cone_points)} cone_pos={cone_pos}/{self.history_len} "
            f"cone_detect={cone_detect or '-'} "
            f"cars={len(car_points)} car_pos={car_pos}/{self.history_len} "
            f"car_detect={car_detect or '-'} car0=({cx},{cy})"
        )

    # ------------------------------------------------------------------
    def vote(self, n_cones, n_cars):
        self.cone_count_history.append(n_cones)
        cone_pos = sum(1 for c in self.cone_count_history
                       if c >= self.min_cone_count)
        cone_detect = "rubbercone" if cone_pos >= self.min_positive_frames else ""

        self.car_count_history.append(n_cars)
        car_pos = sum(1 for c in self.car_count_history if c >= 1)
        car_detect = "car" if car_pos >= self.car_min_positive_frames else ""

        return cone_detect, cone_pos, car_detect, car_pos

    # ------------------------------------------------------------------
    def print_summary(self):
        def dist(name, arr):
            if not arr:
                print(f"  {name}: (없음)")
                return
            a = np.asarray(arr)
            qs = np.percentile(a, [5, 25, 50, 75, 95])
            print(f"  {name}: n={len(a)} min={a.min():.2f} "
                  f"p5={qs[0]:.2f} p25={qs[1]:.2f} p50={qs[2]:.2f} "
                  f"p75={qs[3]:.2f} p95={qs[4]:.2f} max={a.max():.2f}")

        print("\n" + "=" * 70)
        print(f"SUMMARY  frames={self.frame_id}  "
              f"band=[{self.x_min:+.2f},{self.x_max:+.2f}] "
              f"ratio_thr={self.ratio_threshold}")
        print("-" * 70)
        print("[ratio 분포]  ← 임계값은 이 두 분포가 갈리는 지점에서 고를 것")
        dist("width통과(차 후보)", self.stat_ratio_carwidth)
        dist("width미통과(콘/벽)", self.stat_ratio_other)
        print("[perp_spread 분포]  ← width통과 클러스터만")
        dist("perp", self.stat_perp_carwidth)
        print("=" * 70)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCalibNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.print_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()