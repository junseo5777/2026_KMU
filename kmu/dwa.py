import numpy as np
import math


class DWA():
    def __init__(self, wheel_base=0.4, sensor_forward_offset=0.0):
        self.wheel_base = float(wheel_base)
        # 0이면 nuclear.py를 포함한 기존 LiDAR-원점 경로를 그대로 쓴다.
        # 양수이면 경로는 LiDAR 뒤의 rear axle에서 시작한다. 장애물은
        # LiDAR 좌표를 유지하며, 비용은 RViz에 표시된 rear-axle 경로점과
        # 장애물 사이의 거리를 그대로 사용한다.
        self.sensor_forward_offset = float(sensor_forward_offset)
        self.max_steer = np.deg2rad(21.0)
        self.dwa_count =13
        self.step = 0.21
        # 모든 항을 "작을수록 좋은 비용"으로 통일 → 가중치 전부 양수
        self.w1 = 0.1   # heading 오차 (|yaw|)
        self.w2 = 2.5   # 중간 점 장애물 근접 페널티
        self.w3 = 2.2   # 끝 점 장애물 근접 패널티
        self.middle_safe_dist = 0.32  # 중간점(idx 2) 안전거리 [m]
        self.end_safe_dist = 0.34     # 끝점(idx 5) 안전거리 [m]

    def dynamic_window(self, steer=0.):
        dwa_step = 2 * self.max_steer / (self.dwa_count - 1)
        dwa_steer = [steer - self.max_steer + dwa_step * i
                     for i in range(self.dwa_count)
                     if abs(steer - self.max_steer + dwa_step * i) <= self.max_steer]
        return dwa_steer

    def predict_path(self, steer = 0.):
        if self.sensor_forward_offset > 0.0:
            return self.predict_rear_axle_path(steer)

        all_path = []
        steer_list = self.dynamic_window(steer)
        for st in steer_list:
            # 차량 원점(0,0)을 path 첫 점으로 포함 → RViz LINE_STRIP 이 차량에서 시작
            path = [(0., 0., math.pi / 2)]
            px = py = 0.
            # 라이다 좌표계: 차량 정면 = +y → 초기 heading 은 π/2
            pyaw = st-np.pi/2
            for _ in range(5):
                px += self.step*math.cos(pyaw) 
                py += self.step*math.sin(pyaw)
                # 양수 steer = 우회전(decide_turn 규약) → yaw 감소(+y → +x)
                pyaw += self.step / self.wheel_base * math.tan(st)
                path.append((-px, -py, -pyaw))
            all_path.append((st, path))
        return all_path

    def predict_rear_axle_path(self, steer=0.):
        """Predict rear-axle paths in the legacy x-right/y-forward axes."""
        all_path = []
        steer_list = self.dynamic_window(steer)
        rear_y = -self.sensor_forward_offset

        for st in steer_list:
            curvature = math.tan(st) / self.wheel_base
            path = []
            for idx in range(6):
                arc = self.step * idx
                if abs(curvature) < 1.0e-9:
                    x = 0.0
                    y = rear_y + arc
                    yaw = math.pi / 2
                else:
                    yaw_delta = curvature * arc
                    # 표준 base_link(+x 전방/+y 좌측) rear-axle bicycle
                    # 궤적을 레거시 축(+x 우측/+y 전방)으로 옮긴 값이다.
                    x = -(1.0 - math.cos(yaw_delta)) / curvature
                    y = rear_y + math.sin(yaw_delta) / curvature
                    yaw = math.pi / 2 + yaw_delta
                path.append((x, y, yaw))
            all_path.append((st, path))

        return all_path

    def cost_function(self, point, yaw, obs, obs_weight, safe_dist):
        p = np.array(point[:2])
        obs_cost = 0.
        # 장애물이 없으면 충돌 위험 없음, 거리 페널티 0
        if obs is None or len(obs) == 0:
            dis_min = float('inf')
        else:
            # 원형 안전거리 안의 점은 반드시 이 사각 경계 안에도 있다.
            # 경계 밖 점은 기존 결과에 영향을 줄 수 없으므로 sqrt 계산 전에 제외한다.
            delta = obs - p
            near_mask = (
                (np.abs(delta[:, 0]) < safe_dist)
                & (np.abs(delta[:, 1]) < safe_dist)
            )
            if np.any(near_mask):
                dis_min = np.min(
                    np.linalg.norm(delta[near_mask], axis=1)
                )
                if dis_min < safe_dist:
                    obs_cost = 1.0 / dis_min
        heading_cost = abs(yaw-math.pi/2)
        # obs_weight: 끝 점이면 w2, 중간 점이면 w3
        return self.w1 * heading_cost + obs_weight * obs_cost


    def dwa(self,s, obs):
        best_cost = float('inf')
        best_path = None
        best_steer = 0.
        # 호출당 한 번만 배열화한다. cost_function 26회에서 같은 배열을 재사용한다.
        if obs is not None:
            obs = np.asarray(obs, dtype=float)
        all_path = self.predict_path(s)
        check_specs = (
            (2, self.w2, self.middle_safe_dist),
            (5, self.w3, self.end_safe_dist),
        )
        for steer, path in all_path: # [path1, path2, path3, path4 ...]
            obs_cost = 0.
            # RViz에 그린 rear-axle 경로의 중간점과 끝점에서 장애물 거리 계산
            # 중간점(idx 2)은 w2, 끝점(idx 5)은 w3 적용
            for idx, obs_weight, safe_dist in check_specs:
                p = path[idx]
                c = self.cost_function(
                    p, p[2], obs, obs_weight, safe_dist)
                obs_cost += c

            # heading은 경로 끝점만 평가하면 충분
            total_cost = obs_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path
                best_steer = steer
                


        return best_path, best_steer, all_path
