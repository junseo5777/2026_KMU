# Copyright 2026 junseo5777 and kimmc53
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Path-guided DWA-style steering rollouts in the ROS ``base_link`` frame."""

import math
from dataclasses import dataclass

import numpy as np


def normalize_angle(angle):
    """Wrap one angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def poses_to_local(poses, vehicle_pose):
    """Transform map-frame ``(x, y, yaw)`` poses into ``base_link``."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    vehicle_x, vehicle_y, vehicle_yaw = vehicle_pose
    delta_x = poses[:, 0] - vehicle_x
    delta_y = poses[:, 1] - vehicle_y
    cosine = math.cos(vehicle_yaw)
    sine = math.sin(vehicle_yaw)

    local = np.empty_like(poses)
    local[:, 0] = cosine * delta_x + sine * delta_y
    local[:, 1] = -sine * delta_x + cosine * delta_y
    yaw_delta = poses[:, 2] - vehicle_yaw
    local[:, 2] = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
    return local


def poses_to_map(poses, vehicle_pose):
    """Transform ``base_link`` ``(x, y, yaw)`` poses into the map frame."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    vehicle_x, vehicle_y, vehicle_yaw = vehicle_pose
    cosine = math.cos(vehicle_yaw)
    sine = math.sin(vehicle_yaw)

    mapped = np.empty_like(poses)
    mapped[:, 0] = vehicle_x + cosine * poses[:, 0] - sine * poses[:, 1]
    mapped[:, 1] = vehicle_y + sine * poses[:, 0] + cosine * poses[:, 1]
    yaw_sum = poses[:, 2] + vehicle_yaw
    mapped[:, 2] = np.arctan2(np.sin(yaw_sum), np.cos(yaw_sum))
    return mapped


def nearest_path_index(
        poses,
        position,
        progress_index=0,
        search_back=5,
        search_ahead=80,
        end_index=None):
    """Find a nearby path index without allowing an arbitrary path jump."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    if len(poses) == 0:
        raise ValueError('poses cannot be empty')
    progress_index = int(np.clip(progress_index, 0, len(poses) - 1))
    first = max(0, progress_index - max(0, int(search_back)))
    last = min(
        len(poses) - 1,
        progress_index + max(0, int(search_ahead)),
    )
    if end_index is not None:
        last = min(last, max(first, int(end_index)))
    candidates = poses[first:last + 1, :2]
    distances = np.hypot(
        candidates[:, 0] - position[0],
        candidates[:, 1] - position[1],
    )
    local_index = int(np.argmin(distances))
    return first + local_index, float(distances[local_index])


def path_start_reacquire_metrics(poses, vehicle_pose):
    """
    Return position and body-yaw errors from the first path pose.

    The follower uses both errors before rewinding progress.  Position alone
    is unsafe when the path loops back near its own start with another
    heading.
    """
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    if len(poses) == 0:
        raise ValueError('poses cannot be empty')
    vehicle_pose = np.asarray(
        vehicle_pose, dtype=np.float64
    ).reshape(3)
    position_error = float(np.hypot(
        poses[0, 0] - vehicle_pose[0],
        poses[0, 1] - vehicle_pose[1],
    ))
    yaw_error = abs(normalize_angle(poses[0, 2] - vehicle_pose[2]))
    return position_error, yaw_error


def direction_change_index(directions, start_index, current_direction):
    """Return the first pose index using a new gear, or the last pose."""
    directions = np.asarray(directions, dtype=np.int8).reshape(-1)
    if len(directions) == 0:
        raise ValueError('directions cannot be empty')
    start_index = int(np.clip(start_index, 0, len(directions) - 1))
    current_direction = 1 if current_direction >= 0 else -1
    changes = np.flatnonzero(
        directions[start_index:] != current_direction
    )
    if len(changes):
        return start_index + int(changes[0])
    return len(directions) - 1


def path_distance(poses, start_index, end_index):
    """Return path arclength between two ordered pose indices."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    start_index = int(np.clip(start_index, 0, len(poses) - 1))
    end_index = int(np.clip(end_index, start_index, len(poses) - 1))
    if end_index <= start_index:
        return 0.0
    steps = np.diff(poses[start_index:end_index + 1, :2], axis=0)
    return float(np.hypot(steps[:, 0], steps[:, 1]).sum())


def lookahead_index(poses, start_index, end_index, distance):
    """Return the first ordered path index at least ``distance`` ahead."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    start_index = int(np.clip(start_index, 0, len(poses) - 1))
    end_index = int(np.clip(end_index, start_index, len(poses) - 1))
    target_distance = max(0.0, float(distance))
    accumulated = 0.0
    for index in range(start_index + 1, end_index + 1):
        accumulated += float(np.hypot(
            poses[index, 0] - poses[index - 1, 0],
            poses[index, 1] - poses[index - 1, 1],
        ))
        if accumulated >= target_distance:
            return index
    return end_index


def filter_vehicle_self_points(
        points,
        vehicle_length,
        vehicle_width,
        rear_overhang,
        padding=0.04):
    """Remove scan returns inside the vehicle's own rectangular body."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return points
    rear = -float(rear_overhang) - float(padding)
    front = float(vehicle_length) - float(rear_overhang) + float(padding)
    half_width = 0.5 * float(vehicle_width) + float(padding)
    is_self = (
        (points[:, 0] >= rear)
        & (points[:, 0] <= front)
        & (np.abs(points[:, 1]) <= half_width)
    )
    return points[~is_self]


@dataclass
class DWAResult:
    """One path-guided local rollout selection result."""

    success: bool
    reason: str
    trajectory: np.ndarray
    steer: float
    score: float
    clearance: float
    candidates: tuple
    valid_candidates: tuple


class PathTrackingDWA:
    """Run the track-drive DWA structure in the SLAM ``base_link`` frame."""

    def __init__(self):
        """Set the single authoritative parameter values for the DWA core."""
        self.wheel_base = 0.345
        self.max_steer = math.radians(27.0)
        self.dwa_count = 13
        self.step = 0.13
        self.prediction_distance = 0.7

        # SLAM 차량의 실제 차체 footprint. 추가 collision margin은 없다.
        self.vehicle_length = 0.585
        self.vehicle_width = 0.290
        self.rear_overhang = 0.120
        self.footprint_disc_count = 3
        self.safe_dist = 0.23

        # 모든 항을 "작을수록 좋은 비용"으로 통일 -> 가중치 전부 양수.
        self.w1 = 0.1   # heading 오차
        self.w2 = 5.0   # 중간 점 장애물 근접 페널티
        self.w3 = 4.0   # 끝 점 장애물 근접 페널티
        self.w4 = 3.0   # SLAM lookahead 점 거리 페널티

    def validate(self):
        """Raise ValueError for unsafe or nonsensical parameters."""
        positive = {
            'wheel_base': self.wheel_base,
            'max_steer': self.max_steer,
            'dwa_count': self.dwa_count,
            'step': self.step,
            'prediction_distance': self.prediction_distance,
            'vehicle_length': self.vehicle_length,
            'vehicle_width': self.vehicle_width,
            'footprint_disc_count': self.footprint_disc_count,
            'safe_dist': self.safe_dist,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(
                f'DWA parameters must be positive: {", ".join(invalid)}'
            )
        if self.max_steer >= math.pi / 2.0:
            raise ValueError('max_steer must be less than pi/2')
        if self.dwa_count < 3:
            raise ValueError('dwa_count must be at least 3')
        if self.rear_overhang < 0.0:
            raise ValueError('rear_overhang cannot be negative')
        weights = (self.w1, self.w2, self.w3, self.w4)
        if any(weight < 0.0 for weight in weights):
            raise ValueError('DWA cost weights cannot be negative')

    def dynamic_window(self, steer=0.0):
        """Create the same current-steer-centered window as ``kmu/dwa.py``."""
        current = float(np.clip(
            steer, -self.max_steer, self.max_steer
        ))
        dwa_step = 2.0 * self.max_steer / (int(self.dwa_count) - 1)
        dwa_steer = current - self.max_steer + dwa_step * np.arange(
            int(self.dwa_count), dtype=np.float64
        )
        return dwa_steer[
            np.abs(dwa_steer) <= self.max_steer + 1.0e-12
        ]

    def rollout(self, steer, direction, distance=None):
        """Roll a constant-steer bicycle path in ``base_link`` coordinates."""
        direction = 1.0 if direction >= 0 else -1.0
        if distance is None:
            distance = self.prediction_distance
        distance = max(float(distance), self.step)
        count = max(2, int(math.ceil(distance / self.step)) + 1)
        arc = direction * np.linspace(0.0, distance, count)
        curvature = math.tan(float(steer)) / self.wheel_base
        trajectory = np.zeros((count, 3), dtype=np.float64)
        if abs(curvature) < 1.0e-9:
            trajectory[:, 0] = arc
        else:
            yaw = curvature * arc
            trajectory[:, 0] = np.sin(yaw) / curvature
            trajectory[:, 1] = (1.0 - np.cos(yaw)) / curvature
            trajectory[:, 2] = yaw
        return trajectory

    def _footprint_geometry(self):
        """Return physical covering-disc geometry without an extra margin."""
        count = int(self.footprint_disc_count)
        segment = self.vehicle_length / count
        radius = math.hypot(0.5 * segment, 0.5 * self.vehicle_width)
        offsets = np.array([
            -self.rear_overhang + segment * (index + 0.5)
            for index in range(count)
        ])
        return offsets, radius

    def trajectory_clearance(self, trajectory, obstacles):
        """Return minimum obstacle distance beyond the covered footprint."""
        obstacles = np.asarray(obstacles, dtype=np.float64).reshape(-1, 2)
        if len(obstacles) == 0:
            return math.inf
        offsets, required = self._footprint_geometry()
        yaw = trajectory[:, 2]
        centers_x = (
            trajectory[:, 0:1] + np.cos(yaw[:, None]) * offsets[None, :]
        )
        centers_y = (
            trajectory[:, 1:2] + np.sin(yaw[:, None]) * offsets[None, :]
        )
        delta_x = centers_x[:, :, None] - obstacles[None, None, :, 0]
        delta_y = centers_y[:, :, None] - obstacles[None, None, :, 1]
        distance = np.sqrt(delta_x * delta_x + delta_y * delta_y)
        return float(distance.min()) - required

    def predict_path(self, steer=0.0, direction=1, distance=None):
        """Create every constant-steer path in the dynamic window."""
        all_path = []
        for dwa_steer in self.dynamic_window(steer):
            path = self.rollout(float(dwa_steer), direction, distance)
            all_path.append((float(dwa_steer), path))
        return all_path

    def _obstacle_cost(self, clearance):
        """Scale obstacle cost by the fraction of ``safe_dist`` invaded."""
        if not math.isfinite(clearance) or clearance >= self.safe_dist:
            return 0.0
        return (self.safe_dist - float(clearance)) / self.safe_dist

    def cost_function(
            self, point, target, obs, obs_weight, clearance=None):
        """Return heading plus obstacle cost for one predicted path point."""
        point = np.asarray(point, dtype=np.float64).reshape(1, 3)
        target = np.asarray(target, dtype=np.float64).reshape(3)
        if clearance is None:
            clearance = self.trajectory_clearance(point, obs)
        heading_cost = abs(normalize_angle(point[0, 2] - target[2]))
        obs_cost = self._obstacle_cost(clearance)
        return self.w1 * heading_cost + obs_weight * obs_cost

    def dwa(
            self,
            target,
            obs,
            direction,
            steer=0.0,
            prediction_distance=None):
        """Select one path using track-drive costs plus lookahead distance."""
        self.validate()
        target = np.asarray(target, dtype=np.float64).reshape(3)
        obs = np.asarray(obs, dtype=np.float64).reshape(-1, 2)

        best = None
        candidates = []
        valid_candidates = []
        all_path = self.predict_path(steer, direction, prediction_distance)
        for dwa_steer, trajectory in all_path:
            candidates.append(trajectory)

            # track_drive와 같이 후보의 중간점과 끝점에서만 footprint를
            # 계산한다. 같은 clearance를 충돌 판정과 비용에 함께 쓴다.
            middle = trajectory[len(trajectory) // 2]
            endpoint = trajectory[-1]
            middle_clearance = self.trajectory_clearance(
                middle[None, :], obs
            )
            endpoint_clearance = self.trajectory_clearance(
                endpoint[None, :], obs
            )
            clearance = min(middle_clearance, endpoint_clearance)
            if clearance < 0.0:
                valid_candidates.append(False)
                continue

            valid_candidates.append(True)
            total_cost = self.cost_function(
                middle,
                target,
                obs,
                self.w2,
                clearance=middle_clearance,
            ) + self.cost_function(
                endpoint,
                target,
                obs,
                self.w3,
                clearance=endpoint_clearance,
            )

            # 후보 끝점이 SLAM lookahead 점에 가까워지도록 유도한다.
            lookahead_cost = float(np.linalg.norm(
                endpoint[:2] - target[:2]
            ))
            total_cost += self.w4 * lookahead_cost

            if best is None or total_cost < best[0]:
                best = (total_cost, trajectory, dwa_steer, clearance)

        if best is None:
            return DWAResult(
                False,
                'no collision-free path-tracking rollout',
                np.empty((0, 3)),
                0.0,
                math.inf,
                -math.inf,
                tuple(candidates),
                tuple(valid_candidates),
            )
        return DWAResult(
            True,
            'selected',
            best[1],
            best[2],
            best[0],
            best[3],
            tuple(candidates),
            tuple(valid_candidates),
        )
