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

"""
SLAM 지도 위에서 global path를 뽑는 양방향 Hybrid A* 플래너.

원래 메모(작성자):

1. dwa 마냥 속도, 스티어 후보군에 대해 1스텝 뻗음 -> 목적지의 pose로부터 탈출하여
   출발지의 pose로 도달하는 경로 1개, 출발지에서 목적지의 pose로 도달하는 경로 1개
2. dwa 마냥 코스트 매김 (ex)
3. 뻗어나가며 2가지 각각의 노드가 클로소이드로 부드럽게 이어질 수 있는 노드이면
   클로소이드로 이음
4. glob_path 생성 후 dwa 코스트 계산하면서 (q가 멀어지지 않게) dwa path 생성
5. 경로 추종 (lidar_odom, acml 기반 pose로 dwa path 추종)

이 모듈은 1~3을 담당한다. 4는 기존 ``kmu/dwa.py``가, 5는 주행 노드가 맡는다.

동작 요약::

    kmu_parking.yaml/pgm (또는 nav_msgs/OccupancyGrid)
      -> OccupancyMap: 점유/미지 셀까지의 거리장(clearance) 생성
      -> HybridAStar.plan(start, goal)
           출발 트리: start에서 자전거 모델로 1스텝씩 전진(필요하면 후진)
           목적 트리: goal에서 같은 모델로 시간 역방향 1스텝씩 "탈출"
           매 확장마다 두 트리의 노드를 G1 클로소이드로 이어보고,
           곡률 한계와 충돌 검사를 통과하면 그 순간 경로 확정
      -> PlanResult.poses: map frame 기준 (x, y, yaw) 배열

좌표계 주의::

    이 모듈은 전부 map frame ROS 규약을 쓴다. +x 전방, +y 좌측,
    yaw는 +x 기준 CCW.  ``kmu/dwa.py``는 laser frame(+y 전방, yaw 기준 pi/2)에
    부호까지 뒤집어 쓰고 있으므로, 4단계에서 이 global path를 DWA에 먹일 때는
    map -> base_link -> (dwa 좌표) 변환을 반드시 한 번 거쳐야 한다.

CLI::

    python3 -m cartographer_2d_slam.hs \
        --map maps/kmu_parking.yaml \
        --start 0.1,1.5,0 --goal 6.5,-1.5,-90 \
        --preview /tmp/hiass.ppm --out /tmp/glob_path.csv
"""

import argparse
import heapq
import math
import os
import time

import numpy as np


# 실측 차량 제원. vehicle_marker_publisher.py와 같은 값을 쓴다.
WHEEL_BASE_M = 0.345          # 앞뒤 축간 거리(m). base_link 원점 = 뒷바퀴 축 중앙
VEHICLE_LENGTH_M = 0.585      # 차체 전장(m)
VEHICLE_WIDTH_M = 0.290       # 차체 전폭(m)
MAX_STEER_RAD = math.radians(25.0)  # Hybrid A*와 DWA가 공유하는 조향각

# 전후 오버행을 따로 재지 않았으므로 축 사이에 차체를 중앙 배치한 근사값을 쓴다.
REAR_OVERHANG_M = (VEHICLE_LENGTH_M - WHEEL_BASE_M) / 2.0

_LARGE = 1.0e9


def normalize_angle(angle):
    """각도를 [-pi, pi]로 감는다."""
    return math.atan2(math.sin(angle), math.cos(angle))


def footprint_discs(disc_count=3):
    """차체 사각형을 덮는 (base_link 기준 x offset, 반지름) 목록을 만든다."""
    if disc_count < 1:
        raise ValueError('disc_count must be positive')
    half_width = VEHICLE_WIDTH_M / 2.0
    segment = VEHICLE_LENGTH_M / disc_count
    radius = math.hypot(segment / 2.0, half_width)
    offsets = [
        -REAR_OVERHANG_M + segment * (index + 0.5)
        for index in range(disc_count)
    ]
    return np.asarray(offsets, dtype=np.float64), radius


# ---------------------------------------------------------------------------
# 점유 격자
# ---------------------------------------------------------------------------


def _squared_distance_transform(blocked):
    """
    막힌 셀까지의 정확한 제곱 거리(셀 단위)를 numpy만으로 계산한다.

    scipy.ndimage가 이 환경에서 numpy 2.x와 ABI가 맞지 않아 쓸 수 없으므로,
    분리 가능(separable) 거리 변환을 직접 구현했다. 1축은 누적 최소값으로,
    2축은 열 묶음 단위 min-plus 연산으로 처리한다.
    """
    height, width = blocked.shape
    limit = float(height + width)

    columns = np.arange(width, dtype=np.float64)
    seed = np.where(blocked, 0.0, _LARGE)
    forward = np.minimum.accumulate(seed - columns, axis=1) + columns
    backward = (
        np.minimum.accumulate((seed + columns)[:, ::-1], axis=1)[:, ::-1]
        - columns
    )
    row_distance = np.minimum(np.minimum(forward, backward), limit)
    squared = (row_distance ** 2).astype(np.float32)

    rows = np.arange(height, dtype=np.float32)
    offsets = (rows[:, np.newaxis] - rows[np.newaxis, :]) ** 2
    result = np.empty_like(squared)
    budget = 8_000_000
    chunk = max(1, budget // max(height * height, 1))
    for start in range(0, width, chunk):
        stop = min(start + chunk, width)
        block = (
            squared[np.newaxis, :, start:stop]
            + offsets[:, :, np.newaxis]
        )
        result[:, start:stop] = block.min(axis=1)
    return result


def _read_pgm(path):
    """Nav2 지도용 이진 PGM(P5)을 (height, width) uint8 배열로 읽는다."""
    with open(path, 'rb') as pgm_file:
        raw = pgm_file.read()

    tokens = []
    index = 0
    while len(tokens) < 4:
        if index >= len(raw):
            raise ValueError(f'{path} is not a complete PGM file')
        character = raw[index:index + 1]
        if character == b'#':
            while index < len(raw) and raw[index:index + 1] != b'\n':
                index += 1
        elif character.isspace():
            index += 1
        else:
            stop = index
            while stop < len(raw) and not raw[stop:stop + 1].isspace():
                stop += 1
            tokens.append(raw[index:stop])
            index = stop
    index += 1

    if tokens[0] != b'P5':
        raise ValueError(f'{path} must be a binary P5 PGM, got {tokens[0]!r}')
    width, height, maximum = (int(token) for token in tokens[1:4])
    if maximum > 255:
        raise ValueError('16-bit PGM maps are not supported')
    expected = width * height
    pixels = np.frombuffer(raw[index:index + expected], dtype=np.uint8)
    if len(pixels) != expected:
        raise ValueError(
            f'{path} holds {len(pixels)} pixels; expected {expected}'
        )
    return pixels.reshape(height, width)


class OccupancyMap:
    """
    점유 격자와 거리장(clearance)을 함께 들고 있는 지도.

    ``occupancy``는 nav_msgs/OccupancyGrid와 같은 row-major 배열이며 행 0이
    최소 y에 해당한다. 값은 0..100 점유 확률, -1은 미지 영역이다.
    """

    def __init__(
            self,
            occupancy,
            resolution,
            origin,
            occupied_value=65,
            unknown_is_blocked=True):
        """격자를 받아 좌표 변환 상수와 거리장을 준비한다."""
        occupancy = np.asarray(occupancy)
        if occupancy.ndim != 2:
            raise ValueError('occupancy must be a 2-D array')
        if resolution <= 0.0:
            raise ValueError('resolution must be positive')

        self.occupancy = occupancy
        self.resolution = float(resolution)
        self.origin_x = float(origin[0])
        self.origin_y = float(origin[1])
        self.origin_yaw = float(origin[2]) if len(origin) > 2 else 0.0
        self.height, self.width = occupancy.shape

        blocked = occupancy >= occupied_value
        if unknown_is_blocked:
            blocked |= occupancy < 0
        self.blocked = blocked

        if not blocked.any():
            self.clearance = np.full(
                occupancy.shape, _LARGE, dtype=np.float32
            )
        else:
            self.clearance = (
                np.sqrt(_squared_distance_transform(blocked))
                * np.float32(self.resolution)
            )

        self._cos_yaw = math.cos(-self.origin_yaw)
        self._sin_yaw = math.sin(-self.origin_yaw)

    @classmethod
    def from_nav2_yaml(
            cls,
            yaml_path,
            free_thresh=None,
            occupied_thresh=None,
            unknown_is_blocked=True):
        """
        Nav2 지도 yaml/pgm 한 쌍을 읽어 OccupancyMap을 만든다.

        ``free_thresh``/``occupied_thresh``를 주면 yaml 값을 덮어쓴다.
        ``free_thresh=0.19``이면 map_saver가 남긴 미지 영역(픽셀 205)은
        unknown으로 분류된다. ``unknown_is_blocked=True``인 플래너에서는
        이 영역을 통과할 수 없다.
        """
        import yaml  # PyYAML은 ROS 환경에 항상 있다.

        yaml_path = os.path.abspath(os.path.expanduser(yaml_path))
        with open(yaml_path, 'r', encoding='utf-8') as handle:
            metadata = yaml.safe_load(handle)

        image_name = metadata['image']
        image_path = image_name
        if not os.path.isabs(image_path):
            image_path = os.path.join(os.path.dirname(yaml_path), image_path)

        resolution = float(metadata['resolution'])
        origin = [float(value) for value in metadata['origin']]
        negate = int(metadata.get('negate', 0))
        if free_thresh is None:
            free_thresh = float(metadata.get('free_thresh', 0.25))
        if occupied_thresh is None:
            occupied_thresh = float(metadata.get('occupied_thresh', 0.65))

        # PGM은 행 0이 최대 y이므로 OccupancyGrid 순서로 뒤집는다.
        pixels = _read_pgm(image_path)[::-1, :].astype(np.float32)
        shade = pixels / 255.0 if negate else (255.0 - pixels) / 255.0

        occupancy = np.full(shade.shape, -1, dtype=np.int16)
        occupancy[shade > occupied_thresh] = 100
        occupancy[shade < free_thresh] = 0
        return cls(
            occupancy,
            resolution,
            origin,
            unknown_is_blocked=unknown_is_blocked,
        )

    @classmethod
    def from_occupancy_grid(cls, message, unknown_is_blocked=True):
        """nav_msgs/OccupancyGrid 메시지에서 바로 만든다 (map_server 구독용)."""
        info = message.info
        occupancy = np.asarray(message.data, dtype=np.int16).reshape(
            info.height, info.width
        )
        orientation = info.origin.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z
                   + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y
                         + orientation.z * orientation.z),
        )
        origin = (info.origin.position.x, info.origin.position.y, yaw)
        return cls(
            occupancy,
            info.resolution,
            origin,
            unknown_is_blocked=unknown_is_blocked,
        )

    def world_to_grid(self, x, y):
        """Map frame 좌표를 (열, 행) 인덱스로 바꾼다. 배열 입력을 지원한다."""
        local_x = np.asarray(x, dtype=np.float64) - self.origin_x
        local_y = np.asarray(y, dtype=np.float64) - self.origin_y
        rotated_x = self._cos_yaw * local_x - self._sin_yaw * local_y
        rotated_y = self._sin_yaw * local_x + self._cos_yaw * local_y
        column = np.floor(rotated_x / self.resolution).astype(np.int64)
        row = np.floor(rotated_y / self.resolution).astype(np.int64)
        return column, row

    def clearance_at(self, x, y):
        """
        해당 map 좌표에서 가장 가까운 장애물까지의 거리(m)를 돌려준다.

        지도 밖은 0.0으로 취급해 항상 충돌로 판정한다.
        """
        column, row = self.world_to_grid(x, y)
        inside = (
            (column >= 0) & (column < self.width)
            & (row >= 0) & (row < self.height)
        )
        safe_column = np.where(inside, column, 0)
        safe_row = np.where(inside, row, 0)
        values = self.clearance[safe_row, safe_column]
        return np.where(inside, values, 0.0)

    def is_free_cell(self, x, y):
        """해당 좌표의 셀이 지도 안이고 막혀 있지 않은지 본다."""
        column, row = self.world_to_grid(x, y)
        if not (0 <= column < self.width and 0 <= row < self.height):
            return False
        return not bool(self.blocked[row, column])


# ---------------------------------------------------------------------------
# 클로소이드 (G1 Hermite 보간)
# ---------------------------------------------------------------------------

_QUAD_NODES, _QUAD_WEIGHTS = np.polynomial.legendre.leggauss(24)
_QUAD_T = 0.5 * (_QUAD_NODES + 1.0)
_QUAD_W = 0.5 * _QUAD_WEIGHTS
_QUAD_SHAPE = _QUAD_T ** 2 - _QUAD_T


class Clothoid:
    """곡률이 호길이에 대해 선형으로 변하는 오일러 나선 한 조각."""

    def __init__(self, x, y, yaw, curvature, curvature_rate, length):
        """시작 pose와 곡률 파라미터로 클로소이드를 정의한다."""
        self.x = x
        self.y = y
        self.yaw = yaw
        self.curvature = curvature
        self.curvature_rate = curvature_rate
        self.length = length

    @property
    def peak_curvature(self):
        """구간 양 끝 중 더 큰 곡률 크기. 곡률이 선형이라 끝점만 보면 된다."""
        end_curvature = self.curvature + self.curvature_rate * self.length
        return max(abs(self.curvature), abs(end_curvature))

    def sample(self, spacing):
        """일정 간격으로 (x, y, yaw)를 뽑아 (N, 3) 배열로 돌려준다."""
        count = max(2, int(math.ceil(self.length / max(spacing, 1e-6))) + 1)
        arc = np.linspace(0.0, self.length, count)
        yaw = (
            self.yaw
            + self.curvature * arc
            + 0.5 * self.curvature_rate * arc ** 2
        )
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        # 누적 사다리꼴 적분. 표본 간격이 촘촘해 끝점 오차는 1e-4 m 수준이다.
        step = np.diff(arc)
        x = self.x + np.concatenate((
            [0.0], np.cumsum(0.5 * (cosine[:-1] + cosine[1:]) * step)
        ))
        y = self.y + np.concatenate((
            [0.0], np.cumsum(0.5 * (sine[:-1] + sine[1:]) * step)
        ))
        return np.column_stack((x, y, yaw))


def _clothoid_angle(sharpness, base):
    """
    구적점마다의 클로소이드 헤딩 (n, 24) 배열.

    ``theta0 + (delta - A) t + A t^2 == base + A (t^2 - t)`` 로 묶으면 A와
    무관한 부분(``base``)을 한 번만 계산하면 되고, 근을 찾는 반복에서는
    곱셈과 덧셈 한 번씩만 남는다. 이 함수가 탐색 전체의 가장 뜨거운 지점이라
    numpy 호출 수를 최소로 줄여 놓았다.
    """
    return base + np.asarray(sharpness)[:, np.newaxis] * _QUAD_SHAPE


def _clothoid_sine(sharpness, base):
    """도착점의 횡방향 오차에 해당하는 sin 적분. 근을 찾을 때 쓴다."""
    return np.sin(_clothoid_angle(sharpness, base)) @ _QUAD_W


def _clothoid_cosine(sharpness, base):
    """현(chord) 방향 성분에 해당하는 cos 적분. 길이를 낼 때 쓴다."""
    return np.cos(_clothoid_angle(sharpness, base)) @ _QUAD_W


def _refine_roots(low, high, base, iterations=14):
    """
    부호가 바뀌는 구간들을 이분법으로 동시에 좁힌다.

    반복 횟수는 A를 1e-5 수준까지 좁히면 충분해서 정했다. 이 오차가 경로에
    남기는 위치 오차는 마이크로미터 단위이고, 남는 잔차는 _connect가 끝점에
    맞춰 흡수한다.
    """
    low = np.array(low, dtype=np.float64)
    high = np.array(high, dtype=np.float64)
    sine_low = _clothoid_sine(low, base)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        sine_middle = _clothoid_sine(middle, base)
        same_side = np.signbit(sine_middle) == np.signbit(sine_low)
        low = np.where(same_side, middle, low)
        sine_low = np.where(same_side, sine_middle, sine_low)
        high = np.where(same_side, high, middle)
    return 0.5 * (low + high)


def fit_clothoid(start, goal, max_curvature=None, span=4.0 * math.pi,
                 samples=161):
    """
    두 pose를 하나의 클로소이드로 잇는 G1 Hermite 보간.

    Bertolazzi & Frego(2015)의 표준화를 따른다. 헤딩을
    ``theta(t) = theta0 + (delta - A) t + A t^2`` 로 두면 도착점의 횡방향
    오차가 0이 되는 조건이 ``A``에 대한 1차원 방정식이 되고, 그 근을 찾으면
    곡률과 길이가 바로 나온다.

    전진 방향으로만 이어지므로 (길이가 양수) 후진 연결은 두 pose의 yaw를
    각각 pi만큼 돌려서 호출하면 된다.

    맞는 클로소이드가 없거나 ``max_curvature``를 넘으면 None을 준다.
    """
    x0, y0, yaw0 = start
    x1, y1, yaw1 = goal
    delta_x = x1 - x0
    delta_y = y1 - y0
    chord = math.hypot(delta_x, delta_y)
    if chord < 1.0e-9:
        return None

    heading = math.atan2(delta_y, delta_x)
    theta0 = normalize_angle(yaw0 - heading)
    theta1 = normalize_angle(yaw1 - heading)
    delta = theta1 - theta0

    base = theta0 + delta * _QUAD_T
    grid = np.linspace(-span, span, samples)
    sine = _clothoid_sine(grid, base)
    # 근이 표본점 위에 정확히 걸리면 이분법이 무너지므로 따로 골라낸다.
    sign = np.sign(sine)
    sign[np.abs(sine) <= 1.0e-12] = 0.0
    exact = grid[sign == 0.0]
    brackets = np.flatnonzero(sign[:-1] * sign[1:] < 0.0)
    if len(brackets) == 0 and len(exact) == 0:
        return None

    roots = [exact]
    if len(brackets):
        roots.append(
            _refine_roots(grid[brackets], grid[brackets + 1], base)
        )
    roots = np.concatenate(roots)
    cosine = _clothoid_cosine(roots, base)

    best = None
    best_peak = math.inf
    for sharpness, cosine_value in zip(roots, cosine):
        if cosine_value <= 1.0e-9:
            continue  # 길이가 음수 -> 전진으로는 못 잇는 해
        length = chord / float(cosine_value)
        curvature = (delta - sharpness) / length
        curvature_rate = 2.0 * sharpness / (length * length)
        peak = max(abs(curvature), abs((delta + sharpness) / length))
        if max_curvature is not None and peak > max_curvature:
            continue
        if peak < best_peak:
            best_peak = peak
            best = Clothoid(
                x0, y0, yaw0, curvature, curvature_rate, length
            )
    return best


# ---------------------------------------------------------------------------
# 휴리스틱
# ---------------------------------------------------------------------------


class _GridHeuristic:
    """
    장애물은 보되 차량 기구학은 무시한 2D 거리 휴리스틱.

    원본 해상도(2 cm)로 Dijkstra를 돌리면 파이썬에서 너무 느리므로 격자를
    묶어서(기본 5배) 계산하고, 조회할 때 가장 가까운 굵은 셀 값을 쓴다.
    """

    def __init__(self, occupancy_map, target, radius, coarse_factor=5):
        """목적지에서 출발하는 8방향 Dijkstra 비용장을 만든다."""
        self.resolution = occupancy_map.resolution * coarse_factor
        self.origin_x = occupancy_map.origin_x
        self.origin_y = occupancy_map.origin_y
        self.target = target
        self.available = False

        height, width = occupancy_map.clearance.shape
        rows = height // coarse_factor
        columns = width // coarse_factor
        if rows < 2 or columns < 2:
            return

        trimmed = occupancy_map.clearance[
            :rows * coarse_factor, :columns * coarse_factor
        ]
        # 묶음 안의 최소 clearance를 대표값으로 써서 보수적으로 만든다.
        coarse = trimmed.reshape(
            rows, coarse_factor, columns, coarse_factor
        ).min(axis=(1, 3))
        passable = coarse >= radius

        self.rows = rows
        self.columns = columns
        self.cost = np.full((rows, columns), math.inf, dtype=np.float64)

        seed = self._seed_cell(passable, target)
        if seed is None:
            return

        neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        ]
        self.cost[seed] = 0.0
        queue = [(0.0, seed[0], seed[1])]
        while queue:
            distance, row, column = heapq.heappop(queue)
            if distance > self.cost[row, column]:
                continue
            for row_step, column_step, weight in neighbors:
                next_row = row + row_step
                next_column = column + column_step
                if not (0 <= next_row < rows and 0 <= next_column < columns):
                    continue
                if not passable[next_row, next_column]:
                    continue
                candidate = distance + weight * self.resolution
                if candidate < self.cost[next_row, next_column]:
                    self.cost[next_row, next_column] = candidate
                    heapq.heappush(queue, (candidate, next_row, next_column))
        self.available = True

    def _seed_cell(self, passable, target):
        """목표 셀이 막혀 있으면 주변에서 가장 가까운 통행 가능 셀을 찾는다."""
        column = int((target[0] - self.origin_x) // self.resolution)
        row = int((target[1] - self.origin_y) // self.resolution)
        if not (0 <= row < self.rows and 0 <= column < self.columns):
            return None
        if passable[row, column]:
            return (row, column)
        for ring in range(1, 8):
            for row_step in range(-ring, ring + 1):
                for column_step in range(-ring, ring + 1):
                    if max(abs(row_step), abs(column_step)) != ring:
                        continue
                    next_row = row + row_step
                    next_column = column + column_step
                    if not (0 <= next_row < self.rows
                            and 0 <= next_column < self.columns):
                        continue
                    if passable[next_row, next_column]:
                        return (next_row, next_column)
        return None

    def __call__(self, x, y):
        """Map frame 좌표에서 목적지까지의 하한 비용(m)을 돌려준다."""
        straight = math.hypot(x - self.target[0], y - self.target[1])
        if not self.available:
            return straight  # Dijkstra를 못 깔았으면 직선 거리로 물러선다
        column = int((x - self.origin_x) // self.resolution)
        row = int((y - self.origin_y) // self.resolution)
        if not (0 <= row < self.rows and 0 <= column < self.columns):
            return math.inf
        return max(straight, float(self.cost[row, column]))


# ---------------------------------------------------------------------------
# 양방향 Hybrid A*
# ---------------------------------------------------------------------------


class _Node:
    """트리의 한 상태. 부모로부터의 1스텝 정보를 함께 들고 있다."""

    __slots__ = (
        'x', 'y', 'yaw', 'steer', 'direction', 'arc', 'cost', 'parent',
        'phase', 'reverse_distance',
    )

    def __init__(self, x, y, yaw, steer, direction, arc, cost, parent,
                 phase=0, reverse_distance=0.0):
        """pose와 도달 비용, 그리고 부모에서 오는 호를 기록한다."""
        self.x = x
        self.y = y
        self.yaw = yaw
        self.steer = steer
        self.direction = direction
        self.arc = arc
        self.cost = cost
        self.parent = parent
        self.phase = phase
        self.reverse_distance = reverse_distance

    @property
    def pose(self):
        """(x, y, yaw) 튜플."""
        return (self.x, self.y, self.yaw)


class _Tree:
    """한쪽 방향의 탐색 트리."""

    def __init__(self, root, growth, heuristic):
        """루트 노드, 시간 진행 방향(+1/-1), 휴리스틱을 묶는다."""
        self.root = root
        self.growth = growth
        self.heuristic = heuristic
        self.nodes = {}
        self.closed = set()
        self.open = []
        self.counter = 0
        self.mate = None


class PlanResult:
    """플래너 실행 결과."""

    def __init__(self, success, poses=None, directions=None, reason='',
                 expansions=0, elapsed=0.0, cost=math.inf):
        """성공 여부와 경로, 통계를 담는다."""
        self.success = success
        self.poses = (
            np.zeros((0, 3)) if poses is None else np.asarray(poses)
        )
        self.directions = (
            np.zeros(0) if directions is None else np.asarray(directions)
        )
        self.reason = reason
        self.expansions = expansions
        self.elapsed = elapsed
        self.cost = cost

    @property
    def length(self):
        """경로의 총 주행 거리(m)."""
        if len(self.poses) < 2:
            return 0.0
        steps = np.diff(self.poses[:, :2], axis=0)
        return float(np.hypot(steps[:, 0], steps[:, 1]).sum())

    @property
    def gear_changes(self):
        """전/후진이 바뀌는 횟수."""
        if len(self.directions) < 2:
            return 0
        return int(np.count_nonzero(np.diff(self.directions) != 0))

    def as_tuples(self):
        """[(x, y, yaw, direction), ...] 형태로 변환한다."""
        return [
            (float(pose[0]), float(pose[1]), float(pose[2]), float(gear))
            for pose, gear in zip(self.poses, self.directions)
        ]

    def __repr__(self):
        """디버깅용 한 줄 요약."""
        if not self.success:
            return f'<PlanResult failed: {self.reason}>'
        return (
            f'<PlanResult {len(self.poses)} poses, {self.length:.2f} m, '
            f'gear changes {self.gear_changes}, '
            f'{self.expansions} expansions, {self.elapsed:.2f} s>'
        )


class HybridAStar:
    """SLAM 지도 위에서 출발/목적 양쪽으로 뻗어 global path를 찾는 플래너."""

    def __init__(self, occupancy_map):
        """지도를 물리고 탐색/비용 파라미터 기본값을 세운다."""
        self.map = occupancy_map

        # 차량 기구학. dwa.py와 같은 자전거 모델을 쓴다.
        self.wheel_base = WHEEL_BASE_M     # 축간 거리(m)
        self.max_steer = MAX_STEER_RAD     # 최대 조향각(rad)
        self.steer_count = 9               # 1스텝에서 시도할 조향 후보 수
        self.step = 0.25                   # 1스텝의 호 길이(m)
        self.substeps = 3                  # 한 스텝 안에서 충돌을 볼 표본 수
        self.allow_reverse = True          # 후진 스텝 허용 여부

        # 상태 격자. 같은 칸에 들어온 상태 중 싼 것만 남긴다.
        self.xy_resolution = 0.15          # 상태 중복 판정용 위치 격자(m)
        self.yaw_resolution = math.radians(5.0)  # 상태 중복 판정용 각도 격자

        # 비용 가중치. 전부 "작을수록 좋은 비용"이라 부호가 모두 양수다.
        self.w_length = 1.0        # 주행 거리
        self.w_steer = 0.35        # 조향 크기 (dwa.py의 w1 heading 항 대응)
        self.w_steer_change = 0.5  # 조향 변화량 -> 경로를 매끄럽게
        self.w_obstacle = 1.5      # 장애물 근접 페널티 (dwa.py의 w2/w3 대응)
        self.w_reverse = 1.7       # 후진 1 m당 추가 비용
        self.w_gear_change = 2.0   # 전/후진 전환 1회 비용
        self.safe_dist = 0.3      # 이 여유보다 가까우면 페널티 (dwa.py와 동일)
        self.collision_margin = 0.01  # 충돌 판정에 더 얹는 여유(m)

        # 탐색 제어.
        self.heuristic_weight = 1.4     # >1이면 weighted A*, 빠르지만 준최적
        self.heuristic_coarse_factor = 5  # 휴리스틱 격자를 몇 배로 묶을지
        self.max_expansions = 40000     # 확장 상한
        self.time_limit = 10.0          # 탐색 시간 상한(s)

        # 클로소이드 연결.
        self.connect_interval = 5       # 몇 번 확장마다 반대편 루트로 쏴볼지
        self.connect_radius = 2.0       # 이 거리 안이면 매번 시도(m)
        self.connect_max_length = 5.0   # 연결 클로소이드 최대 길이(m)
        self.connect_candidates = 6     # 반대 트리에서 시도할 최대 노드 수
        self.connect_cell_span = 1      # 이웃 상태 칸 탐색 반경(칸)

        # 출력.
        self.output_spacing = 0.05      # 최종 경로 표본 간격(m)

        self._disc_offsets, self._disc_radius = footprint_discs()
        self._prepare()

    # -- 기본 도구 ---------------------------------------------------------

    def _prepare(self):
        """
        탐색 직전에 파라미터에서 파생되는 배열을 만들어 둔다.

        가중치나 조향 개수를 생성 뒤에 바꿔도 plan()이 그 값을 쓰도록,
        파생값 계산을 생성자에 묶지 않고 매 plan()마다 다시 한다.
        """
        self._required = self._disc_radius + self.collision_margin
        self._steers = np.asarray(self.steer_candidates(), dtype=np.float64)
        self._curvatures = np.tan(self._steers) / self.wheel_base
        self._gears = np.array(
            [1.0, -1.0] if self.allow_reverse else [1.0], dtype=np.float64
        )
        count = max(1, int(self.substeps))
        self._fractions = np.arange(1, count + 1, dtype=np.float64) / count

    @property
    def max_curvature(self):
        """조향 한계에서 나오는 최대 곡률(1/m)."""
        return math.tan(self.max_steer) / self.wheel_base

    def steer_candidates(self):
        """
        조향 후보를 균등하게 만든다. dwa.py dynamic_window의 전역 버전이다.

        DWA는 현재 조향 주변만 훑지만, global path는 매 스텝에서 전 범위를
        볼 수 있어야 좁은 주차 통로를 돌아나갈 수 있다. 대신 조향 변화량에
        비용(w_steer_change)을 매겨 경로가 튀는 것을 막는다.
        """
        if self.steer_count < 2:
            return [0.0]
        step = 2.0 * self.max_steer / (self.steer_count - 1)
        return [-self.max_steer + step * index
                for index in range(self.steer_count)]

    def _propagate(self, x, y, yaw, steer, arc):
        """자전거 모델을 호 길이 ``arc``만큼 정확히 적분한다. 음수면 후진."""
        if abs(steer) < 1.0e-6:
            return x + arc * math.cos(yaw), y + arc * math.sin(yaw), yaw
        curvature = math.tan(steer) / self.wheel_base
        next_yaw = yaw + curvature * arc
        next_x = x + (math.sin(next_yaw) - math.sin(yaw)) / curvature
        next_y = y - (math.cos(next_yaw) - math.cos(yaw)) / curvature
        return next_x, next_y, next_yaw

    def _state_index(self, x, y, yaw):
        """상태 중복 판정용 SE(2) 격자 인덱스."""
        return (
            int(math.floor(x / self.xy_resolution)),
            int(math.floor(y / self.xy_resolution)),
            int(math.floor(normalize_angle(yaw) / self.yaw_resolution)),
        )

    def _footprint_margin(self, x, y, yaw):
        """차체 표면에서 장애물까지 남는 여유(m). 음수면 충돌이다."""
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        centers_x = x + self._disc_offsets * cosine
        centers_y = y + self._disc_offsets * sine
        clearance = self.map.clearance_at(centers_x, centers_y)
        return float(clearance.min()) - self._required

    def _obstacle_penalty(self, margin):
        """DWA와 같은 방식으로 근접 페널티를 매긴다."""
        if margin >= self.safe_dist:
            return 0.0
        return 1.0 / max(margin, 0.02)

    def _path_is_free(self, poses):
        """표본 pose 배열이 전부 충돌 없는지 한 번에 본다."""
        cosine = np.cos(poses[:, 2])[:, np.newaxis]
        sine = np.sin(poses[:, 2])[:, np.newaxis]
        centers_x = poses[:, 0:1] + self._disc_offsets * cosine
        centers_y = poses[:, 1:2] + self._disc_offsets * sine
        clearance = self.map.clearance_at(centers_x, centers_y)
        return bool((clearance >= self._required).all())

    # -- 탐색 --------------------------------------------------------------

    def plan(self, start, goal):
        """
        start에서 goal까지의 global path를 찾는다.

        ``start``/``goal``은 map frame의 (x, y, yaw)다. 반환값은 PlanResult로,
        ``poses``가 (N, 3) 배열, ``directions[i]``가 pose i에서 pose i+1로 갈 때
        써야 할 기어(+1 전진, -1 후진)다. 마지막 원소는 직전 값을 반복한다.
        """
        started_at = time.monotonic()
        self._prepare()

        for name, pose in (('start', start), ('goal', goal)):
            if self._footprint_margin(*pose) < 0.0:
                return PlanResult(
                    False,
                    reason=f'{name} pose is in collision or outside the map',
                    elapsed=time.monotonic() - started_at,
                )

        # 두 pose가 이미 한 번에 이어지면 탐색할 것도 없다.
        direct = self._connect(start, goal)
        if direct is not None:
            poses, directions = direct
            steps = np.diff(poses[:, :2], axis=0)
            return PlanResult(
                True, poses, directions,
                reason='single clothoid',
                elapsed=time.monotonic() - started_at,
                cost=float(np.hypot(steps[:, 0], steps[:, 1]).sum()),
            )

        radius = self._disc_radius + self.collision_margin
        forward_tree = _Tree(
            _Node(start[0], start[1], start[2], 0.0, 1.0, 0.0, 0.0, None),
            1.0,
            _GridHeuristic(
                self.map, goal, radius, self.heuristic_coarse_factor
            ),
        )
        reverse_tree = _Tree(
            _Node(goal[0], goal[1], goal[2], 0.0, 1.0, 0.0, 0.0, None),
            -1.0,
            _GridHeuristic(
                self.map, start, radius, self.heuristic_coarse_factor
            ),
        )
        forward_tree.mate = reverse_tree
        reverse_tree.mate = forward_tree
        for tree in (forward_tree, reverse_tree):
            index = self._state_index(*tree.root.pose)
            tree.nodes[index] = tree.root
            heapq.heappush(tree.open, (0.0, 0, index))

        expansions = 0
        deadline = started_at + self.time_limit
        while forward_tree.open and reverse_tree.open:
            if expansions >= self.max_expansions:
                return PlanResult(
                    False, reason='expansion limit reached',
                    expansions=expansions,
                    elapsed=time.monotonic() - started_at,
                )
            if time.monotonic() > deadline:
                return PlanResult(
                    False, reason='time limit reached',
                    expansions=expansions,
                    elapsed=time.monotonic() - started_at,
                )

            # 열린 목록의 f값이 작은 쪽을 번갈아 키운다.
            tree = (
                forward_tree
                if forward_tree.open[0][0] <= reverse_tree.open[0][0]
                else reverse_tree
            )
            _, _, index = heapq.heappop(tree.open)
            if index in tree.closed:
                continue
            node = tree.nodes.get(index)
            if node is None:
                continue
            tree.closed.add(index)

            expansions += 1
            joined = self._try_join(tree, node, expansions)
            if joined is not None:
                poses, directions, cost = joined
                return PlanResult(
                    True, poses, directions,
                    reason='clothoid join',
                    expansions=expansions,
                    elapsed=time.monotonic() - started_at,
                    cost=cost,
                )
            self._expand(tree, node)

        return PlanResult(
            False, reason='search space exhausted',
            expansions=expansions,
            elapsed=time.monotonic() - started_at,
        )

    def plan_forward_then_final_reverse(
            self,
            start,
            goal,
            reverse_activation_radius=1.2,
            max_reverse_length=0.8):
        """
        Drive forward until the goal vicinity and permit one final reverse.

        This is deliberately separate from :meth:`plan`, which preserves the
        original bidirectional Hybrid A* behavior. Phase 0 expands forward
        primitives only. Once a node is inside ``reverse_activation_radius``,
        it may switch to phase 1; phase 1 expands reverse primitives only and
        cannot switch back to forward. The total reverse distance is bounded
        by ``max_reverse_length``.
        """
        started_at = time.monotonic()
        self._prepare()
        if reverse_activation_radius <= 0.0:
            return PlanResult(
                False,
                reason='reverse_activation_radius must be positive',
            )
        if max_reverse_length <= 0.0:
            return PlanResult(
                False,
                reason='max_reverse_length must be positive',
            )

        for name, pose in (('start', start), ('goal', goal)):
            if self._footprint_margin(*pose) < 0.0:
                return PlanResult(
                    False,
                    reason=f'{name} pose is in collision or outside the map',
                    elapsed=time.monotonic() - started_at,
                )

        radius = self._disc_radius + self.collision_margin
        heuristic = _GridHeuristic(
            self.map, goal, radius, self.heuristic_coarse_factor
        )
        root = _Node(
            start[0], start[1], start[2], 0.0, 1.0, 0.0, 0.0, None,
            phase=0, reverse_distance=0.0,
        )
        nodes = {}
        closed = set()
        open_nodes = []
        counter = 0
        root_index = (*self._state_index(*root.pose), root.phase, 0)
        nodes[root_index] = root
        heapq.heappush(open_nodes, (0.0, counter, root_index))

        expansions = 0
        deadline = started_at + self.time_limit
        while open_nodes:
            if expansions >= self.max_expansions:
                return PlanResult(
                    False,
                    reason='expansion limit reached',
                    expansions=expansions,
                    elapsed=time.monotonic() - started_at,
                )
            if time.monotonic() > deadline:
                return PlanResult(
                    False,
                    reason='time limit reached',
                    expansions=expansions,
                    elapsed=time.monotonic() - started_at,
                )

            _, _, index = heapq.heappop(open_nodes)
            if index in closed:
                continue
            node = nodes.get(index)
            if node is None:
                continue
            closed.add(index)
            expansions += 1

            joined = self._try_final_mode_connection(
                node,
                goal,
                reverse_activation_radius,
                max_reverse_length,
            )
            if joined is not None:
                poses, directions, reason = joined
                steps = np.diff(poses[:, :2], axis=0)
                return PlanResult(
                    True,
                    poses,
                    directions,
                    reason=reason,
                    expansions=expansions,
                    elapsed=time.monotonic() - started_at,
                    cost=float(np.hypot(steps[:, 0], steps[:, 1]).sum()),
                )

            end_x, end_y, end_yaw, worst = self._sweep(node, 1.0)
            goal_distance = math.hypot(node.x - goal[0], node.y - goal[1])
            for gear_index, direction in enumerate(self._gears):
                direction = float(direction)
                if node.phase == 1 and direction > 0.0:
                    continue
                if (
                        node.phase == 0
                        and direction < 0.0
                        and goal_distance > reverse_activation_radius):
                    continue

                phase = 1 if direction < 0.0 else node.phase
                reverse_distance = node.reverse_distance
                if direction < 0.0:
                    reverse_distance += self.step
                    if reverse_distance > max_reverse_length + 1.0e-9:
                        continue

                arc = direction * self.step
                for steer_index, steer in enumerate(self._steers):
                    margin = float(worst[gear_index, steer_index])
                    if margin < 0.0:
                        continue

                    next_x = float(end_x[gear_index, steer_index])
                    next_y = float(end_y[gear_index, steer_index])
                    next_yaw = normalize_angle(
                        float(end_yaw[gear_index, steer_index])
                    )
                    reverse_step = (
                        int(round(reverse_distance / self.step))
                        if phase == 1 else 0
                    )
                    next_index = (
                        *self._state_index(next_x, next_y, next_yaw),
                        phase,
                        reverse_step,
                    )
                    if next_index in closed:
                        continue

                    cost = node.cost + self._step_cost(
                        node, float(steer), direction, margin
                    )
                    existing = nodes.get(next_index)
                    if existing is not None and existing.cost <= cost:
                        continue

                    estimate = heuristic(next_x, next_y)
                    if math.isinf(estimate):
                        continue
                    child = _Node(
                        next_x,
                        next_y,
                        next_yaw,
                        float(steer),
                        direction,
                        arc,
                        cost,
                        node,
                        phase=phase,
                        reverse_distance=reverse_distance,
                    )
                    nodes[next_index] = child
                    counter += 1
                    heapq.heappush(
                        open_nodes,
                        (
                            cost + self.heuristic_weight * estimate,
                            counter,
                            next_index,
                        ),
                    )

        return PlanResult(
            False,
            reason='search space exhausted',
            expansions=expansions,
            elapsed=time.monotonic() - started_at,
        )

    def _try_final_mode_connection(
            self,
            node,
            goal,
            reverse_activation_radius,
            max_reverse_length):
        """Inside the goal vicinity, try forward then a bounded reverse."""
        distance = math.hypot(node.x - goal[0], node.y - goal[1])
        attempt_forward = node.phase == 0 and (
            distance <= reverse_activation_radius
        )
        if attempt_forward:
            bridge = self._connect_direction(
                node.pose,
                goal,
                1.0,
                max_length=reverse_activation_radius,
            )
            if bridge is not None:
                poses, directions = self._assemble_single_tree(
                    node, *bridge
                )
                return poses, directions, 'forward-only finish'

        reverse_remaining = max_reverse_length - node.reverse_distance
        if (
                self.allow_reverse
                and distance <= reverse_activation_radius
                and reverse_remaining > 0.0):
            bridge = self._connect_direction(
                node.pose,
                goal,
                -1.0,
                max_length=reverse_remaining,
            )
            if bridge is not None:
                poses, directions = self._assemble_single_tree(
                    node, *bridge
                )
                return poses, directions, 'forward then final reverse'
        return None

    def _sweep(self, node, growth):
        """
        조향 x 기어 후보를 한꺼번에 1스텝 뻗는다.

        DWA가 후보 경로를 한 번에 펼쳐 보는 것과 같은 발상인데, 여기서는
        중간 표본의 발자국 충돌까지 numpy 한 번에 계산한다. 파이썬 반복문으로
        하면 확장 하나에 수십 번씩 지도를 조회하게 되어 너무 느리다.

        (기어, 조향)마다의 도착 pose와 구간 최소 여유를 돌려준다.
        """
        arcs = self._gears * (self.step * growth)
        partial = arcs[:, np.newaxis, np.newaxis] * self._fractions
        curvature = self._curvatures[np.newaxis, :, np.newaxis]
        straight = np.abs(curvature) < 1.0e-6
        divisor = np.where(straight, 1.0, curvature)

        cos_yaw = math.cos(node.yaw)
        sin_yaw = math.sin(node.yaw)
        yaw = node.yaw + curvature * partial
        x = node.x + np.where(
            straight, partial * cos_yaw, (np.sin(yaw) - sin_yaw) / divisor
        )
        y = node.y + np.where(
            straight, partial * sin_yaw, -(np.cos(yaw) - cos_yaw) / divisor
        )

        centers_x = (
            x[..., np.newaxis] + self._disc_offsets * np.cos(yaw)[..., None]
        )
        centers_y = (
            y[..., np.newaxis] + self._disc_offsets * np.sin(yaw)[..., None]
        )
        margins = (
            self.map.clearance_at(centers_x, centers_y) - self._required
        )
        return x[:, :, -1], y[:, :, -1], yaw[:, :, -1], margins.min(
            axis=(2, 3)
        )

    def _expand(self, tree, node):
        """한 노드에서 뻗은 후보들을 비용순으로 트리에 넣는다."""
        end_x, end_y, end_yaw, worst = self._sweep(node, tree.growth)
        for gear_index, direction in enumerate(self._gears):
            arc = float(direction) * self.step * tree.growth
            for steer_index, steer in enumerate(self._steers):
                margin = float(worst[gear_index, steer_index])
                if margin < 0.0:
                    continue  # 구간 어딘가에서 차체가 장애물에 닿는다

                next_x = float(end_x[gear_index, steer_index])
                next_y = float(end_y[gear_index, steer_index])
                next_yaw = float(end_yaw[gear_index, steer_index])
                index = self._state_index(next_x, next_y, next_yaw)
                if index in tree.closed:
                    continue

                cost = node.cost + self._step_cost(
                    node, float(steer), float(direction), margin
                )
                existing = tree.nodes.get(index)
                if existing is not None and existing.cost <= cost:
                    continue

                estimate = tree.heuristic(next_x, next_y)
                if math.isinf(estimate):
                    continue  # 반대편 루트에서 도달할 수 없는 영역

                child = _Node(
                    next_x, next_y, normalize_angle(next_yaw), float(steer),
                    float(direction), arc, cost, node,
                )
                tree.nodes[index] = child
                tree.counter += 1
                heapq.heappush(
                    tree.open,
                    (cost + self.heuristic_weight * estimate,
                     tree.counter, index),
                )

    def _step_cost(self, node, steer, direction, margin):
        """DWA와 같은 항들로 1스텝 비용을 매긴다."""
        cost = self.w_length * self.step
        cost += self.w_steer * abs(steer) * self.step
        cost += self.w_steer_change * abs(steer - node.steer)
        cost += self.w_obstacle * self._obstacle_penalty(margin) * self.step
        if direction < 0.0:
            cost += self.w_reverse * self.step
        if node.parent is not None and direction != node.direction:
            cost += self.w_gear_change
        return cost

    # -- 연결 --------------------------------------------------------------

    def _try_join(self, tree, node, expansions):
        """방금 꺼낸 노드를 반대 트리와 클로소이드로 이어본다."""
        mate = tree.mate
        candidates = []

        # 1) 반대 트리의 루트로 바로 쏴보는 analytic expansion.
        distance = math.hypot(node.x - mate.root.x, node.y - mate.root.y)
        periodic = expansions % self.connect_interval == 0
        if distance <= self.connect_radius or periodic:
            candidates.append(mate.root)

        # 2) 반대 트리가 이미 와 있는 이웃 상태 칸들.
        base = self._state_index(node.x, node.y, node.yaw)
        span = self.connect_cell_span
        for x_step in range(-span, span + 1):
            for y_step in range(-span, span + 1):
                for yaw_step in range(-span, span + 1):
                    neighbor = mate.nodes.get(
                        (base[0] + x_step, base[1] + y_step,
                         base[2] + yaw_step)
                    )
                    if neighbor is not None:
                        candidates.append(neighbor)
        if len(candidates) > self.connect_candidates:
            candidates.sort(key=lambda item: item.cost)
            candidates = candidates[:self.connect_candidates]

        for other in candidates:
            # head는 항상 출발 트리 쪽, tail은 항상 목적 트리 쪽 노드다.
            head, tail = (
                (node, other) if tree.growth > 0.0 else (other, node)
            )
            bridge = self._connect(head.pose, tail.pose)
            if bridge is not None:
                return self._assemble(head, tail, *bridge)
        return None

    def _connect(self, start, goal):
        """
        두 pose를 클로소이드로 잇고 충돌까지 검사한다.

        전진 연결을 먼저 보고, 안 되면 (후진을 허용할 때) 두 헤딩을 pi만큼
        돌려 후진 연결을 본다. 반환값은 (poses, directions)다.
        """
        attempts = [1.0]
        if self.allow_reverse:
            attempts.append(-1.0)
        for direction in attempts:
            result = self._connect_direction(start, goal, direction)
            if result is not None:
                return result
        return None

    def _connect_direction(
            self, start, goal, direction, max_length=None):
        """Connect two poses with one collision-free forward or reverse arc."""
        if direction not in {-1.0, 1.0}:
            raise ValueError('direction must be +1.0 or -1.0')
        length_limit = self.connect_max_length
        if max_length is not None:
            length_limit = min(length_limit, float(max_length))

        # 클로소이드 길이는 두 점 사이 직선 거리보다 짧을 수 없다. 탐색 중
        # 대부분의 시도가 여기서 걸러지므로 값비싼 근 찾기를 아낄 수 있다.
        chord = math.hypot(goal[0] - start[0], goal[1] - start[1])
        if chord > length_limit:
            return None

        head = start
        tail = goal
        if direction < 0.0:
            head = (start[0], start[1], start[2] + math.pi)
            tail = (goal[0], goal[1], goal[2] + math.pi)
        clothoid = fit_clothoid(
            head, tail, max_curvature=self.max_curvature
        )
        if clothoid is None or clothoid.length > length_limit:
            return None

        poses = clothoid.sample(self.output_spacing)
        # 적분 잔차를 끝점에 맞춰 선형으로 흡수시킨다.
        residual = np.asarray(tail[:2]) - poses[-1, :2]
        if np.hypot(*residual) > 1.0e-6:
            weights = np.linspace(0.0, 1.0, len(poses))[:, np.newaxis]
            poses[:, :2] += weights * residual
        if direction < 0.0:
            poses[:, 2] -= math.pi  # 후진이면 차체 헤딩은 접선의 반대
        if not self._path_is_free(poses):
            return None
        directions = np.full(len(poses), direction)
        return poses, directions

    # -- 경로 복원 ---------------------------------------------------------

    def _segment_poses(self, from_pose, steer, arc):
        """한 스텝 구간을 촘촘히 표본화한다(시작점 제외)."""
        count = max(
            2, int(math.ceil(abs(arc) / max(self.output_spacing, 1e-6))) + 1
        )
        poses = []
        for index in range(1, count):
            fraction = arc * index / (count - 1)
            x, y, yaw = self._propagate(*from_pose, steer, fraction)
            poses.append((x, y, normalize_angle(yaw)))
        return poses

    def _chain_to_root(self, node):
        """노드에서 루트까지 부모를 따라 올라간 목록(노드 -> 루트 순)."""
        chain = []
        current = node
        while current is not None:
            chain.append(current)
            current = current.parent
        return chain

    def _assemble(self, head, tail, bridge_poses, bridge_directions):
        """
        출발 트리 구간 + 클로소이드 + 목적 트리 구간을 이어 붙인다.

        ``head``는 출발 트리의 노드, ``tail``은 목적 트리의 노드다. 목적 트리는
        시간을 거꾸로 뻗었으므로, 노드에서 부모로 가는 방향이 실제 진행 방향이
        된다.
        """
        # 출발 트리: 루트 -> head 순으로 되돌려 앞에 붙인다.
        forward_chain = self._chain_to_root(head)[::-1]
        poses = [forward_chain[0].pose]
        directions = [1.0]
        for previous, current in zip(forward_chain, forward_chain[1:]):
            segment = self._segment_poses(
                previous.pose, current.steer, current.arc
            )
            poses.extend(segment)
            directions.extend([current.direction] * len(segment))

        # 클로소이드 다리. 첫 점은 head와 겹치므로 버린다.
        poses.extend(tuple(pose) for pose in bridge_poses[1:])
        directions.extend(bridge_directions[1:].tolist())

        # 목적 트리: tail -> 루트(goal) 순. 마지막 원소가 goal 루트다.
        reverse_chain = self._chain_to_root(tail)
        for current in reverse_chain[:-1]:
            segment = self._segment_poses(
                current.pose, current.steer, -current.arc
            )
            poses.extend(segment)
            directions.extend([current.direction] * len(segment))

        poses = np.asarray(poses, dtype=np.float64)
        # 여기까지 directions[i]는 "pose i에 도착할 때"의 기어다. 경로 추종
        # 쪽이 쓰기 좋게 "pose i에서 떠날 때"의 기어로 한 칸 당긴다.
        directions = np.asarray(directions[1:] + directions[-1:],
                                dtype=np.float64)
        poses[:, 2] = np.arctan2(np.sin(poses[:, 2]), np.cos(poses[:, 2]))
        steps = np.diff(poses[:, :2], axis=0)
        cost = float(np.hypot(steps[:, 0], steps[:, 1]).sum())
        return poses, directions, cost

    def _assemble_single_tree(self, node, bridge_poses, bridge_directions):
        """Join a start-rooted primitive chain to one final clothoid."""
        chain = self._chain_to_root(node)[::-1]
        poses = [chain[0].pose]
        directions = [1.0]
        for previous, current in zip(chain, chain[1:]):
            segment = self._segment_poses(
                previous.pose, current.steer, current.arc
            )
            poses.extend(segment)
            directions.extend([current.direction] * len(segment))

        # The first bridge pose duplicates the search node.
        poses.extend(tuple(pose) for pose in bridge_poses[1:])
        directions.extend(bridge_directions[1:].tolist())
        poses = np.asarray(poses, dtype=np.float64)
        directions = np.asarray(
            directions[1:] + directions[-1:], dtype=np.float64
        )
        poses[:, 2] = np.arctan2(np.sin(poses[:, 2]), np.cos(poses[:, 2]))
        return poses, directions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_pose(text):
    """'x,y,yaw_deg' 문자열을 (x, y, yaw_rad)로 바꾼다."""
    parts = [item.strip() for item in text.split(',')]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f'pose must be "x,y,yaw_deg", got {text!r}'
        )
    return (
        float(parts[0]), float(parts[1]), math.radians(float(parts[2]))
    )


def write_preview_ppm(occupancy_map, result, path):
    """지도 위에 경로를 그린 PPM(P6) 미리보기를 저장한다. 외부 의존성 없음."""
    height, width = occupancy_map.occupancy.shape
    image = np.full((height, width, 3), 205, dtype=np.uint8)
    image[occupancy_map.occupancy == 0] = 254
    image[occupancy_map.occupancy >= 65] = 0

    if len(result.poses):
        columns, rows = occupancy_map.world_to_grid(
            result.poses[:, 0], result.poses[:, 1]
        )
        inside = (
            (columns >= 0) & (columns < width)
            & (rows >= 0) & (rows < height)
        )
        columns = columns[inside]
        rows = rows[inside]
        forward = result.directions[inside] > 0.0
        image[rows[forward], columns[forward]] = (220, 30, 30)   # 전진: 빨강
        image[rows[~forward], columns[~forward]] = (30, 90, 220)  # 후진: 파랑
        for point, color in (
                (result.poses[0], (0, 200, 0)),
                (result.poses[-1], (200, 0, 200))):
            column, row = occupancy_map.world_to_grid(point[0], point[1])
            block_rows = slice(max(0, row - 2), min(height, row + 3))
            block_columns = slice(max(0, column - 2), min(width, column + 3))
            image[block_rows, block_columns] = color

    header = f'P6\n{width} {height}\n255\n'.encode('ascii')
    with open(path, 'wb') as preview:
        preview.write(header)
        preview.write(image[::-1, :, :].tobytes())  # 위쪽이 최대 y가 되게 뒤집기


def main(argv=None):
    """지도를 읽어 global path를 뽑는 명령행 진입점."""
    parser = argparse.ArgumentParser(
        description='SLAM 지도 위에서 양방향 Hybrid A* global path를 뽑는다.'
    )
    parser.add_argument(
        '--map', required=True, help='Nav2 지도 yaml 경로'
    )
    parser.add_argument(
        '--start', required=True, type=_parse_pose,
        help='출발 pose "x,y,yaw_deg" (map frame)',
    )
    parser.add_argument(
        '--goal', required=True, type=_parse_pose,
        help='목적 pose "x,y,yaw_deg" (map frame)',
    )
    parser.add_argument(
        '--free-thresh', type=float, default=None,
        help='yaml의 free_thresh를 덮어쓴다. 미지 영역을 막으려면 0.19',
    )
    parser.add_argument(
        '--no-reverse', action='store_true', help='후진 스텝을 금지한다'
    )
    parser.add_argument(
        '--step', type=float, default=None, help='1스텝 호 길이(m)'
    )
    parser.add_argument(
        '--time-limit', type=float, default=None, help='탐색 시간 상한(s)'
    )
    parser.add_argument('--out', default=None, help='경로 CSV 저장 경로')
    parser.add_argument(
        '--preview', default=None, help='경로를 그린 PPM 미리보기 저장 경로'
    )
    arguments = parser.parse_args(argv)

    occupancy_map = OccupancyMap.from_nav2_yaml(
        arguments.map, free_thresh=arguments.free_thresh
    )
    max_x = (
        occupancy_map.origin_x
        + occupancy_map.width * occupancy_map.resolution
    )
    max_y = (
        occupancy_map.origin_y
        + occupancy_map.height * occupancy_map.resolution
    )
    print(
        f'map {occupancy_map.width}x{occupancy_map.height} @ '
        f'{occupancy_map.resolution:.3f} m, '
        f'x[{occupancy_map.origin_x:.2f}, {max_x:.2f}] '
        f'y[{occupancy_map.origin_y:.2f}, {max_y:.2f}], '
        f'blocked {occupancy_map.blocked.mean() * 100.0:.1f}%'
    )

    planner = HybridAStar(occupancy_map)
    if arguments.no_reverse:
        planner.allow_reverse = False
    if arguments.step is not None:
        planner.step = arguments.step
    if arguments.time_limit is not None:
        planner.time_limit = arguments.time_limit

    result = planner.plan(arguments.start, arguments.goal)
    print(result)
    if not result.success:
        return 1

    if arguments.out:
        np.savetxt(
            arguments.out,
            np.column_stack((result.poses, result.directions)),
            delimiter=',',
            header='x,y,yaw,direction',
            comments='',
            fmt='%.6f',
        )
        print(f'path saved to {arguments.out}')
    if arguments.preview:
        write_preview_ppm(occupancy_map, result, arguments.preview)
        print(f'preview saved to {arguments.preview}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
