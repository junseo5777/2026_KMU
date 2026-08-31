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

"""Low-speed reverse path tracking with Shin Stanley Pursuit (SSP)."""

import math
from dataclasses import dataclass

import numpy as np


def normalize_angle(angle):
    """Wrap one angle to ``[-pi, pi]``."""
    return math.atan2(math.sin(angle), math.cos(angle))


def direction_segment_start(directions, index):
    """Return the first pose index of the gear segment containing ``index``."""
    directions = np.asarray(directions, dtype=np.int8).reshape(-1)
    if len(directions) == 0:
        raise ValueError('directions cannot be empty')
    index = int(np.clip(index, 0, len(directions) - 1))
    direction = 1 if directions[index] >= 0 else -1
    start = index
    while start > 0:
        previous = 1 if directions[start - 1] >= 0 else -1
        if previous != direction:
            break
        start -= 1
    return start


@dataclass(frozen=True)
class ReverseSSPResult:
    """One reverse SSP calculation in the map frame."""

    steer: float
    heading_error: float
    cross_track_error: float
    reference: np.ndarray
    rear_projection: np.ndarray
    virtual_front: np.ndarray
    front_projection: np.ndarray

    @property
    def tracking_point_at_endpoint(self):
        """Return whether the virtual-front projection reached the tail end."""
        endpoint = np.asarray(self.reference[-1, :2], dtype=np.float64)
        projection = np.asarray(self.front_projection, dtype=np.float64)
        return bool(np.allclose(
            projection, endpoint, rtol=0.0, atol=1.0e-9
        ))


class ReverseSSP:
    """
    Track one Hybrid A* reverse segment without DWA obstacle sampling.

    Reverse motion is represented as a virtual forward bicycle.  Its rear axle
    is the real vehicle's rear axle (``base_link``), its heading is body yaw +
    pi, and its front axle lies one wheelbase farther in the reverse direction.
    The paper's SSP structure uses path-heading error at the virtual rear axle
    and cross-track error at the virtual front axle.  The resulting virtual
    steering is negated to obtain the real reverse steering.

    The constants below are authoritative for this controller.  Motor command
    4 is assumed to be a very slow reverse, approximately 0.2 m/s.  The
    effective speed is used only in the Stanley denominator; it does not alter
    the motor command sent by the follower.
    """

    def __init__(self):
        """Set vehicle geometry and very-low-speed SSP tuning."""
        self.wheel_base = 0.345
        self.max_steer = math.radians(27.0)

        # Very-low-speed SSP tuning from the offline closed-loop comparison.
        self.gain = 1.17
        self.effective_speed = 0.20
        self.speed_softening = 0.1

        # Continue the final path tangent so endpoint tracking remains defined.
        self.endpoint_extension = 0.45
        self.preview_distance = 0.50
        self.preview_spacing = 0.05
        self.validate()

    def validate(self):
        """Reject invalid geometry or low-speed tuning values."""
        positive = {
            'wheel_base': self.wheel_base,
            'max_steer': self.max_steer,
            'gain': self.gain,
            'effective_speed': self.effective_speed,
            'speed_softening': self.speed_softening,
            'endpoint_extension': self.endpoint_extension,
            'preview_distance': self.preview_distance,
            'preview_spacing': self.preview_spacing,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            names = ', '.join(invalid)
            raise ValueError(
                f'reverse SSP parameters must be positive: {names}'
            )
        if self.max_steer >= math.pi / 2.0:
            raise ValueError('max_steer must be less than pi/2')

    def _extended_reference(self, reference):
        """Append one pose along the final reverse-motion tangent."""
        reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
        if len(reference) == 0:
            raise ValueError('reverse reference cannot be empty')
        endpoint = reference[-1]
        motion_yaw = normalize_angle(endpoint[2] + math.pi)
        extension = np.array([
            endpoint[0] + self.endpoint_extension * math.cos(motion_yaw),
            endpoint[1] + self.endpoint_extension * math.sin(motion_yaw),
            endpoint[2],
        ])
        return np.vstack((reference, extension))

    @staticmethod
    def _project(point, reference):
        """Project a map point and interpolate the reverse path heading."""
        point = np.asarray(point, dtype=np.float64).reshape(2)
        reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
        if len(reference) < 2:
            raise ValueError('reverse reference needs at least two poses')

        starts = reference[:-1, :2]
        vectors = reference[1:, :2] - starts
        lengths_squared = np.sum(vectors * vectors, axis=1)
        valid = lengths_squared > 1.0e-12
        if not np.any(valid):
            body_yaw = float(reference[0, 2])
            return reference[0, :2].copy(), normalize_angle(
                body_yaw + math.pi
            )

        factors = np.zeros(len(vectors), dtype=np.float64)
        factors[valid] = (
            np.sum((point - starts[valid]) * vectors[valid], axis=1)
            / lengths_squared[valid]
        )
        factors = np.clip(factors, 0.0, 1.0)
        projections = starts + factors[:, None] * vectors
        distances_squared = np.sum((point - projections) ** 2, axis=1)
        distances_squared[~valid] = math.inf
        index = int(np.argmin(distances_squared))

        yaw_delta = normalize_angle(
            float(reference[index + 1, 2] - reference[index, 2])
        )
        body_yaw = normalize_angle(
            float(reference[index, 2]) + factors[index] * yaw_delta
        )
        return projections[index], normalize_angle(body_yaw + math.pi)

    def track(self, reference, vehicle_pose):
        """Return the reverse SSP steering angle for one map-frame pose."""
        self.validate()
        vehicle_pose = np.asarray(vehicle_pose, dtype=np.float64).reshape(3)
        extended = self._extended_reference(reference)

        rear = vehicle_pose[:2]
        virtual_yaw = normalize_angle(float(vehicle_pose[2]) + math.pi)
        rear_projection, rear_path_yaw = self._project(rear, extended)
        heading_error = normalize_angle(rear_path_yaw - virtual_yaw)

        virtual_front = rear + self.wheel_base * np.array([
            math.cos(virtual_yaw), math.sin(virtual_yaw)
        ])
        front_projection, front_path_yaw = self._project(
            virtual_front, extended
        )
        left_normal = np.array([
            -math.sin(front_path_yaw), math.cos(front_path_yaw)
        ])
        cross_track_error = float(np.dot(
            virtual_front - front_projection, left_normal
        ))

        virtual_steer = (
            heading_error
            - math.atan2(
                self.gain * cross_track_error,
                self.effective_speed + self.speed_softening,
            )
        )
        reverse_steer = float(np.clip(
            -virtual_steer, -self.max_steer, self.max_steer
        ))
        return ReverseSSPResult(
            reverse_steer,
            heading_error,
            cross_track_error,
            extended,
            rear_projection,
            virtual_front,
            front_projection,
        )

    def predict(self, steer, distance=None):
        """Predict one constant-steer reverse path for RViz only."""
        if distance is None:
            distance = self.preview_distance
        distance = max(float(distance), self.preview_spacing)
        count = max(2, int(math.ceil(distance / self.preview_spacing)) + 1)
        arc = -np.linspace(0.0, distance, count)
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
