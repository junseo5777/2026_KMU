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

"""Small, dependency-light helpers for planar LiDAR odometry."""

import math

import numpy as np
from scipy.spatial import cKDTree


class SubmapIndex:
    """Cache exact nearest-neighbor data and surface normals for a submap."""

    def __init__(
            self,
            points,
            normal_neighbors=8,
            normal_radius=0.40,
            normal_min_neighbors=5,
            normal_min_linearity=0.80,
            compute_normals=True):
        """Build the search tree and normals once for immutable points."""
        self.points = np.asarray(points, dtype=np.float64).copy()
        if self.points.size == 0:
            self.points = self.points.reshape((0, 2))
        if self.points.ndim != 2 or self.points.shape[1:] != (2,):
            raise ValueError('submap points must have shape (N, 2)')

        self.tree = cKDTree(self.points) if len(self.points) else None
        if compute_normals:
            (
                self.normals,
                self.normal_valid,
                self.linearity,
            ) = estimate_normals(
                self.points,
                neighbor_count=normal_neighbors,
                search_radius=normal_radius,
                min_neighbors=normal_min_neighbors,
                min_linearity=normal_min_linearity,
                tree=self.tree,
            )
        else:
            self.normals = np.zeros_like(self.points)
            self.normal_valid = np.zeros(len(self.points), dtype=bool)
            self.linearity = np.zeros(len(self.points), dtype=np.float64)

    def query(self, query_points, distance_upper_bound=math.inf):
        """Return exact neighbor indices and squared distances."""
        query_points = np.asarray(query_points, dtype=np.float64)
        if len(query_points) == 0 or self.tree is None:
            return (
                np.full(len(query_points), -1, dtype=np.int64),
                np.full(len(query_points), math.inf, dtype=np.float64),
            )

        distances, indices = self.tree.query(
            query_points,
            k=1,
            eps=0.0,
            distance_upper_bound=distance_upper_bound,
            workers=1,
        )
        indices = np.asarray(indices, dtype=np.int64)
        distances = np.asarray(distances, dtype=np.float64)
        invalid = ~np.isfinite(distances) | (indices >= len(self.points))
        indices[invalid] = -1
        squared_distances = distances * distances
        squared_distances[invalid] = math.inf
        return indices, squared_distances


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def adaptive_imu_yaw_weight(
        inlier_ratio,
        rmse,
        minimum_inlier_ratio,
        good_inlier_ratio,
        good_rmse,
        maximum_rmse,
        nominal_weight,
        low_quality_weight):
    """Increase gyro trust as LiDAR scan-matching quality degrades."""
    if not 0.0 <= nominal_weight <= low_quality_weight <= 1.0:
        raise ValueError(
            'yaw weights must satisfy 0 <= nominal <= low_quality <= 1'
        )
    if good_inlier_ratio <= minimum_inlier_ratio:
        raise ValueError(
            'good_inlier_ratio must exceed minimum_inlier_ratio'
        )
    if good_rmse >= maximum_rmse:
        raise ValueError('good_rmse must be less than maximum_rmse')

    if not math.isfinite(inlier_ratio) or not math.isfinite(rmse):
        return low_quality_weight
    inlier_confidence = (
        (inlier_ratio - minimum_inlier_ratio)
        / (good_inlier_ratio - minimum_inlier_ratio)
    )
    rmse_confidence = (
        (maximum_rmse - rmse)
        / (maximum_rmse - good_rmse)
    )
    lidar_confidence = min(
        max(inlier_confidence, 0.0),
        max(rmse_confidence, 0.0),
        1.0,
    )
    return (
        low_quality_weight
        - lidar_confidence * (low_quality_weight - nominal_weight)
    )


def fuse_relative_yaw(lidar_yaw_delta, imu_yaw_delta, imu_weight):
    """Blend relative headings across the pi boundary without discontinuity."""
    if not 0.0 <= imu_weight <= 1.0:
        raise ValueError('imu_weight must be in [0, 1]')
    disagreement = normalize_angle(imu_yaw_delta - lidar_yaw_delta)
    return normalize_angle(lidar_yaw_delta + imu_weight * disagreement)


def pose_matrix(x, y, yaw):
    """Return the homogeneous SE(2) matrix for a planar pose."""
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array([
        [cosine, -sine, x],
        [sine, cosine, y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def matrix_pose(transform):
    """Return ``(x, y, yaw)`` from an SE(2) matrix."""
    return (
        float(transform[0, 2]),
        float(transform[1, 2]),
        math.atan2(float(transform[1, 0]), float(transform[0, 0])),
    )


def constrain_nonholonomic(previous_pose, matched_pose):
    """Project scan motion onto a no-lateral-slip vehicle arc."""
    relative = np.linalg.inv(previous_pose) @ matched_pose
    forward, lateral, heading = matrix_pose(relative)
    arc_lateral = forward * math.tan(heading / 2.0)
    constrained = previous_pose @ pose_matrix(
        forward,
        arc_lateral,
        heading,
    )
    return constrained, lateral - arc_lateral


def transform_points(points, transform):
    """Apply an SE(2) transform to an N-by-2 point array."""
    return points @ transform[:2, :2].T + transform[:2, 2]


def deskew_points(
        points,
        time_offsets,
        base_to_laser,
        velocity_x,
        velocity_y,
        yaw_rate):
    """Express moving-scan points in the laser frame of the first ray."""
    points = np.asarray(points, dtype=np.float64)
    offsets = np.asarray(time_offsets, dtype=np.float64)
    laser_rotation = base_to_laser[:2, :2]
    laser_translation = base_to_laser[:2, 2]
    points_in_moving_base = points @ laser_rotation.T + laser_translation

    yaw_offsets = yaw_rate * offsets
    cosine = np.cos(yaw_offsets)
    sine = np.sin(yaw_offsets)
    rotated_x = (
        cosine * points_in_moving_base[:, 0]
        - sine * points_in_moving_base[:, 1]
    )
    rotated_y = (
        sine * points_in_moving_base[:, 0]
        + cosine * points_in_moving_base[:, 1]
    )
    points_in_start_base = np.column_stack((rotated_x, rotated_y))
    points_in_start_base[:, 0] += velocity_x * offsets
    points_in_start_base[:, 1] += velocity_y * offsets
    return (points_in_start_base - laser_translation) @ laser_rotation


def voxel_downsample(points, voxel_size, max_points=None):
    """Replace points in the same grid cell with their centroid."""
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    if voxel_size <= 0.0:
        downsampled = np.asarray(points, dtype=np.float64).copy()
    else:
        points = np.asarray(points, dtype=np.float64)
        cells = np.floor(points / voxel_size).astype(np.int64)
        _, inverse = np.unique(cells, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.zeros((len(counts), 2), dtype=np.float64)
        np.add.at(sums, inverse, points)
        downsampled = sums / counts[:, np.newaxis]

    if max_points is not None and len(downsampled) > max_points:
        indices = np.linspace(
            0,
            len(downsampled) - 1,
            num=max_points,
            dtype=np.int64,
        )
        downsampled = downsampled[indices]
    return downsampled


def estimate_normals(
        points,
        neighbor_count=8,
        search_radius=0.40,
        min_neighbors=5,
        min_linearity=0.80,
        tree=None):
    """Estimate order-independent 2D line normals with local PCA."""
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        points = points.reshape((0, 2))
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError('normal input points must have shape (N, 2)')
    if neighbor_count < 2:
        raise ValueError('normal neighbor_count must be at least two')
    if min_neighbors < 2 or min_neighbors > neighbor_count:
        raise ValueError(
            'normal min_neighbors must be in [2, neighbor_count]'
        )
    if search_radius <= 0.0:
        raise ValueError('normal search_radius must be positive')
    if not 0.0 <= min_linearity <= 1.0:
        raise ValueError('normal min_linearity must be in [0, 1]')

    normals = np.zeros((len(points), 2), dtype=np.float64)
    valid = np.zeros(len(points), dtype=bool)
    linearity = np.zeros(len(points), dtype=np.float64)
    if len(points) < min_neighbors:
        return normals, valid, linearity

    if tree is None:
        tree = cKDTree(points)
    query_count = min(neighbor_count, len(points))
    distances, indices = tree.query(
        points,
        k=query_count,
        eps=0.0,
        distance_upper_bound=search_radius,
        workers=1,
    )
    if query_count == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    neighbor_mask = np.isfinite(distances) & (indices < len(points))
    neighbor_counts = np.count_nonzero(neighbor_mask, axis=1)
    safe_indices = np.minimum(indices, len(points) - 1)
    neighbors = points[safe_indices]
    masked_neighbors = np.where(
        neighbor_mask[:, :, np.newaxis], neighbors, 0.0
    )
    safe_counts = np.maximum(neighbor_counts, 1)
    means = (
        np.sum(masked_neighbors, axis=1)
        / safe_counts[:, np.newaxis]
    )
    centered = np.where(
        neighbor_mask[:, :, np.newaxis],
        neighbors - means[:, np.newaxis, :],
        0.0,
    )
    covariance = np.einsum(
        'nki,nkj->nij', centered, centered
    ) / safe_counts[:, np.newaxis, np.newaxis]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent_variance = eigenvalues[:, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        scores = 1.0 - eigenvalues[:, 0] / tangent_variance
    candidate_valid = (
        (neighbor_counts >= min_neighbors)
        & np.isfinite(tangent_variance)
        & (tangent_variance > 1.0e-12)
        & np.isfinite(scores)
        & (scores >= min_linearity)
    )
    candidate_normals = eigenvectors[:, :, 0]
    normal_lengths = np.linalg.norm(candidate_normals, axis=1)
    candidate_valid &= (
        np.isfinite(normal_lengths) & (normal_lengths > 1.0e-12)
    )
    normals[candidate_valid] = (
        candidate_normals[candidate_valid]
        / normal_lengths[candidate_valid, np.newaxis]
    )
    valid[candidate_valid] = True
    linearity[candidate_valid] = scores[candidate_valid]

    return normals, valid, linearity


def nearest_neighbors(source, target, distance_upper_bound=math.inf):
    """Find exact nearest neighbors with a cached or temporary KD-tree."""
    if isinstance(target, SubmapIndex):
        target_index = target
    else:
        target_index = SubmapIndex(target, compute_normals=False)
    return target_index.query(source, distance_upper_bound)


def estimate_rigid_transform(source, target):
    """Estimate the least-squares SE(2) transform from source to target."""
    if len(source) != len(target) or len(source) < 2:
        raise ValueError('at least two paired points are required')

    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center

    transform = np.eye(3, dtype=np.float64)
    transform[:2, :2] = rotation
    transform[:2, 2] = translation
    return transform


def icp_match(
        source,
        target,
        initial_transform,
        max_iterations=20,
        max_correspondence_distance=0.35,
        trim_fraction=0.8,
        min_correspondences=30,
        translation_epsilon=0.001,
        rotation_epsilon=0.001):
    """
    Align source points with exact KD-tree point-to-point ICP.

    The returned transform maps points from the source frame into the target
    frame.  The initial transform should already be close to the solution;
    LiDAR odometry supplies it from constant velocity and the IMU yaw delta.
    """
    source = np.asarray(source, dtype=np.float64)
    if isinstance(target, SubmapIndex):
        target_index = target
    else:
        target_index = SubmapIndex(target, compute_normals=False)
    target_points = target_index.points
    transform = np.asarray(initial_transform, dtype=np.float64).copy()
    minimum = min(min_correspondences, len(source))
    converged = False
    iterations = 0

    if len(source) < 2 or len(target_points) < 2:
        return transform, {
            'method': 'point_to_point',
            'converged': False,
            'iterations': 0,
            'correspondences': 0,
            'inlier_ratio': 0.0,
            'rmse': math.inf,
            'point_to_line_rmse': math.inf,
            'rank': 0,
            'condition_number': math.inf,
            'valid': False,
        }

    for iteration in range(max_iterations):
        iterations = iteration + 1
        transformed = transform_points(source, transform)
        nearest, squared_distances = target_index.query(
            transformed,
            max_correspondence_distance,
        )
        candidates = np.flatnonzero(nearest >= 0)
        if len(candidates) < minimum:
            break

        keep_count = max(minimum, int(len(candidates) * trim_fraction))
        if keep_count < len(candidates):
            order = np.argpartition(
                squared_distances[candidates], keep_count - 1
            )[:keep_count]
            candidates = candidates[order]

        correction = estimate_rigid_transform(
            transformed[candidates],
            target_points[nearest[candidates]],
        )
        transform = correction @ transform
        _, _, rotation_step = matrix_pose(correction)
        translation_step = math.hypot(
            correction[0, 2], correction[1, 2]
        )
        if (
                translation_step <= translation_epsilon
                and abs(rotation_step) <= rotation_epsilon):
            converged = True
            break

    transformed = transform_points(source, transform)
    nearest, squared_distances = target_index.query(
        transformed,
        max_correspondence_distance,
    )
    valid_correspondence = nearest >= 0
    correspondence_count = int(np.count_nonzero(valid_correspondence))
    if correspondence_count:
        rmse = math.sqrt(
            float(np.mean(squared_distances[valid_correspondence]))
        )
    else:
        rmse = math.inf

    return transform, {
        'method': 'point_to_point',
        'converged': converged,
        'iterations': iterations,
        'correspondences': correspondence_count,
        'inlier_ratio': correspondence_count / len(source),
        'rmse': rmse,
        'point_to_line_rmse': math.inf,
        'rank': 3 if correspondence_count >= minimum else 0,
        'condition_number': math.nan,
        'valid': correspondence_count >= minimum and math.isfinite(rmse),
    }


def _normal_correspondences(
        source,
        source_normals,
        source_normal_valid,
        target_index,
        transform,
        max_correspondence_distance,
        max_normal_angle):
    """Return distance- and normal-compatible point-to-line matches."""
    transformed = transform_points(source, transform)
    nearest, squared_distances = target_index.query(
        transformed,
        max_correspondence_distance,
    )

    valid = (nearest >= 0) & source_normal_valid
    candidate_indices = np.flatnonzero(valid)
    normal_alignment = np.zeros(len(source), dtype=np.float64)
    if len(candidate_indices):
        rotation = transform[:2, :2]
        transformed_normals = source_normals @ rotation.T
        target_indices = nearest[candidate_indices]
        target_valid = target_index.normal_valid[target_indices]
        dots = np.abs(np.einsum(
            'ij,ij->i',
            transformed_normals[candidate_indices],
            target_index.normals[target_indices],
        ))
        dots = np.clip(dots, 0.0, 1.0)
        normal_alignment[candidate_indices] = dots
        valid[candidate_indices] &= (
            target_valid & (dots >= math.cos(max_normal_angle))
        )

    return transformed, nearest, squared_distances, normal_alignment, valid


def _point_to_line_quality(
        source,
        source_normals,
        source_normal_valid,
        target_index,
        transform,
        max_correspondence_distance,
        max_normal_angle,
        iterations,
        converged,
        rank,
        condition_number,
        valid_solution):
    """Evaluate Euclidean and surface-normal residuals after matching."""
    (
        transformed,
        nearest,
        squared_distances,
        normal_alignment,
        valid,
    ) = _normal_correspondences(
        source,
        source_normals,
        source_normal_valid,
        target_index,
        transform,
        max_correspondence_distance,
        max_normal_angle,
    )
    candidates = np.flatnonzero(valid)
    correspondence_count = len(candidates)
    euclidean_rmse = math.inf
    line_rmse = math.inf
    mean_alignment = 0.0
    if correspondence_count:
        target_points = target_index.points[nearest[candidates]]
        target_normals = target_index.normals[nearest[candidates]]
        residuals = np.einsum(
            'ij,ij->i',
            transformed[candidates] - target_points,
            target_normals,
        )
        euclidean_rmse = math.sqrt(
            float(np.mean(squared_distances[candidates]))
        )
        line_rmse = math.sqrt(float(np.mean(residuals * residuals)))
        mean_alignment = float(np.mean(normal_alignment[candidates]))

    return {
        'method': 'point_to_line',
        'converged': converged,
        'iterations': iterations,
        'correspondences': correspondence_count,
        'inlier_ratio': (
            correspondence_count / len(source) if len(source) else 0.0
        ),
        # Preserve the old quality contract: rmse is Euclidean NN RMSE.
        'rmse': euclidean_rmse,
        'point_to_line_rmse': line_rmse,
        'normal_alignment': mean_alignment,
        'rank': rank,
        'condition_number': condition_number,
        'valid': valid_solution and correspondence_count > 0,
    }


def _trim_candidates(candidates, scores, trim_fraction, minimum):
    """Keep the lowest-residual candidates without sorting the full set."""
    if len(candidates) <= minimum:
        return candidates
    keep_count = max(minimum, int(len(candidates) * trim_fraction))
    if keep_count >= len(candidates):
        return candidates
    keep = np.argpartition(scores, keep_count - 1)[:keep_count]
    return candidates[keep]


def _hybrid_quality(
        source,
        source_normals,
        source_normal_valid,
        target_index,
        transform,
        max_correspondence_distance,
        max_normal_angle,
        min_group_correspondences,
        iterations,
        converged,
        rank,
        condition_number,
        valid_solution):
    """Evaluate disjoint line and point residual groups after matching."""
    (
        transformed,
        nearest,
        squared_distances,
        normal_alignment,
        line_valid,
    ) = _normal_correspondences(
        source,
        source_normals,
        source_normal_valid,
        target_index,
        transform,
        max_correspondence_distance,
        max_normal_angle,
    )
    matched = nearest >= 0
    point_valid = matched & ~line_valid
    line_candidates = np.flatnonzero(line_valid)
    point_candidates = np.flatnonzero(point_valid)
    all_candidates = np.flatnonzero(matched)

    euclidean_rmse = math.inf
    line_rmse = math.inf
    point_rmse = math.inf
    mean_alignment = 0.0
    if len(all_candidates):
        euclidean_rmse = math.sqrt(
            float(np.mean(squared_distances[all_candidates]))
        )
    if len(line_candidates):
        target_points = target_index.points[nearest[line_candidates]]
        target_normals = target_index.normals[nearest[line_candidates]]
        residuals = np.einsum(
            'ij,ij->i',
            transformed[line_candidates] - target_points,
            target_normals,
        )
        line_rmse = math.sqrt(float(np.mean(residuals * residuals)))
        mean_alignment = float(
            np.mean(normal_alignment[line_candidates])
        )
    if len(point_candidates):
        point_rmse = math.sqrt(
            float(np.mean(squared_distances[point_candidates]))
        )

    correspondence_count = len(all_candidates)
    return {
        'method': 'hybrid',
        'converged': converged,
        'iterations': iterations,
        'correspondences': correspondence_count,
        'line_correspondences': len(line_candidates),
        'point_correspondences': len(point_candidates),
        'line_group_active': (
            len(line_candidates) >= min_group_correspondences
        ),
        'point_group_active': (
            len(point_candidates) >= min_group_correspondences
        ),
        'inlier_ratio': (
            correspondence_count / len(source) if len(source) else 0.0
        ),
        'rmse': euclidean_rmse,
        'point_to_line_rmse': line_rmse,
        'point_to_point_rmse': point_rmse,
        'normal_alignment': mean_alignment,
        'rank': rank,
        'condition_number': condition_number,
        'valid': (
            valid_solution
            and correspondence_count > 0
            and math.isfinite(euclidean_rmse)
        ),
    }


def hybrid_icp(
        source,
        target_index,
        initial_transform,
        source_normals=None,
        source_normal_valid=None,
        normal_neighbors=8,
        normal_radius=0.40,
        normal_min_neighbors=5,
        normal_min_linearity=0.80,
        max_iterations=12,
        max_correspondence_distance=0.30,
        max_normal_angle=math.radians(45.0),
        trim_fraction=0.80,
        min_correspondences=30,
        min_group_correspondences=5,
        line_weight=1.0,
        point_weight=1.0,
        translation_epsilon=0.001,
        rotation_epsilon=0.001,
        svd_rcond=1.0e-3,
        max_condition_number=1.0e4,
        max_translation_step=0.30,
        max_rotation_step=0.20):
    """
    Optimize line-like and compact obstacle points in one ICP solve.

    Each iteration performs one shared nearest-neighbor query.  Normal-
    compatible correspondences contribute a scalar Point-to-Line residual;
    all other matched returns contribute a two-axis Point-to-Point residual.
    The two groups are normalized independently so numerous wall points do
    not drown out a smaller set of distinctive obstacles.
    """
    source = np.asarray(source, dtype=np.float64)
    transform = np.asarray(initial_transform, dtype=np.float64).copy()
    if not isinstance(target_index, SubmapIndex):
        raise TypeError('hybrid_icp requires a SubmapIndex')
    if source.ndim != 2 or source.shape[1:] != (2,):
        raise ValueError('source points must have shape (N, 2)')
    if not 0.0 < trim_fraction <= 1.0:
        raise ValueError('trim_fraction must be in (0, 1]')
    if min_group_correspondences < 2:
        raise ValueError('min_group_correspondences must be at least two')
    if line_weight <= 0.0 or point_weight <= 0.0:
        raise ValueError('hybrid residual weights must be positive')

    if source_normals is None or source_normal_valid is None:
        source_normals, source_normal_valid, _ = estimate_normals(
            source,
            neighbor_count=normal_neighbors,
            search_radius=normal_radius,
            min_neighbors=normal_min_neighbors,
            min_linearity=normal_min_linearity,
        )
    else:
        source_normals = np.asarray(source_normals, dtype=np.float64)
        source_normal_valid = np.asarray(source_normal_valid, dtype=bool)
        if source_normals.shape != source.shape:
            raise ValueError('source normals must match source point shape')
        if source_normal_valid.shape != (len(source),):
            raise ValueError(
                'source normal validity must have one value per point'
            )

    minimum = min(min_correspondences, len(source))
    converged = False
    iterations = 0
    rank = 0
    condition_number = math.inf
    valid_solution = False

    if len(source) < 2 or len(target_index.points) < 2:
        return transform, _hybrid_quality(
            source,
            source_normals,
            source_normal_valid,
            target_index,
            transform,
            max_correspondence_distance,
            max_normal_angle,
            min_group_correspondences,
            iterations,
            converged,
            rank,
            condition_number,
            valid_solution,
        )

    for iteration in range(max_iterations):
        iterations = iteration + 1
        (
            transformed,
            nearest,
            squared_distances,
            _,
            line_valid,
        ) = _normal_correspondences(
            source,
            source_normals,
            source_normal_valid,
            target_index,
            transform,
            max_correspondence_distance,
            max_normal_angle,
        )
        matched = nearest >= 0
        if int(np.count_nonzero(matched)) < minimum:
            break

        line_candidates = np.flatnonzero(line_valid)
        point_candidates = np.flatnonzero(matched & ~line_valid)

        line_residuals = np.empty(0, dtype=np.float64)
        line_normals = np.empty((0, 2), dtype=np.float64)
        if len(line_candidates) >= min_group_correspondences:
            line_target_points = target_index.points[
                nearest[line_candidates]
            ]
            line_normals = target_index.normals[nearest[line_candidates]]
            line_residuals = np.einsum(
                'ij,ij->i',
                transformed[line_candidates] - line_target_points,
                line_normals,
            )
            keep = _trim_candidates(
                np.arange(len(line_candidates), dtype=np.int64),
                np.abs(line_residuals),
                trim_fraction,
                min_group_correspondences,
            )
            line_candidates = line_candidates[keep]
            line_normals = line_normals[keep]
            line_residuals = line_residuals[keep]
        else:
            line_candidates = np.empty(0, dtype=np.int64)

        point_residuals = np.empty((0, 2), dtype=np.float64)
        if len(point_candidates) >= min_group_correspondences:
            keep = _trim_candidates(
                np.arange(len(point_candidates), dtype=np.int64),
                squared_distances[point_candidates],
                trim_fraction,
                min_group_correspondences,
            )
            point_candidates = point_candidates[keep]
            point_targets = target_index.points[nearest[point_candidates]]
            point_residuals = transformed[point_candidates] - point_targets
        else:
            point_candidates = np.empty(0, dtype=np.int64)

        if not len(line_candidates) and not len(point_candidates):
            break

        rotation = transform[:2, :2]
        local_x_axis = rotation[:, 0]
        local_y_axis = rotation[:, 1]
        jacobian_groups = []
        residual_groups = []
        radius_indices = []

        if len(line_candidates):
            line_perpendicular = source[line_candidates][:, ::-1].copy()
            line_perpendicular[:, 0] *= -1.0
            line_perpendicular = line_perpendicular @ rotation.T
            line_jacobian = np.column_stack((
                line_normals @ local_x_axis,
                line_normals @ local_y_axis,
                np.einsum(
                    'ij,ij->i', line_normals, line_perpendicular
                ),
            ))
            line_scale = math.sqrt(line_weight / len(line_candidates))
            jacobian_groups.append(line_jacobian * line_scale)
            residual_groups.append(line_residuals * line_scale)
            radius_indices.append(line_candidates)

        if len(point_candidates):
            point_perpendicular = source[point_candidates][:, ::-1].copy()
            point_perpendicular[:, 0] *= -1.0
            point_perpendicular = point_perpendicular @ rotation.T
            point_jacobian = np.empty(
                (len(point_candidates), 2, 3), dtype=np.float64
            )
            point_jacobian[:, :, 0] = local_x_axis
            point_jacobian[:, :, 1] = local_y_axis
            point_jacobian[:, :, 2] = point_perpendicular
            point_scale = math.sqrt(point_weight / len(point_candidates))
            jacobian_groups.append(
                point_jacobian.reshape((-1, 3)) * point_scale
            )
            residual_groups.append(
                point_residuals.reshape(-1) * point_scale
            )
            radius_indices.append(point_candidates)

        jacobian = np.vstack(jacobian_groups)
        residuals = np.concatenate(residual_groups)
        selected = np.concatenate(radius_indices)
        characteristic_radius = float(np.median(
            np.linalg.norm(source[selected], axis=1)
        ))
        characteristic_radius = float(np.clip(
            characteristic_radius, 1.0, 10.0
        ))
        scaled_jacobian = jacobian.copy()
        scaled_jacobian[:, 2] /= characteristic_radius
        try:
            scaled_delta, _, rank, singular_values = np.linalg.lstsq(
                scaled_jacobian,
                -residuals,
                rcond=svd_rcond,
            )
        except np.linalg.LinAlgError:
            break

        rank = int(rank)
        if rank < 2 or not np.all(np.isfinite(scaled_delta)):
            break
        condition_number = float(
            singular_values[0] / singular_values[rank - 1]
        )
        if (
                not math.isfinite(condition_number)
                or condition_number > max_condition_number):
            break

        delta = scaled_delta.copy()
        delta[2] /= characteristic_radius
        translation_step = math.hypot(delta[0], delta[1])
        rotation_step = abs(float(delta[2]))
        if (
                translation_step > max_translation_step
                or rotation_step > max_rotation_step):
            break

        transform = transform @ pose_matrix(*delta)
        valid_solution = True
        if (
                translation_step <= translation_epsilon
                and rotation_step <= rotation_epsilon):
            converged = True
            break

    return transform, _hybrid_quality(
        source,
        source_normals,
        source_normal_valid,
        target_index,
        transform,
        max_correspondence_distance,
        max_normal_angle,
        min_group_correspondences,
        iterations,
        converged,
        rank,
        condition_number,
        valid_solution,
    )


def point_to_line_icp(
        source,
        target_index,
        initial_transform,
        source_normals=None,
        source_normal_valid=None,
        normal_neighbors=8,
        normal_radius=0.40,
        normal_min_neighbors=5,
        normal_min_linearity=0.80,
        max_iterations=12,
        max_correspondence_distance=0.30,
        max_normal_angle=math.radians(45.0),
        trim_fraction=0.80,
        min_correspondences=30,
        translation_epsilon=0.001,
        rotation_epsilon=0.001,
        svd_rcond=1.0e-3,
        max_condition_number=1.0e4,
        max_translation_step=0.30,
        max_rotation_step=0.20):
    """Align a scan to cached submap lines with trimmed Gauss-Newton ICP."""
    source = np.asarray(source, dtype=np.float64)
    transform = np.asarray(initial_transform, dtype=np.float64).copy()
    if not isinstance(target_index, SubmapIndex):
        raise TypeError('point_to_line_icp requires a SubmapIndex')
    if source.ndim != 2 or source.shape[1:] != (2,):
        raise ValueError('source points must have shape (N, 2)')
    if not 0.0 < trim_fraction <= 1.0:
        raise ValueError('trim_fraction must be in (0, 1]')

    if source_normals is None or source_normal_valid is None:
        source_normals, source_normal_valid, _ = estimate_normals(
            source,
            neighbor_count=normal_neighbors,
            search_radius=normal_radius,
            min_neighbors=normal_min_neighbors,
            min_linearity=normal_min_linearity,
        )
    else:
        source_normals = np.asarray(source_normals, dtype=np.float64)
        source_normal_valid = np.asarray(source_normal_valid, dtype=bool)
        if source_normals.shape != source.shape:
            raise ValueError('source normals must match source point shape')
        if source_normal_valid.shape != (len(source),):
            raise ValueError(
                'source normal validity must have one value per point'
            )

    minimum = min(min_correspondences, len(source))
    converged = False
    iterations = 0
    rank = 0
    condition_number = math.inf
    valid_solution = False

    if len(source) < 2 or len(target_index.points) < 2:
        return transform, _point_to_line_quality(
            source,
            source_normals,
            source_normal_valid,
            target_index,
            transform,
            max_correspondence_distance,
            max_normal_angle,
            iterations,
            converged,
            rank,
            condition_number,
            valid_solution,
        )

    for iteration in range(max_iterations):
        iterations = iteration + 1
        (
            transformed,
            nearest,
            _,
            _,
            valid,
        ) = _normal_correspondences(
            source,
            source_normals,
            source_normal_valid,
            target_index,
            transform,
            max_correspondence_distance,
            max_normal_angle,
        )
        candidates = np.flatnonzero(valid)
        if len(candidates) < minimum:
            break

        target_points = target_index.points[nearest[candidates]]
        target_normals = target_index.normals[nearest[candidates]]
        residuals = np.einsum(
            'ij,ij->i',
            transformed[candidates] - target_points,
            target_normals,
        )
        keep_count = max(minimum, int(len(candidates) * trim_fraction))
        if keep_count < len(candidates):
            keep = np.argpartition(
                np.abs(residuals), keep_count - 1
            )[:keep_count]
            candidates = candidates[keep]
            target_normals = target_normals[keep]
            residuals = residuals[keep]

        rotation = transform[:2, :2]
        local_x_axis = rotation[:, 0]
        local_y_axis = rotation[:, 1]
        rotated_perpendicular = source[candidates][:, ::-1].copy()
        rotated_perpendicular[:, 0] *= -1.0
        rotated_perpendicular = rotated_perpendicular @ rotation.T
        jacobian = np.column_stack((
            target_normals @ local_x_axis,
            target_normals @ local_y_axis,
            np.einsum('ij,ij->i', target_normals, rotated_perpendicular),
        ))

        characteristic_radius = float(np.median(
            np.linalg.norm(source[candidates], axis=1)
        ))
        characteristic_radius = float(np.clip(
            characteristic_radius, 1.0, 10.0
        ))
        scaled_jacobian = jacobian.copy()
        scaled_jacobian[:, 2] /= characteristic_radius
        try:
            scaled_delta, _, rank, singular_values = np.linalg.lstsq(
                scaled_jacobian,
                -residuals,
                rcond=svd_rcond,
            )
        except np.linalg.LinAlgError:
            break

        rank = int(rank)
        if rank < 2 or not np.all(np.isfinite(scaled_delta)):
            break
        condition_number = float(
            singular_values[0] / singular_values[rank - 1]
        )
        if (
                not math.isfinite(condition_number)
                or condition_number > max_condition_number):
            break

        delta = scaled_delta.copy()
        delta[2] /= characteristic_radius
        translation_step = math.hypot(delta[0], delta[1])
        rotation_step = abs(float(delta[2]))
        if (
                translation_step > max_translation_step
                or rotation_step > max_rotation_step):
            break

        transform = transform @ pose_matrix(*delta)
        valid_solution = True
        if (
                translation_step <= translation_epsilon
                and rotation_step <= rotation_epsilon):
            converged = True
            break

    return transform, _point_to_line_quality(
        source,
        source_normals,
        source_normal_valid,
        target_index,
        transform,
        max_correspondence_distance,
        max_normal_angle,
        iterations,
        converged,
        rank,
        condition_number,
        valid_solution,
    )
