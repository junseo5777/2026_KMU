"""Small, dependency-light helpers for planar LiDAR odometry."""

import math

import numpy as np


def normalize_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


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


def nearest_neighbors(source, target, chunk_size=256):
    """Find the closest target point for every source point."""
    if len(source) == 0 or len(target) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )

    indices = np.empty(len(source), dtype=np.int64)
    squared_distances = np.empty(len(source), dtype=np.float64)
    for start in range(0, len(source), chunk_size):
        stop = min(start + chunk_size, len(source))
        differences = (
            source[start:stop, np.newaxis, :]
            - target[np.newaxis, :, :]
        )
        distances = np.einsum('ijk,ijk->ij', differences, differences)
        nearest = np.argmin(distances, axis=1)
        indices[start:stop] = nearest
        squared_distances[start:stop] = distances[
            np.arange(stop - start), nearest
        ]
    return indices, squared_distances


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
    Align source points to a target submap with trimmed point ICP.

    The returned transform maps points from the source frame into the target
    frame.  The initial transform should already be close to the solution;
    LiDAR odometry supplies it from constant velocity and the IMU yaw delta.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    transform = np.asarray(initial_transform, dtype=np.float64).copy()
    minimum = min(min_correspondences, len(source))
    converged = False
    iterations = 0

    if len(source) < 2 or len(target) < 2:
        return transform, {
            'converged': False,
            'iterations': 0,
            'correspondences': 0,
            'inlier_ratio': 0.0,
            'rmse': math.inf,
        }

    distance_limit_squared = max_correspondence_distance ** 2
    for iteration in range(max_iterations):
        iterations = iteration + 1
        transformed = transform_points(source, transform)
        nearest, squared_distances = nearest_neighbors(transformed, target)
        candidates = np.flatnonzero(
            squared_distances <= distance_limit_squared
        )
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
            target[nearest[candidates]],
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
    _, squared_distances = nearest_neighbors(transformed, target)
    valid = squared_distances <= distance_limit_squared
    correspondence_count = int(np.count_nonzero(valid))
    if correspondence_count:
        rmse = math.sqrt(float(np.mean(squared_distances[valid])))
    else:
        rmse = math.inf

    return transform, {
        'converged': converged,
        'iterations': iterations,
        'correspondences': correspondence_count,
        'inlier_ratio': correspondence_count / len(source),
        'rmse': rmse,
    }
