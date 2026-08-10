"""Utilities for saving ROS occupancy grids in the Nav2 map format."""

import json
import math
import os
from pathlib import Path
import tempfile


UNKNOWN_PIXEL = 205   # PGM의 미확인 영역(회색)
FREE_PIXEL = 254      # PGM의 주행 가능 영역(흰색)
OCCUPIED_PIXEL = 0    # PGM의 장애물 영역(검은색)


def _validate_thresholds(free_thresh, occupied_thresh):
    if not 0.0 <= free_thresh < occupied_thresh <= 1.0:
        raise ValueError(
            'thresholds must satisfy '
            '0.0 <= free_thresh < occupied_thresh <= 1.0'
        )


def occupancy_grid_to_pgm(
        width, height, data, free_thresh=0.25, occupied_thresh=0.65):
    """Convert row-major OccupancyGrid values into a binary PGM image."""
    _validate_thresholds(free_thresh, occupied_thresh)

    if width <= 0 or height <= 0:
        raise ValueError('map width and height must be positive')
    if len(data) != width * height:
        raise ValueError(
            f'map data has {len(data)} cells; expected {width * height}'
        )

    pixels = bytearray()
    for y in range(height - 1, -1, -1):
        row_start = y * width
        for x in range(width):
            occupancy = data[row_start + x]
            if occupancy < 0:
                pixel = UNKNOWN_PIXEL
            elif occupancy / 100.0 >= occupied_thresh:
                pixel = OCCUPIED_PIXEL
            elif occupancy / 100.0 <= free_thresh:
                pixel = FREE_PIXEL
            else:
                pixel = UNKNOWN_PIXEL
            pixels.append(pixel)

    header = (
        'P5\n'
        '# CREATOR: cartographer_2d_slam auto_map_saver\n'
        f'{width} {height}\n'
        '255\n'
    ).encode('ascii')
    return header + bytes(pixels)


def quaternion_to_yaw(orientation):
    """Return planar yaw from a geometry_msgs-style quaternion."""
    sin_yaw = 2.0 * (
        orientation.w * orientation.z
        + orientation.x * orientation.y
    )
    cos_yaw = 1.0 - 2.0 * (
        orientation.y * orientation.y
        + orientation.z * orientation.z
    )
    return math.atan2(sin_yaw, cos_yaw)


def _map_base_path(output_file):
    path = Path(os.path.expanduser(output_file))
    if path.suffix.lower() in {'.pgm', '.yaml', '.yml'}:
        path = path.with_suffix('')
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _atomic_write(path, content, binary):
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f'.{path.name}.',
        suffix='.tmp',
        dir=str(path.parent),
    )
    try:
        mode = 'wb' if binary else 'w'
        kwargs = {} if binary else {'encoding': 'utf-8'}
        with os.fdopen(file_descriptor, mode, **kwargs) as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, path)
    except OSError:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def save_occupancy_grid(
        map_message, output_file, free_thresh=0.25, occupied_thresh=0.65):
    """Save an OccupancyGrid message as Nav2-compatible YAML and PGM files."""
    info = map_message.info
    if info.resolution <= 0.0:
        raise ValueError('map resolution must be positive')

    base_path = _map_base_path(output_file)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    image_path = base_path.with_suffix('.pgm')
    yaml_path = base_path.with_suffix('.yaml')

    pgm_content = occupancy_grid_to_pgm(
        info.width,
        info.height,
        map_message.data,
        free_thresh,
        occupied_thresh,
    )
    yaw = quaternion_to_yaw(info.origin.orientation)
    origin = info.origin.position
    yaml_content = (
        f'image: {json.dumps(image_path.name)}\n'
        'mode: trinary\n'
        f'resolution: {float(info.resolution):.12g}\n'
        f'origin: [{float(origin.x):.12g}, '
        f'{float(origin.y):.12g}, {yaw:.12g}]\n'
        'negate: 0\n'
        f'occupied_thresh: {float(occupied_thresh):.12g}\n'
        f'free_thresh: {float(free_thresh):.12g}\n'
    )

    _atomic_write(image_path, pgm_content, binary=True)
    _atomic_write(yaml_path, yaml_content, binary=False)
    return yaml_path, image_path
