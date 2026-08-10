#!/usr/bin/env python3

"""Launch LiDAR-only Cartographer mapping and automatic map saving."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# Vehicle frame and measured LiDAR mount defaults.
# base_link origin: rear axle center
# ROS axes: +x forward, +y left, +z up
BASE_FRAME = 'base_link'
LIDAR_FRAME = 'laser_frame'
PACKAGE_NAME = 'cartographer_2d_slam'

LIDAR_XYZ = ('0.410', '0.0', '0.04')
LEVEL_RPY = ('0.0', '0.0', '0.0')

SLAM_SCAN_TOPIC = '/scan_slam'
MAP_RESOLUTION = '0.02'
MAP_PUBLISH_PERIOD_SEC = '1.0'
DEFAULT_MAP_OUTPUT_FILE = os.path.join(
    os.path.expanduser('~'),
    '.ros',
    'cartographer_2d_slam',
)
DEFAULT_PBSTREAM_OUTPUT_FILE = os.path.join(
    os.path.expanduser('~'),
    '.ros',
    'cartographer_2d_slam.pbstream',
)


def _static_tf_node(name, child_frame, xyz, rpy=LEVEL_RPY):
    """Publish the measured base_link-to-sensor mounting transform."""
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=name,
        output='screen',
        arguments=[
            '--x', xyz[0],
            '--y', xyz[1],
            '--z', xyz[2],
            '--roll', rpy[0],
            '--pitch', rpy[1],
            '--yaw', rpy[2],
            '--frame-id', BASE_FRAME,
            '--child-frame-id', child_frame,
        ],
    )


def generate_launch_description():
    """Build the complete LiDAR-only mapping launch description."""
    package_root = get_package_share_directory(PACKAGE_NAME)
    configuration_directory = os.path.join(package_root, 'config')

    save_state_argument = DeclareLaunchArgument(
        'save_state_filename',
        default_value=DEFAULT_PBSTREAM_OUTPUT_FILE,
        description=(
            'Write the final Cartographer state to this .pbstream file '
            'during a clean shutdown.'
        ),
    )
    map_output_argument = DeclareLaunchArgument(
        'map_output_file',
        default_value=DEFAULT_MAP_OUTPUT_FILE,
        description='YAML/PGM output path without a file extension.',
    )
    use_sim_time_argument = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use the simulation clock when true.',
    )
    lidar_x_argument = DeclareLaunchArgument(
        'lidar_x',
        default_value=LIDAR_XYZ[0],
        description='LiDAR x offset from base_link in metres.',
    )
    lidar_y_argument = DeclareLaunchArgument(
        'lidar_y',
        default_value=LIDAR_XYZ[1],
        description='LiDAR y offset from base_link in metres.',
    )
    lidar_z_argument = DeclareLaunchArgument(
        'lidar_z',
        default_value=LIDAR_XYZ[2],
        description='LiDAR z offset from base_link in metres.',
    )
    lidar_yaw_argument = DeclareLaunchArgument(
        'lidar_yaw',
        default_value=LEVEL_RPY[2],
        description='LiDAR yaw offset from base_link in radians.',
    )

    use_sim_time = ParameterValue(
        LaunchConfiguration('use_sim_time'),
        value_type=bool,
    )

    lidar_static_tf = _static_tf_node(
        name='base_to_lidar_static_tf',
        child_frame=LIDAR_FRAME,
        xyz=(
            LaunchConfiguration('lidar_x'),
            LaunchConfiguration('lidar_y'),
            LaunchConfiguration('lidar_z'),
        ),
        rpy=(
            LEVEL_RPY[0],
            LEVEL_RPY[1],
            LaunchConfiguration('lidar_yaw'),
        ),
    )

    rear_scan_filter_node = Node(
        package=PACKAGE_NAME,
        executable='rear_scan_filter',
        name='rear_scan_filter',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-configuration_directory', configuration_directory,
            '-configuration_basename', 'parking_2d.lua',
            '-save_state_filename',
            LaunchConfiguration('save_state_filename'),
        ],
        remappings=[('scan', SLAM_SCAN_TOPIC)],
    )

    occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-resolution', MAP_RESOLUTION,
            '-publish_period_sec', MAP_PUBLISH_PERIOD_SEC,
        ],
    )

    auto_map_saver_node = Node(
        package=PACKAGE_NAME,
        executable='auto_map_saver',
        name='auto_map_saver',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'output_file': LaunchConfiguration('map_output_file'),
        }],
    )

    return LaunchDescription([
        save_state_argument,
        map_output_argument,
        use_sim_time_argument,
        lidar_x_argument,
        lidar_y_argument,
        lidar_z_argument,
        lidar_yaw_argument,
        lidar_static_tf,
        rear_scan_filter_node,
        cartographer_node,
        occupancy_grid_node,
        auto_map_saver_node,
    ])
