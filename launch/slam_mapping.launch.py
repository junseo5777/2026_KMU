#!/usr/bin/env python3

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

"""Launch Cartographer with optional-gyro LiDAR odometry and map saving."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


# Vehicle coordinates: base_link is the rear-axle center, +x forward.
BASE_FRAME = 'base_link'
LIDAR_FRAME = 'laser_frame'
IMU_FRAME = 'base_imu_link'
LIDAR_XYZ = ('0.410', '0.0', '0.04')
IMU_XYZ = ('0.075', '0.0', '0.090')
LEVEL_RPY = ('0.0', '0.0', '0.0')

SLAM_SCAN_TOPIC = '/scan_slam'
ODOM_TOPIC = '/lidar_odom'
MAP_OUTPUT_FILE = os.path.expanduser(
    '~/.ros/cartographer_2d_slam'
)
PBSTREAM_OUTPUT_FILE = f'{MAP_OUTPUT_FILE}.pbstream'


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
    """Build the complete KMU Cartographer mapping launch description."""
    package_root = get_package_share_directory('cartographer_2d_slam')
    configuration_directory = os.path.join(package_root, 'config')
    odom_params_filename = os.path.join(
        configuration_directory,
        'lidar_odom.yaml',
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
    imu_static_tf = _static_tf_node(
        name='base_to_imu_static_tf',
        child_frame=LaunchConfiguration('imu_frame'),
        xyz=(
            LaunchConfiguration('imu_x'),
            LaunchConfiguration('imu_y'),
            LaunchConfiguration('imu_z'),
        ),
        rpy=(
            LaunchConfiguration('imu_roll'),
            LaunchConfiguration('imu_pitch'),
            LaunchConfiguration('imu_yaw'),
        ),
    )

    rear_scan_filter_node = Node(
        package='cartographer_2d_slam',
        executable='rear_scan_filter',
        name='rear_scan_filter',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Consecutive Point-to-Point matching supplies translation and yaw in the
    # current comparison. Gyro yaw is only added when use_imu:=true.
    # Cartographer remains the only TF authority.
    lidar_odom_node = Node(
        package='cartographer_2d_slam',
        executable='lidar_odom_node',
        name='lidar_odom_node',
        output='screen',
        additional_env={'PYTHONNOUSERSITE': '1'},
        parameters=[
            odom_params_filename,
            {
                'use_sim_time': use_sim_time,
                'scan_topic': ParameterValue(
                    LaunchConfiguration('scan_topic'),
                    value_type=str,
                ),
                'imu_topic': ParameterValue(
                    LaunchConfiguration('imu_topic'),
                    value_type=str,
                ),
                'odom_topic': ParameterValue(
                    LaunchConfiguration('odom_topic'),
                    value_type=str,
                ),
                'odom_frame_id': 'odom',
                'base_frame_id': BASE_FRAME,
                'laser_frame_id': LIDAR_FRAME,
                'imu_frame_id': ParameterValue(
                    LaunchConfiguration('imu_frame'),
                    value_type=str,
                ),
                'publish_tf': False,
                'use_imu': ParameterValue(
                    LaunchConfiguration('use_imu'),
                    value_type=bool,
                ),
                'lidar_x': ParameterValue(
                    LaunchConfiguration('lidar_x'),
                    value_type=float,
                ),
                'lidar_y': ParameterValue(
                    LaunchConfiguration('lidar_y'),
                    value_type=float,
                ),
                'lidar_yaw': ParameterValue(
                    LaunchConfiguration('lidar_yaw'),
                    value_type=float,
                ),
            },
        ],
    )

    # Cartographer owns map -> odom -> base_link during mapping and consumes
    # the custom odometry message only as a local motion prior.
    cartographer_node = Node(
        package='cartographer_ros',
        executable='cartographer_node',
        name='cartographer_node',
        output='screen',
        # Cartographer finishes pending pose-graph constraints and serializes
        # the pbstream after SIGINT.  A real parking-lot run can need much
        # longer than launch's five-second default before it exits cleanly.
        sigterm_timeout='180.0',
        sigkill_timeout='30.0',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-configuration_directory', configuration_directory,
            '-configuration_basename', 'parking_2d.lua',
            '-save_state_filename',
            LaunchConfiguration('save_state_filename'),
        ],
        remappings=[
            ('scan', LaunchConfiguration('scan_topic')),
            ('odom', LaunchConfiguration('odom_topic')),
        ],
    )

    occupancy_grid_node = Node(
        package='cartographer_ros',
        executable='cartographer_occupancy_grid_node',
        name='cartographer_occupancy_grid_node',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            '-resolution', '0.02',
            '-publish_period_sec', '1.0',
        ],
    )

    auto_map_saver_node = Node(
        package='cartographer_2d_slam',
        executable='auto_map_saver',
        name='auto_map_saver',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'output_file': LaunchConfiguration('map_output_file'),
        }],
    )

    vehicle_marker_node = Node(
        package='cartographer_2d_slam',
        executable='vehicle_marker_publisher',
        name='vehicle_marker_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'output_frame': BASE_FRAME,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use the simulation clock when true.',
        ),
        DeclareLaunchArgument(
            'scan_topic',
            default_value=SLAM_SCAN_TOPIC,
            description='Filtered LaserScan consumed by Cartographer.',
        ),
        DeclareLaunchArgument(
            'imu_topic',
            default_value='/imu',
            description='Gyroscope topic used for relative yaw only.',
        ),
        DeclareLaunchArgument(
            'use_imu',
            default_value='true',
            description=(
                'Fuse gyro yaw into LiDAR odometry when true. '
                'False runs the LiDAR-only comparison.'
            ),
        ),
        DeclareLaunchArgument(
            'imu_frame',
            default_value=IMU_FRAME,
            description='Frame ID published by the IMU driver.',
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value=ODOM_TOPIC,
            description='Point-to-Point LiDAR odometry input to Cartographer.',
        ),
        DeclareLaunchArgument(
            'map_output_file',
            default_value=MAP_OUTPUT_FILE,
            description='PGM/YAML output path without an extension.',
        ),
        DeclareLaunchArgument(
            'save_state_filename',
            default_value=PBSTREAM_OUTPUT_FILE,
            description=(
                'Final Cartographer state written during clean shutdown.'
            ),
        ),
        DeclareLaunchArgument('lidar_x', default_value=LIDAR_XYZ[0]),
        DeclareLaunchArgument('lidar_y', default_value=LIDAR_XYZ[1]),
        DeclareLaunchArgument('lidar_z', default_value=LIDAR_XYZ[2]),
        DeclareLaunchArgument('lidar_yaw', default_value=LEVEL_RPY[2]),
        DeclareLaunchArgument('imu_x', default_value=IMU_XYZ[0]),
        DeclareLaunchArgument('imu_y', default_value=IMU_XYZ[1]),
        DeclareLaunchArgument('imu_z', default_value=IMU_XYZ[2]),
        DeclareLaunchArgument('imu_roll', default_value=LEVEL_RPY[0]),
        DeclareLaunchArgument('imu_pitch', default_value=LEVEL_RPY[1]),
        DeclareLaunchArgument('imu_yaw', default_value=LEVEL_RPY[2]),
        lidar_static_tf,
        imu_static_tf,
        rear_scan_filter_node,
        lidar_odom_node,
        cartographer_node,
        occupancy_grid_node,
        auto_map_saver_node,
        vehicle_marker_node,
    ])
