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

"""Localize with LiDAR odometry and Nav2 AMCL on a saved map."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessStart
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import LifecycleNode, Node
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.parameter_descriptions import ParameterValue

from lifecycle_msgs.msg import Transition


BASE_FRAME = 'base_link'
LIDAR_FRAME = 'laser_frame'
IMU_FRAME = 'base_imu_link'
MAP_FRAME = 'map'

LIDAR_XYZ = ('0.410', '0.0', '0.04')
LEVEL_RPY = ('0.0', '0.0', '0.0')
IMU_XYZ = ('0.075', '0.0', '0.090')

FILTERED_SCAN_TOPIC = '/scan_slam'


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


def _localization_nodes(context):
    """Validate inputs and construct LiDAR odometry plus AMCL."""
    map_filename = os.path.abspath(
        os.path.expanduser(LaunchConfiguration('map').perform(context))
    )
    params_filename = os.path.abspath(
        os.path.expanduser(
            LaunchConfiguration('params_file').perform(context)
        )
    )
    odom_params_filename = os.path.abspath(
        os.path.expanduser(
            LaunchConfiguration('odom_params_file').perform(context)
        )
    )

    if not os.path.isfile(map_filename):
        raise RuntimeError(
            'AMCL localization requires an existing map YAML file: '
            f'{map_filename}'
        )
    if not map_filename.endswith(('.yaml', '.yml')):
        raise RuntimeError(
            'map must point to a Nav2 .yaml/.yml map file: '
            f'{map_filename}'
        )
    if not os.path.isfile(params_filename):
        raise RuntimeError(
            'AMCL parameter file does not exist: '
            f'{params_filename}'
        )
    if not os.path.isfile(odom_params_filename):
        raise RuntimeError(
            'LiDAR odometry parameter file does not exist: '
            f'{odom_params_filename}'
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

    # AMCL consumes odom -> base_link from scan-to-scan ICP.  The geometry
    # overrides below are shared with the static TF to prevent frame mismatch.
    lidar_odom_node = Node(
        package='cartographer_2d_slam',
        executable='lidar_odom_node',
        name='lidar_odom_node',
        output='screen',
        # Ubuntu 22.04 ships a mutually compatible NumPy/SciPy pair. Ignore
        # user-site Python packages for this process so a user-installed NumPy
        # cannot silently make the system SciPy ABI/version unsupported.
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
                'odom_frame_id': 'odom',
                'base_frame_id': BASE_FRAME,
                'laser_frame_id': LIDAR_FRAME,
                'imu_frame_id': ParameterValue(
                    LaunchConfiguration('imu_frame'),
                    value_type=str,
                ),
                'publish_tf': True,
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

    map_server = LifecycleNode(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        namespace='',
        output='screen',
        parameters=[{
            'yaml_filename': map_filename,
            'use_sim_time': use_sim_time,
        }],
    )

    amcl = LifecycleNode(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        namespace='',
        output='screen',
        parameters=[
            params_filename,
            {'use_sim_time': use_sim_time},
        ],
        remappings=[
            ('scan', LaunchConfiguration('scan_topic')),
        ],
    )

    vehicle_marker_node = Node(
        package='cartographer_2d_slam',
        executable='vehicle_marker_publisher',
        name='vehicle_marker_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'pose_topic': '/amcl_pose',
            'output_frame': MAP_FRAME,
        }],
    )

    # Configure and activate the map server first. AMCL is configured only
    # after /map is available, then activated automatically as well.
    configure_map_server = RegisterEventHandler(
        OnProcessStart(
            target_action=map_server,
            on_start=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(map_server),
                        transition_id=Transition.TRANSITION_CONFIGURE,
                    ),
                ),
            ],
        ),
    )

    activate_map_server = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=map_server,
            start_state='configuring',
            goal_state='inactive',
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(map_server),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    ),
                ),
            ],
        ),
    )

    configure_amcl = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=map_server,
            start_state='activating',
            goal_state='active',
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(amcl),
                        transition_id=Transition.TRANSITION_CONFIGURE,
                    ),
                ),
            ],
        ),
    )

    activate_amcl = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=amcl,
            start_state='configuring',
            goal_state='inactive',
            entities=[
                EmitEvent(
                    event=ChangeState(
                        lifecycle_node_matcher=matches_action(amcl),
                        transition_id=Transition.TRANSITION_ACTIVATE,
                    ),
                ),
            ],
        ),
    )

    return [
        lidar_static_tf,
        imu_static_tf,
        rear_scan_filter_node,
        lidar_odom_node,
        configure_map_server,
        activate_map_server,
        configure_amcl,
        activate_amcl,
        # Start AMCL unconfigured before the map server. Once the map server
        # reaches active, the lifecycle handlers above configure AMCL.
        amcl,
        map_server,
        vehicle_marker_node,
    ]


def generate_launch_description():
    """Build the standard-map AMCL localization launch description."""
    package_root = get_package_share_directory('cartographer_2d_slam')
    default_map = os.path.join(
        package_root, 'maps', 'kmu_parking.yaml'
    )
    default_params = os.path.join(
        package_root,
        'config',
        'amcl.yaml',
    )
    default_odom_params = os.path.join(
        package_root,
        'config',
        'lidar_odom.yaml',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Absolute path to a Nav2 map YAML file.',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Absolute path to the AMCL parameter YAML file.',
        ),
        DeclareLaunchArgument(
            'odom_params_file',
            default_value=default_odom_params,
            description='Absolute path to the LiDAR odometry parameter YAML.',
        ),
        DeclareLaunchArgument(
            'scan_topic',
            default_value=FILTERED_SCAN_TOPIC,
            description='LaserScan topic consumed by AMCL.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use the simulation clock when true.',
        ),
        DeclareLaunchArgument(
            'imu_topic',
            default_value='/imu',
            description='Gyroscope topic used to seed LiDAR ICP yaw.',
        ),
        DeclareLaunchArgument(
            'imu_frame',
            default_value=IMU_FRAME,
            description='Frame ID published by the IMU driver.',
        ),
        DeclareLaunchArgument(
            'lidar_x',
            default_value=LIDAR_XYZ[0],
            description='LiDAR x offset from base_link in metres.',
        ),
        DeclareLaunchArgument(
            'lidar_y',
            default_value=LIDAR_XYZ[1],
            description='LiDAR y offset from base_link in metres.',
        ),
        DeclareLaunchArgument(
            'lidar_z',
            default_value=LIDAR_XYZ[2],
            description='LiDAR z offset from base_link in metres.',
        ),
        DeclareLaunchArgument(
            'lidar_yaw',
            default_value=LEVEL_RPY[2],
            description='LiDAR yaw offset from base_link in radians.',
        ),
        DeclareLaunchArgument('imu_x', default_value=IMU_XYZ[0]),
        DeclareLaunchArgument('imu_y', default_value=IMU_XYZ[1]),
        DeclareLaunchArgument('imu_z', default_value=IMU_XYZ[2]),
        DeclareLaunchArgument('imu_roll', default_value=LEVEL_RPY[0]),
        DeclareLaunchArgument('imu_pitch', default_value=LEVEL_RPY[1]),
        DeclareLaunchArgument('imu_yaw', default_value=LEVEL_RPY[2]),
        OpaqueFunction(function=_localization_nodes),
    ])
