#!/usr/bin/env python3

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
PACKAGE_NAME = 'cartographer_2d_slam'

LIDAR_XYZ = ('0.410', '0.0', '0.04')
LEVEL_RPY = ('0.0', '0.0', '0.0')

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

    rear_scan_filter_node = Node(
        package=PACKAGE_NAME,
        executable='rear_scan_filter',
        name='rear_scan_filter',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # AMCL consumes odom -> base_link from scan-to-submap ICP.  The geometry
    # overrides below are shared with the static TF to prevent frame mismatch.
    lidar_odom_node = Node(
        package=PACKAGE_NAME,
        executable='lidar_odom_node',
        name='lidar_odom_node',
        output='screen',
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
                'publish_tf': True,
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
    ]


def generate_launch_description():
    """Build the standard-map AMCL localization launch description."""
    package_root = get_package_share_directory(PACKAGE_NAME)
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
            description='Required path to a Nav2 map YAML file.',
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
            'use_imu',
            default_value='true',
            description='Use IMU yaw rate to seed LiDAR ICP.',
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
        OpaqueFunction(function=_localization_nodes),
    ])
