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

"""Run the isolated START-A-B-START stack on the competition map."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Start parking-map localization, follower, and sequential mission."""
    package_share = get_package_share_directory('cartographer_2d_slam')
    launch_directory = os.path.join(package_share, 'launch')
    default_map = os.path.join(
        package_share, 'maps', 'parking_map.yaml'
    )
    default_planner_params = os.path.join(
        package_share, 'config', 'hybrid_astar.yaml'
    )
    default_mission_params = os.path.join(
        package_share, 'config', 'competition_mission.yaml'
    )
    default_follower_params = os.path.join(
        package_share, 'config', 'hybrid_dwa.yaml'
    )

    use_sim_time = ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool
    )
    localization = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            launch_directory, 'localization.launch.py'
        )),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }.items(),
    )
    follower = Node(
        package='cartographer_2d_slam',
        executable='hybrid_dwa_follower',
        name='hybrid_dwa_follower',
        output='screen',
        parameters=[
            LaunchConfiguration('follower_params_file'),
            {
                'use_sim_time': use_sim_time,
                'enable_drive': ParameterValue(
                    LaunchConfiguration('enable_drive'), value_type=bool
                ),
            },
        ],
    )
    planner = Node(
        package='cartographer_2d_slam',
        executable='hybrid_astar_planner',
        name='hybrid_astar_planner',
        output='screen',
        parameters=[
            LaunchConfiguration('planner_params_file'),
            LaunchConfiguration('mission_params_file'),
            {
                'map_yaml': ParameterValue(
                    LaunchConfiguration('map'), value_type=str
                ),
                'start_mode': 'competition',
                'return_to_start': False,
                'planning_behavior': ParameterValue(
                    LaunchConfiguration('planning_behavior'), value_type=str
                ),
                'use_sim_time': use_sim_time,
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Competition Nav2 map YAML.',
        ),
        DeclareLaunchArgument(
            'planner_params_file',
            default_value=default_planner_params,
            description='Hybrid A* planner parameters.',
        ),
        DeclareLaunchArgument(
            'mission_params_file',
            default_value=default_mission_params,
            description='START/A/B mission poses and transition parameters.',
        ),
        DeclareLaunchArgument(
            'follower_params_file',
            default_value=default_follower_params,
            description='Hybrid DWA/SSP follower parameters.',
        ),
        DeclareLaunchArgument(
            'planning_behavior',
            default_value='forward_then_final_reverse',
            choices=['forward_then_final_reverse', 'legacy'],
            description='Planner behavior applied independently to each leg.',
        ),
        DeclareLaunchArgument(
            'enable_drive',
            default_value='true',
            description='Publish actuator commands; false runs shadow mode.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use the simulation clock when true.',
        ),
        localization,
        follower,
        planner,
    ])
