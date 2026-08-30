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

"""Run the KMU Hybrid A* planner on an existing localization."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Start the planner with the selected map and start-pose mode."""
    package_share = get_package_share_directory('cartographer_2d_slam')
    default_map = os.path.join(
        package_share, 'maps', 'kmu_parking.yaml'
    )
    default_params = os.path.join(
        package_share, 'config', 'hybrid_astar.yaml'
    )
    planner = Node(
        package='cartographer_2d_slam',
        executable='hybrid_astar_planner',
        name='hybrid_astar_planner',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'map_yaml': ParameterValue(
                    LaunchConfiguration('map'), value_type=str
                ),
                'start_mode': ParameterValue(
                    LaunchConfiguration('start_mode'), value_type=str
                ),
                'planning_behavior': ParameterValue(
                    LaunchConfiguration('planning_behavior'), value_type=str
                ),
                'use_sim_time': ParameterValue(
                    LaunchConfiguration('use_sim_time'), value_type=bool
                ),
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=default_map,
            description='Nav2 map YAML used by localization and planning.',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Hybrid A* planner parameter YAML.',
        ),
        DeclareLaunchArgument(
            'start_mode',
            default_value='preview',
            choices=['preview', 'localization'],
            description=(
                'preview uses /initialpose; localization uses /amcl_pose.'
            ),
        ),
        DeclareLaunchArgument(
            'planning_behavior',
            default_value='forward_then_final_reverse',
            choices=['forward_then_final_reverse', 'legacy'],
            description=(
                'Constrained final-reverse behavior or original planner.'
            ),
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use the simulation clock when true.',
        ),
        planner,
    ])
