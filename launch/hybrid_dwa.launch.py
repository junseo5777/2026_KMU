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

"""Run the path-guided DWA follower without changing the legacy DWA."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Start the follower using only the checked-in YAML parameters."""
    package_share = get_package_share_directory('cartographer_2d_slam')
    default_params = os.path.join(
        package_share, 'config', 'hybrid_dwa.yaml'
    )
    follower = Node(
        package='cartographer_2d_slam',
        executable='hybrid_dwa_follower',
        name='hybrid_dwa_follower',
        output='screen',
        parameters=[default_params],
    )
    return LaunchDescription([follower])
