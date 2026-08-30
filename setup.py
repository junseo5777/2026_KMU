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

import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'cartographer_2d_slam'


setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.lua')
            + glob('config/*.yaml')
            + glob('config/*.rviz'),
        ),
        (
            os.path.join('share', package_name, 'maps'),
            glob('maps/*.yaml') + glob('maps/*.pgm'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='junseo5777',
    maintainer_email='junseo5777@gmail.com',
    url='https://github.com/junseo5777/ros2-cartographer-acml',
    description=(
        'Cartographer mapping, LiDAR odometry, AMCL localization, '
        'Hybrid A* planning, and DWA/SSP following for ROS 2'
    ),
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'auto_map_saver = '
            'cartographer_2d_slam.auto_map_saver:main',
            'lidar_odom_node = '
            'cartographer_2d_slam.lidar_odom_node:main',
            'rear_scan_filter = '
            'cartographer_2d_slam.rear_scan_filter:main',
            'vehicle_marker_publisher = '
            'cartographer_2d_slam.vehicle_marker_publisher:main',
            'imu_odom_node = '
            'cartographer_2d_slam.imu_odom_node:main',
            'hybrid_astar_planner = '
            'cartographer_2d_slam.hybrid_astar_planner_node:main',
            'hybrid_dwa_follower = '
            'cartographer_2d_slam.hybrid_dwa_follower_node:main',
        ],
    },
)
