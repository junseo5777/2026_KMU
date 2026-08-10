import os
from glob import glob

from setuptools import find_packages, setup


package_name = 'cartographer_2d_slam'


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py'),
        ),
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.lua') + glob('config/*.yaml'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=False,
    maintainer='junseo5777',
    maintainer_email='junseo5777@gmail.com',
    description=(
        'LiDAR-only Cartographer mapping and Nav2 AMCL localization for ROS 2'
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
        ],
    },
)
