"""Setuptools configuration for the KMU ROS 2 package."""

from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'kmu'


def existing_files(pattern):
    """Return valid files while excluding stale colcon build symlinks."""
    return [path for path in glob(pattern) if os.path.isfile(path)]


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            existing_files('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            existing_files('config/*.lua')
            + existing_files('config/*.yaml')
            + existing_files('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    package_data={package_name: ['*.pt']},
    include_package_data=True,
    zip_safe=True,
    maintainer='alscks',
    maintainer_email='kimmc_22250@dgu.ac.kr',
    description='KMU track-driving perception and DWA control nodes for ROS 2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'camera_node = kmu.camera_node:main',
            'track_drive = kmu.track_drive:main',
        ],
    },
)
