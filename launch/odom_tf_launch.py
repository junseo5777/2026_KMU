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

"""
odom->base_link와 base_link->laser TF를 함께 띄우는 launch.

라이다 실장 위치(x, y, z)와 자세(roll, pitch, yaw)는 실측해서 아래 기본값을 바꾸거나,
launch argument로 기본값을 덮어쓴다.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("imu_topic", default_value="/imu"),
            DeclareLaunchArgument("odom_frame_id", default_value="odom"),
            DeclareLaunchArgument("base_frame_id", default_value="base_link"),
            DeclareLaunchArgument("laser_frame_id", default_value="laser_frame"),
            # 라이다가 base_link 원점(보통 후륜축 중심) 기준으로 어디에, 얼마나 돌아가서
            # 붙어있는지를 실측해서 채운다. 예: 차체 중앙에서 앞으로 0.3m, 지상 0.2m 위.
            DeclareLaunchArgument("lidar_x", default_value="0.410"),
            DeclareLaunchArgument("lidar_y", default_value="0.0"),
            DeclareLaunchArgument("lidar_z", default_value="0.04"),
            DeclareLaunchArgument("lidar_roll", default_value="0.0"),
            DeclareLaunchArgument("lidar_pitch", default_value="0.0"),
            DeclareLaunchArgument("lidar_yaw", default_value="0.0"),
            Node(
                package="cartographer_2d_slam",
                executable="imu_odom_node",
                name="imu_odom_node",
                output="screen",
                parameters=[
                    {
                        "imu_topic": LaunchConfiguration("imu_topic"),
                        "odom_frame_id": LaunchConfiguration("odom_frame_id"),
                        "base_frame_id": LaunchConfiguration("base_frame_id"),
                    }
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_to_laser_tf",
                output="screen",
                arguments=[
                    "--x", LaunchConfiguration("lidar_x"),
                    "--y", LaunchConfiguration("lidar_y"),
                    "--z", LaunchConfiguration("lidar_z"),
                    "--roll", LaunchConfiguration("lidar_roll"),
                    "--pitch", LaunchConfiguration("lidar_pitch"),
                    "--yaw", LaunchConfiguration("lidar_yaw"),
                    "--frame-id", LaunchConfiguration("base_frame_id"),
                    "--child-frame-id", LaunchConfiguration("laser_frame_id"),
                ],
            ),
        ]
    )
