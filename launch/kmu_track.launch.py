#!/usr/bin/env python3

"""Run the track-driving stack: lane perception plus the driving state machine.

두 노드를 한 번에 띄운다.

    camera_node   /image_raw (sensor_msgs/Image)  →  /cam_data (kmu_msgs/CamData)
    track_drive   /cam_data + /scan               →  /erp_writes (geometry_msgs/Twist)

camera_node 가 차선 인식(lane_detection*.py)을 돌려 조향/속도/차로 정보를
/cam_data 로 발행하고, track_drive(노드 이름 'driver')가 그것과 라이다를 합쳐
최종 구동 명령을 낸다. 즉 둘은 /cam_data 하나로 직결되므로 리맵이 필요 없다.

★ 이 파일은 두 노드만 띄운다 ★ 아래 입력은 외부 드라이버가 제공해야 한다:
    /image_raw          카메라 드라이버 (usb_cam 등)
    /scan               라이다 드라이버
    /xycar_ultrasonic   초음파 (track_drive.USE_ULTRA 가 True 일 때만)
드라이버가 안 떠 있으면 두 노드는 죽지 않고 입력을 기다리기만 한다.

사용:
    ros2 launch kmu kmu_track.launch.py
    ros2 launch kmu kmu_track.launch.py camera:=false   # track_drive 만
    ros2 launch kmu kmu_track.launch.py track:=false    # camera_node 만 (인식 확인용)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    """Start camera_node and track_drive together."""
    # name 은 각 노드가 스스로 쓰는 이름과 같게 둔다 (camera_node.py:46 의
    # 'camera_node', track_drive.py:87 의 'driver'). 다른 이름을 주면 런치가
    # 노드명을 덮어써서 ros2 node list 결과가 소스와 어긋난다.
    camera = Node(
        package='kmu',
        executable='camera_node',
        name='camera_node',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('camera')),
    )
    track = Node(
        package='kmu',
        executable='track_drive',
        name='driver',
        output='screen',
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration('track')),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera',
            default_value='true',
            description='Start camera_node (lane perception).',
        ),
        DeclareLaunchArgument(
            'track',
            default_value='true',
            description='Start track_drive (driving state machine).',
        ),
        camera,
        track,
    ])
