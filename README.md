# 2026 KMU 본선 자율주행 코드

이 브랜치는 `2026_KMU_BD`의 본선 코드 가운데 SLAM을 제외한 인지, 판단,
경로 계획 및 차량 제어 코드를 독립 ROS 2 Python 패키지로 정리한 것입니다.
SLAM과 주차 관련 코드는 `parking` 브랜치에서 관리합니다.

## 주요 구성

- `kmu/camera_node.py`: 카메라 입력과 차선·미션 인지
- `kmu/track_drive.py`: 주행 상태 머신과 구동 명령 생성
- `kmu/dwa.py`: 지역 경로 및 장애물 회피
- `kmu/planner/`: lattice/quintic 기반 경로 계획
- `kmu/total.py`: 자동·수동 구동 명령 중계
- `launch/kmu_track.launch.py`: 카메라 인지와 주행 노드 통합 실행
- `kmu/models/`, `kmu/*.pt`: 본선 주행에 사용하는 추론 모델

## 빌드

```bash
cd ~/ros2_ws
colcon build --packages-select kmu
source install/setup.bash
```

카메라와 주행 노드를 함께 실행합니다.

```bash
ros2 launch kmu kmu_track.launch.py
```

각 노드는 별도로 실행할 수도 있습니다.

```bash
ros2 launch kmu kmu_track.launch.py camera:=false
ros2 launch kmu kmu_track.launch.py track:=false
```

카메라, LiDAR, 초음파 센서 드라이버와 `custom_msgs` 패키지는 별도로 준비되어
있어야 합니다.
