# Cartographer 2D SLAM

ROS 2 Humble 환경에서 2D LiDAR로 지도를 생성하고, 저장한 지도에서 Nav2
AMCL로 차량 위치를 추정하는 패키지입니다.

소형 Ackermann 차량과 전방 장착 LiDAR를 기준으로 작성되었습니다.

## 주요 기능

- Cartographer 기반 LiDAR-only 2D 매핑
- 차체에 가려지는 후방 LiDAR 데이터 제거
- Nav2 형식의 YAML/PGM 지도 자동 저장
- Cartographer PBStream 저장
- LiDAR ICP odometry와 AMCL 기반 로컬라이제이션
- IMU yaw rate를 이용한 LiDAR odometry 보조

## 동작 흐름

### Cartographer 매핑

1. LiDAR 드라이버가 `/scan`을 발행합니다.
2. `rear_scan_filter`가 후방 데이터를 제거하고 `/scan_slam`을 발행합니다.
3. Cartographer가 `/scan_slam`으로 차량 위치와 지도를 계산합니다.
4. Occupancy grid가 `/map`으로 발행됩니다.
5. `auto_map_saver`가 지도를 YAML과 PGM 파일로 저장합니다.
6. launch를 정상 종료하면 Cartographer 상태가 PBStream으로 저장됩니다.

매핑 중에는 Cartographer가 `map`, `odom`, `base_link` 사이의 TF를 담당합니다.
다른 odometry 노드가 `odom → base_link`를 동시에 발행하면 안 됩니다.

### AMCL 로컬라이제이션

1. Nav2 map server가 저장된 YAML/PGM 지도를 불러옵니다.
2. `rear_scan_filter`가 `/scan`을 `/scan_slam`으로 변환합니다.
3. `lidar_odom_node`가 `/scan_slam`과 선택적으로 `/imu`를 사용해
   `odom → base_link`를 계산합니다.
4. AMCL이 지도와 `/scan_slam`을 비교해 `map → odom`을 계산합니다.
5. RViz의 **2D Pose Estimate**로 초기 위치를 지정합니다.

AMCL은 PBStream 파일을 사용하지 않습니다. 매핑으로 생성된 YAML 파일과 해당
YAML에서 참조하는 PGM 또는 PNG 이미지가 필요합니다.

## 주요 파일

### 매핑

- [`slam_mapping.launch.py`](launch/slam_mapping.launch.py)는 scan 필터,
  Cartographer, occupancy grid, 지도 저장 노드를 한 번에 실행합니다.
- [`parking_2d.lua`](config/parking_2d.lua)는 LiDAR 사용 거리, 지도 해상도,
  scan matching과 pose graph 최적화 값을 관리합니다.
- [`auto_map_saver.py`](cartographer_2d_slam/auto_map_saver.py)와
  [`map_io.py`](cartographer_2d_slam/map_io.py)는 `/map`을 Nav2 형식의
  YAML/PGM 파일로 저장합니다.

### 로컬라이제이션

- [`localization.launch.py`](launch/localization.launch.py)는 저장된 지도와
  LiDAR odometry, AMCL을 순서대로 실행합니다.
- [`lidar_odom_node.py`](cartographer_2d_slam/lidar_odom_node.py)는 LiDAR와
  선택적으로 IMU를 사용해 `odom → base_link`를 계산합니다.
- [`lidar_odometry.py`](cartographer_2d_slam/lidar_odometry.py)는 ICP와 2D
  좌표 변환 계산을 담당하며, 세부 값은
  [`lidar_odom.yaml`](config/lidar_odom.yaml)에서 설정합니다.
- [`amcl.yaml`](config/amcl.yaml)은 AMCL의 particle, motion model, LiDAR
  sensor model을 설정합니다.

### 공통 처리

[`rear_scan_filter.py`](cartographer_2d_slam/rear_scan_filter.py)는 원본
`/scan`에서 차체에 가려지는 후방 데이터를 제거하고 `/scan_slam`을
발행합니다. 매핑과 로컬라이제이션에서 동일하게 사용합니다.

## 기본 설정

| 항목 | 기본값 |
|---|---|
| 원본 LiDAR topic | `/scan` |
| 필터 LiDAR topic | `/scan_slam` |
| IMU topic | `/imu` |
| LiDAR frame | `laser_frame` |
| 차량 frame | `base_link` |
| LiDAR 장착 위치 | `x=0.410`, `y=0.0`, `z=0.040 m` |
| LiDAR 장착 yaw | `0.0 rad` |
| 사용하는 LiDAR 영역 | 전방 -90° ~ +90° |

LiDAR의 실제 장착 위치가 다르면 두 launch 파일의 `lidar_x`, `lidar_y`,
`lidar_z`, `lidar_yaw` 값을 변경해야 합니다. 로컬라이제이션에서는
`config/lidar_odom.yaml`의 장착값도 동일하게 유지해야 합니다.

## Cartographer 주요 파라미터

Cartographer 설정은 `config/parking_2d.lua`에서 변경합니다.

| 파라미터 | 기본값 | 설명 |
|---|---:|---|
| `min_range` | `0.1 m` | 이 거리보다 가까운 LiDAR 측정값을 제외합니다. |
| `max_range` | `6.0 m` | 이 거리보다 먼 측정값을 제외합니다. |
| `num_subdivisions_per_laser_scan` | `3` | 한 번의 scan을 시간순으로 나눠 이동 중 왜곡을 줄입니다. |
| `num_accumulated_range_data` | `3` | scan matching 전에 누적할 분할 데이터 수입니다. |
| submap `resolution` | `0.02 m` | 지도 셀 크기입니다. 낮을수록 정밀하지만 계산량이 증가합니다. |
| `use_online_correlative_scan_matching` | `true` | 주변 위치를 추가 탐색해 scan matching 안정성을 높입니다. |
| motion filter 거리/각도 | `0.02 m / 0.5°` | 작은 움직임에서 불필요한 node 생성을 줄입니다. |
| `optimize_every_n_nodes` | `20` | node 20개마다 pose graph를 최적화합니다. |

현재 매핑은 `use_imu_data=false`, `use_odometry=false`인 LiDAR-only
구성입니다. `max_range`는 주변 환경과 LiDAR 성능에 맞게 조정해야 합니다.
submap 해상도를 변경할 때는 `launch/slam_mapping.launch.py`의
`MAP_RESOLUTION`도 같은 값으로 변경하는 것이 좋습니다.
