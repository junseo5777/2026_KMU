-- Copyright 2026 junseo5777 and kimmc53
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

include "map_builder.lua"
include "trajectory_builder.lua"

-- KMU 1:10 vehicle, 2D LiDAR SLAM with continuous LiDAR odometry.
--
-- Required TF tree before starting Cartographer:
--   base_link -> laser_frame
--             -> base_imu_link (gyro is fused by lidar_odom_node)

-- use_odometry = true 이거 오도메트리 false로 하고 해봐야함.
-- 전역 pose graph 탐색 수정해봐야함.
options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,

  map_frame = "map",                 -- 지도의 고정 전역 좌표계
  tracking_frame = "base_link",      -- Cartographer가 추적할 차량 기준점
  published_frame = "base_link",     -- 추정 pose를 발행할 차량 frame
  odom_frame = "odom",               -- 연속적인 로컬 이동 좌표계

  provide_odom_frame = true,          -- map->odom->base_link TF를 생성
  publish_frame_projected_to_2d = true, -- z/roll/pitch를 제거한 2D pose 발행
  use_pose_extrapolator = true,       -- scan 사이 pose를 시간 기준으로 예측
  use_odometry = true,                -- 연속 P2P LiDAR odometry 사용
  use_nav_sat = false,                -- GPS를 사용하지 않음
  use_landmarks = false,              -- landmark 측정을 사용하지 않음

  num_laser_scans = 1,                -- 일반 LaserScan 입력 개수
  num_multi_echo_laser_scans = 0,     -- MultiEchoLaserScan 입력 개수
  num_subdivisions_per_laser_scan = 3, -- scan 한 바퀴를 시간순 3등분
  num_point_clouds = 0,               -- PointCloud2 입력 개수

  lookup_transform_timeout_sec = 0.2, -- 센서 TF를 기다리는 최대 시간
  submap_publish_period_sec = 0.3,    -- submap 발행 주기
  pose_publish_period_sec = 0.02,     -- 차량 pose 발행 주기
  trajectory_publish_period_sec = 0.1, -- trajectory marker 발행 주기

  rangefinder_sampling_ratio = 1.0,   -- LiDAR 데이터를 모두 사용
  odometry_sampling_ratio = 1.0,      -- P2P LiDAR odometry를 모두 사용
  fixed_frame_pose_sampling_ratio = 1.0, -- GPS 등 고정 pose 샘플 비율
  imu_sampling_ratio = 1.0,           -- IMU 사용 시 샘플 비율(현재 미사용)
  landmarks_sampling_ratio = 1.0,     -- landmark 사용 시 샘플 비율
}

MAP_BUILDER.use_trajectory_builder_2d = true -- 2D SLAM trajectory builder 사용

-- Stock Cartographer IMU input also consumes linear acceleration. Keep it
-- disabled while the driver reports the wrong gravity sign. Gyro yaw is
-- instead fused by lidar_odom_node after applying the measured sign fix.
TRAJECTORY_BUILDER_2D.use_imu_data = false
-- /lidar_odom supplies the motion prediction, but Ceres-only matching was too
-- dependent on that prediction when slow motion was underestimated. Allow a
-- bounded local search: wide enough to recover normal motion, much narrower
-- in yaw than Cartographer's 20 degree default to avoid unrelated wall matches.
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 0.1

-- The YDLIDAR reports 0.1 m to 16.0 m. Use 6 m for the compact
-- parking course and increase it only if useful distant returns are observed.
TRAJECTORY_BUILDER_2D.min_range = 0.1 -- 이 거리보다 가까운 점 제거(m)
TRAJECTORY_BUILDER_2D.max_range = 7.0 -- 이 거리보다 먼 점 제거(m)
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 7.0 -- 미검출 beam의 free ray 길이(m)
TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 3 -- 정합 전 누적할 분할 scan 수

TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.02 -- submap 한 셀 크기(m)
TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter.hit_probability = 0.60 -- 검출 셀의 점유 확률
TRAJECTORY_BUILDER_2D.submaps.range_data_inserter.probability_grid_range_data_inserter.miss_probability = 0.495 -- 통과 셀의 점유 확률

TRAJECTORY_BUILDER_2D.motion_filter.max_time_seconds = 0.2 -- 정지 중 데이터 허용 간격(s)
TRAJECTORY_BUILDER_2D.motion_filter.max_distance_meters = 0.02 -- 새 node 생성 이동 기준(m)
TRAJECTORY_BUILDER_2D.motion_filter.max_angle_radians = math.rad(0.5) -- 새 node 생성 회전 기준

POSE_GRAPH.optimize_every_n_nodes = 20 -- node 20개마다 pose graph 최적화

return options
