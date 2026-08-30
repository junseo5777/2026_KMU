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

-- Legacy Cartographer pure-localization configuration.
-- launch/localization.launch.py now uses Nav2 AMCL with config/amcl.yaml and
-- does not load this file. Keep it only as a pbstream-localization fallback.

include "parking_2d.lua"

-- Keep the trajectory loaded from the pbstream frozen and retain only a
-- small number of active submaps for matching the live LiDAR scans.
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}

-- Run pose-graph optimization frequently enough to correct the live
-- trajectory against the frozen map without rebuilding that map.
POSE_GRAPH.optimize_every_n_nodes = 20

return options
