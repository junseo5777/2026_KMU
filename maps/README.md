<!-- Copyright 2026 junseo5777 and kimmc53 -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 지도 설명

## kmu_parking

`kmu_parking.yaml`과 `kmu_parking.pgm`은 KMU 연습 공간을 Cartographer로
직접 매핑하여 만든 연습용 지도입니다.

- 해상도: 0.02 m/cell

## parking_map

`parking_map.yaml`과 `parking_map.pgm`은 대회 측에서 제공한 실제 대회
지도입니다.

- 해상도: 0.05 m/cell
- Map Origin: `(-2.15, -1.05, 0)`

Map Origin은 차량의 시작 Pose가 아니라 PGM 이미지를 실제 지도 좌표계에
배치하기 위한 기준값입니다.
