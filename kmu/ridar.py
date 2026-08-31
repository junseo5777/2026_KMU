from collections import deque
import numpy as np


class LidarPerception:
    def __init__(self):

        self.angle_increment = None  # 첫 콜백에서 채워짐

        self.cluster_threshold = 0.16  # 실측 후 조정
        self.min_cluster_points = 3

        # 라바콘 판별용 임계값 (실측 캘리브레이션 필요)
        self.min_cone_width = 0.045
        self.max_cone_width = 0.10
        self.min_cone_count = 2          # 프레임당 콘 개수 임계값
        self.history_len = 3            # 최근 몇 프레임 볼지
        self.min_positive_frames = 1    # 그 중 몇 프레임이 조건 만족해야 하는지
        self.cone_count_history = deque(maxlen=self.history_len)


        self.static_y_max = 2.3      # 룩어헤드 (실측 필요)
        self.static_y_min = 0.02
        self.static_x_max = 0.2
        self.static_x_min = -0.2
        self.min_band_points = 2
        

        # 차량 판별용 임계값
        self.min_car_width = 0.10
        self.max_car_width = 0.24
        self.perp_spread_threshold = 0.02
        # self.car_min_positive_frames = 1    
        # self.car_count_history = deque(maxlen=self.history_len)

        #x 값 band 
        self.x_min = -0.25
        self.x_max = 0.25
        self.ratio_threshold = 0.8

    def clusterize(self, points):
        if len(points) < 2:
            return []

        gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        break_idx = np.where(gaps > self.cluster_threshold)[0] + 1
        clusters = np.split(points, break_idx)

        return [c for c in clusters if len(c) >= self.min_cluster_points]

    def classfi(self, clusters):
        cone_points = []
        car_points = []

    
        for cluster in clusters:
            cluster_center = np.mean(cluster, axis=0)
            d = np.linalg.norm(cluster_center)

            cluster_width = (
                np.linalg.norm(cluster[0] - cluster[-1]) + d * self.angle_increment
            )


            is_cone = self.min_cone_width < cluster_width < self.max_cone_width
            car_width  = self.min_car_width  < cluster_width < self.max_car_width
            point_len = len(cluster)

            # in_band = (cluster[:,0] >= self.x_min) & (cluster[:,0]<= self.x_max)
            # n_in = int(np.count_nonzero(in_band))
            # ratio = n_in /len(cluster)


            if is_cone:
                cone_points.append(cluster_center) 
     

            if car_width:
                in_band = (cluster[:, 0] >= self.x_min) & (cluster[:, 0] <= self.x_max)
                if np.count_nonzero(in_band) / len(cluster) >= self.ratio_threshold:
                    eigvals = np.linalg.eigh(np.cov(cluster.T))[0]
                    perp_spread = np.sqrt(max(eigvals[0], 0.0))
                    if perp_spread <= self.perp_spread_threshold:
                        car_points.append(cluster_center)

        

        cone_detect = self.vote(len(cone_points)) 
        car_detect = "car" if len(car_points)>= 1 else ""


        return cone_detect, car_detect, car_points


    def vote(self, n_cone):
        # ── 콘 판정 (프레임 히스토리 기반) ──
        self.cone_count_history.append(n_cone)
        cone_positive_frames = sum(1 for c in self.cone_count_history if c >= self.min_cone_count)
        cone_detect = "rubbercone" if cone_positive_frames >= self.min_positive_frames else ""


        return cone_detect

    def detect_static(self, points):
        if len(points) == 0:
            return ""
        x, y = points[:, 0], points[:, 1]
        m = ((x >= self.static_x_min) & (x <= self.static_x_max) &
            (y >= self.static_y_min) & (y <= self.static_y_max))
        if np.count_nonzero(m) < self.min_band_points:
            return ""
        return "static"