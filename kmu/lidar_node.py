import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from custom_msgs.msg import LidarData
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# 센서 원본 구독용 (Unity 발행 RELIABLE 에 맞춤)
# QOS_SENSOR = QoSProfile(
#     reliability=ReliabilityPolicy.RELIABLE,
#     history=HistoryPolicy.KEEP_LAST,
#     depth=1,
# )

# # 처리 결과/제어 명령용 (내부 노드 간)
# QOS_CMD = QoSProfile(
#     reliability=ReliabilityPolicy.RELIABLE,
#     history=HistoryPolicy.KEEP_LAST,
#     depth=1,
# )

class LidarPublisher(Node):
    def __init__(self):
        super().__init__('lidar_publisher')
        self.msg = None
        self.CLUSTER_DIST = 0.5
        self.MIN_POINTS = 5
        self.LINE_THRESH = 0.05
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, qos)
        self.lidar_pub = self.create_publisher(LidarData, '/lidar_data', qos)
        self.create_timer(0.05, self.timer_callback)

    def lidar_callback(self, msg):
        self.msg = msg

    def origin_roi_point(self,x_min,x_max,y_min,y_max):
        points = []

        for i, r in enumerate(self.msg.ranges):
            if not math.isfinite(r):
                continue
            angle = self.msg.angle_min + i * self.msg.angle_increment - math.radians(90)
            x = r * math.cos(angle)
            y = r * math.sin(angle)

            if y_min < y < y_max and x_min < x < x_max:
                points.append((x,y))
        
        return points

    def process_roi(self, x_min, x_max, y_min, y_max):
        roi_points = []

        for i, r in enumerate(self.msg.ranges):
            if not math.isfinite(r):
                continue
            angle = self.msg.angle_min + i * self.msg.angle_increment + math.radians(90)
            x = r * math.cos(angle)
            y = r * math.sin(angle)

            if y_min < y < y_max and x_min < x < x_max:
                roi_points.append((x, y))

        cone_list, car_list, car_side_list, police_list = [], [], [], []

        if len(roi_points) < 2:
            return cone_list, car_list, car_side_list, police_list

    # --- 클러스터링 ---
        clustering = []
        current = [roi_points[0]]

        for i in range(1, len(roi_points)):
            dx = roi_points[i][0] - roi_points[i-1][0]
            dy = roi_points[i][1] - roi_points[i-1][1]
            dist = math.hypot(dx, dy)

            if dist <= self.CLUSTER_DIST:
                current.append(roi_points[i])
            else:
                clustering.append(current)
                current = [roi_points[i]]

        clustering.append(current)   # 마지막 클러스터 추가

    # --- 분류 ---
        for s in clustering:
            if len(s) < 2:
                continue

            cluster_x = np.array([k[0] for k in s])
            cluster_y = np.array([k[1] for k in s])
            avg_x = float(np.mean(cluster_x))
            avg_y = float(np.mean(cluster_y))

            x_size = max(cluster_x) - min(cluster_x)
            y_size = max(cluster_y) - min(cluster_y)
            size = math.hypot(x_size, y_size)

            coeffs = np.polyfit(cluster_x, cluster_y, 1)
            slope = abs(coeffs[0])


        # 콘: 점 개수 상관없이 크기로만 판별
            if size <= 0.15:
                cone_list.append((avg_x, avg_y))


        # 차 후보: 점이 충분할 때만 직선성 검사
            if 0.7 <= size <= 4 and len(s) >= self.MIN_POINTS:
                pts = np.column_stack([cluster_x, cluster_y])
                eigvals = np.linalg.eigh(np.cov(pts.T))[0]   # 오름차순
                perp_spread = math.sqrt(max(eigvals[0], 0.0))  # 작은 고유값 = 수직 퍼짐

                if perp_spread <= self.LINE_THRESH:
                    if slope <1:
                        car_list.append((avg_x, avg_y))
                    else:
                        car_side_list.append((avg_x, avg_y))

                else:
                    A = np.array(s[0]); B = np.array(s[-1])
                    AB = B - A
                    AP = pts - A                                  # pts는 이미 Nx2로 있음
                    cross = AB[0]*AP[:,1] - AB[1]*AP[:,0]          # 각 점의 외적(스칼라들)
                    dist = np.abs(cross) / np.hypot(AB[0], AB[1])  # 각 점의 수직거리
                    k = int(np.argmax(dist)) # 가장 멀리 튀어나온 점 = 코너
                    arm1 = s[:k+1]
                    arm2 = s[k:]

                    arm1_x = np.array([t[0] for t in arm1])
                    arm1_y = np.array([t[1] for t in arm1])

                    arm2_x = np.array([t[0] for t in arm2])
                    arm2_y = np.array([t[1] for t in arm2])

                    pts = np.column_stack([arm1_x, arm1_y])
                    eigvals = np.linalg.eigh(np.cov(pts.T))[0]   # 오름차순
                    arm1_spread = math.sqrt(max(eigvals[0], 0.0))  # 작은 고유값 = 수직 퍼짐

                    pts = np.column_stack([arm2_x, arm2_y])
                    eigvals = np.linalg.eigh(np.cov(pts.T))[0]   # 오름차순
                    arm2_spread = math.sqrt(max(eigvals[0], 0.0))  # 작은 고유값 = 수직 퍼짐
                    

                    if arm1_spread <= self.LINE_THRESH and arm2_spread <= self.LINE_THRESH:
                        police_list.append((avg_x, avg_y))
            

        return cone_list, car_list, car_side_list, police_list
    
    def police_detected(self):
        if (self.w_cone and self.w_car) or (self.w_side_car and self.w_cone) or (self.w_police and self.w_cone):
            return "police_car"

        else:   
            return "None"
        
        
    def timer_callback(self):   
        if self.msg is None:
            return
        
        self.s_cone, self.s_car, self.s_side_car, self.s_police = self.process_roi(-5., 5., 0, 8.73)    # small
        self.w_cone, self.w_car, self.w_side_car, self.w_police = self.process_roi(-7, 1, 0, 8)   # wide
        origin_points = self.origin_roi_point(-4, 4, 0, 20)

        msg_out = LidarData()

        small_obs = self.s_cone + self.s_car + self.s_side_car
        msg_out.small_x = [float(p[0]) for p in small_obs]
        msg_out.small_y = [float(p[1]) for p in small_obs]


        
        msg_out.origin_x = [float(k[0]) for k in origin_points]
        msg_out.origin_y = [float(k[1]) for k in origin_points]


        detect = self.police_detected()
        msg_out.detect = detect


        car_count = len(self.s_car)
        msg_out.car_count = car_count

        self.lidar_pub.publish(msg_out)
        
def main(args=None):
    rclpy.init(args=args)
    node = LidarPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()