import numpy as np
import time
LANE_ONE_HOLD_SEC = 0.5   # 1차로 체류 확정 시간 (기존 하드코딩 1.0)
LANE_ONE_MAX_SEC  = 6.0   # 보정 포기 타임아웃 — 실측 필요
STATIC_ARM_FRAMES = 2     # 정적차 확정에 필요한 연속 검출 틱 (false→true→true)
STATIC_MAX_SEC    = 2.5   # STATIC 포기 타임아웃 — 오검출 고착 시 탈출구

class StateMachine:
    def __init__(self):
        self.state = "LANE_DRIVE"
        # Post-left-turn lane correction timers.  These belong to the state
        # machine and must exist before LANE_DRIVE can inspect them.
        self.lane_one_start = None
        self.lane_one_t = None
        self.tim = False
        self.tim_start = None
        # STATIC(정적차) 구간 상태. prev_static 자체는 track_drive.__init__ 이
        # deque(maxlen=2) 로 만든다 — 여기서는 판정 플래그와 진입 시각만 든다.
        self.static_armed = False   # false→true→true 를 본 적이 있는가
        self.static_t = None        # STATIC 진입 시각 (타임아웃 기준)
        
        self.state_dict = {
            "START": self.state_start,
            # "INTERSECTION_WAIT": self.state_intersection_wait,
            "LANE_DRIVE": self.state_lane_drive,
            "OVERTAKE": self.state_overtake,
            "LANE_LEFT": self.state_laneleft,
            'GO': self.state_GO,
            'STATIC':self.state_static,

        }

    def run(self):
        self.lidar_scan_process()
        self.lidar_prepro()
        self.traffic_edge()
        self.state_dict[self.state]()
        self.drive(self.steer, self.speed)
        

    def traffic_edge(self):
        if self.prev_traffic == 'LEFT' and self.traffic == 'NONE':
            self.left_pending = True
        self.prev_traffic = self.traffic

    def shortcut_edge(self, detected):
        """두 번 연속 검출된 shortcut이 사라지는 순간을 latch한다."""
        self.shortcut_detected = bool(detected)

        if self.prev_shortcut_detected and not self.shortcut_detected:
            self.shortcut = True

        self.shortcut_detection_history.append(self.shortcut_detected)
        self.prev_shortcut_detected = (
            len(self.shortcut_detection_history) == 3
            and all(self.shortcut_detection_history)
        )
    def state_static(self):

        self.speed = 10.5
        self.steer = self.lane_steer

        if self.overtake_mode:
            self.state = "OVERTAKE"
            return

        self.prev_static.append(self.static_detected)   # 매 틱 이력 갱신

        if (len(self.prev_static) >= STATIC_ARM_FRAMES
                and all(self.prev_static)):
            self.static_armed = True

        # 하강엣지: 정적차를 확실히 본 뒤에 사라진 경우에만 풀어준다
        if self.static_armed and not self.static_detected:
            self.state = 'LANE_DRIVE'
            return

        # 진입 시각이 없으면(짝 어긋남) 지금을 기준으로 잡아 None 뺄셈을 피한다
        if self.static_t is None:
            self.static_t = time.time()
        if time.time() - self.static_t > STATIC_MAX_SEC:
            self.state = 'LANE_DRIVE'
            return


    def state_GO(self):
        self.speed = 17.
        self.steer = np.rad2deg(self.dwa_steer()) * (-2.8)
        if not self.cone_detect:
            # 이전 GO 사이클의 이력이 남아 있으면 무장 판정이 오염된다.
            # 관측은 state_static 이 매 틱 직접 쌓으므로 여기서는 비우기만 한다.
            self.prev_static.clear()
            self.static_armed = False
            self.static_t = time.time()
            self.state = "STATIC"
            return
    
    def state_start(self):
        if self.traffic == "GREEN":
            self.state = 'LANE_DRIVE'
        else:
            self.speed = 0.
            self.steer = 0.

    def state_laneleft(self):
        self.speed = 20.
        self.lane_one = False
        elapsed = time.time() - self.left_t_trigger
    
        if elapsed < 0.35: # 실측할것
            # 좌회전 시작 전
            self.steer = self.lane_steer
        elif not self.shortcut_detected_on: # 실측할것
            self.steer = self.lane_steer*0.2 -33.
        else:
            # 좌회전 완료 # 실측할것
            self.left_done = True
            self.steer = self.lane_steer
            self.state = 'LANE_DRIVE'
            return
        
    def state_lane_drive(self):
        if self.traffic == 'RED' or self.traffic == 'YELLOW':
            self.speed = 0.
            return
        
        self.speed = self.target_speed
        self.steer = self.lane_steer 
        
        if self.checkerboard_detected:
            if self.steer > 22.:
                self.steer = self.lane_steer*0.2
            elif self.steer < -22.:
                self.steer = self.lane_steer*0.2
                
        
        now = time.time()
        if self.tim:
            self.speed = 30.
            self.steer =  - 35.
            if now - self.tim_start > 0.6:
                self.tim = False
                self.tim_start = None
            return

        if self.left_done and self.shortcut:
            self.shortcut = False
            self.tim = True
            self.tim_start = now
            return
        
        if self.lane_one is False and self.left_done:
            if self.lane_one_start is None:
                self.lane_one_start = now

            if self.current_lane == 1:
                if self.lane_one_t is None:
                    self.lane_one_t = now
                self.steer = self.lane_steer
                if now - self.lane_one_t > LANE_ONE_HOLD_SEC:
                    self.lane_one = True        # 보정 영구 종료
            elif self.current_lane == 2:
                self.lane_one_t = None          # 유지 타이머 리셋
                self.steer = self.lane_steer * 0.25 - 12.
            else:
                self.lane_one_t = None
                self.steer = self.lane_steer

            if now - self.lane_one_start > LANE_ONE_MAX_SEC:
                self.lane_one = True

            return


        if self.cone_detect and self.cone_cam_detected:
            self.dwa_time = False
            self.dwa_time_t = None
            self.state = 'GO'
            return

        # 좌회전은 LEFT 가 보이는 동안이 아니라 LEFT 가 꺼진 순간부터 시작한다.
        # 엣지 판정 자체는 traffic_edge() 가 하고 여기서는 플래그만 소비한다.
        if self.left_pending:
            self.left_pending = False
            self.left_done = False
            self.left_t_trigger = time.time()
            self.lane_one_start = None
            self.lane_one_t = None
            self.state = 'LANE_LEFT'
            return

        # 추월
        if self.overtake_mode:
            self.state = "OVERTAKE"
            return
        

        
            
    def state_overtake(self):
        self.overtake()

        if self.overtake_done:
            # 상태 초기화는 track_drive.reset_overtake 한 곳에서만 한다.
            # 여기서 개별 변수를 나열하면 새 상태(overtake_start_t 등)를
            # 추가할 때마다 빠뜨려 다음 추월이 이전 시각으로 시작된다.
            self.reset_overtake()
            self.abs_overtake = True
            self.state = 'LANE_DRIVE'
            
            
