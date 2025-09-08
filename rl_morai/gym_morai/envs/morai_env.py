import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rospy
import time
import subprocess
import cv2
import sys

from .morai_sensor import MoraiSensor
from src.utils import Cal_CTE

# 사용자 정의 예외 클래스
class SensorTimeoutError(Exception):
    """타임아웃 동안 유효한 센서 데이터를 받지 못한 경우 발생하는 예외"""
    pass

def press_key(key, delay=0.3):
    """
    Simulator 창에 지정된 키 입력을 보냄.
    :param key: 'q', 'i' 등 입력할 키
    :param delay: 키 입력 후 대기 시간
    """
    try:
        window_id = subprocess.check_output(
            ['xdotool', 'search', '--name', 'Simulator']
        ).decode().strip().split('\n')[-1]

        subprocess.run(
            ['xdotool', 'windowactivate', '--sync', window_id, 'key', key],
            check=True
        )
        time.sleep(delay)

    except subprocess.CalledProcessError as e:
        rospy.logerr(f"press_key('{key}') 실패: {e}")

class MoraiEnv(gym.Env):
    def __init__(self, reward_fn=None, terminated_fn=None, action_bounds=None, csv_path=None, lookahead=5):
        super(MoraiEnv, self).__init__()
        rospy.init_node('morai_rl_env', anonymous=True)

        self.sensor = MoraiSensor(csv_path=csv_path)
        self._reward_fn = reward_fn
        self._terminated_fn = terminated_fn
        self._first_reset = True
        
        self.lookahead = lookahead #앞의 경로를 얼마나 볼지 결정
        self.path = Cal_CTE.load_centerline(csv_path)
        self.index = 0
        obs_dim = 3 + 2*self.lookahead

        self.last_steering = 0.0

        # 액션 설정
        if action_bounds is None:
            action_bounds = [(-0.7, 0.7), (10.0, 30.0)]  # 기본: 조향 [-0.7, 0.7], 스로틀 [10, 30]

        low = np.array([bound[0] for bound in action_bounds], dtype=np.float32)
        high = np.array([bound[1] for bound in action_bounds], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.full(obs_dim, np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            shape=(obs_dim,), dtype=np.float32
        )

        rospy.loginfo("Init Sensors...")
        while self.sensor.get_image() is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("Init Complete")

    def _get_state(self):
        pos = self.sensor.get_position() #차량 현재 위치
        v = self.sensor.get_velocity() #차량 현재 속도
        idx = self.sensor.get_target_index() #차량 경로 인덱스

        # lookahead 만큼의 점 추출
        future_points = []
        for i in range(self.lookahead):
            target_idx = min(idx + i, len(self.path)-1)
            dx = self.path[target_idx,0] - pos[0]
            dy = self.path[target_idx,1] - pos[1]
            future_points.extend([dx, dy])

        # state = [현재 위치, 속도, future_points]
        state = np.array([pos[0], pos[1], v] + future_points, dtype=np.float32)
        return state
    
    def get_observation(self):
        return self._get_state()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.last_steering = 0.0

        # 키보드 명령으로 시뮬레이터 리셋
        try:
            if self._first_reset:
                press_key('i')
                press_key('q')
                self._first_reset = False
            else:
                press_key('q')
                press_key('i')
                press_key('q')

        except subprocess.CalledProcessError as e:
            rospy.logerr(f"Reset failed: {e}")

        # 위치/속도 초기화는 sensor가 관리
        pos = self.sensor.get_position()
        v = self.sensor.get_velocity()
        if pos is None:
            raise SensorTimeoutError("No valid position after reset")

        return self.get_observation(), {}

    def step(self, action):
        steering, throttle = action
        self.last_steering = float(steering)

        self.sensor.send_control(steering, throttle)

        time.sleep(0.1)  # 물리 엔진 반영 시간 대기
        
        # 보상 계산 (CTE 기반)
        pos = self.sensor.get_position()
        cte = Cal_CTE.calculate_cte(pos, self.path)
        reward = -abs(cte)  # 오차 작을수록 보상↑

        # 종료 조건: 오차가 임계값 이상이면 종료
        done = abs(cte) > 3.0

        return self.get_observation(), reward, done, False, {}
    
    def render(self):
        image = self.sensor.get_image()
        if image is not None:
            # 정규화된 이미지를 시각화용으로 변환
            display_image = (image * 255).astype(np.uint8)
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
            show = cv2.resize(display_image, (320, 240))
            
            # 추가 정보 표시
            velocity = self.sensor.get_velocity()
            cv2.putText(show, f"Velocity: {velocity*3.6:.1f} km/h", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(show, f"Steering: {self.last_steering:.2f}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Morai Camera", show)
            cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    def set_reward_fn(self, reward_fn):
        self._reward_fn = reward_fn

    def set_episode_over_fn(self, terminated_fn):
        self._terminated_fn = terminated_fn