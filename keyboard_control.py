import numpy as np
import cv2
import time
import argparse
from enviroment import airsim_gym_env

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="Soccer_Field_Easy")
parser.add_argument("--control_mode", default="acc_rpyt", choices=["acc_rpyt", "vel_rpyt", "rpyt", "new_rpyt"])
args = parser.parse_args()

cv2.namedWindow("image")
env = airsim_gym_env.AirSimEnviroment()
env.load_level(args.env)
env.initialize_drone()
env.get_ground_truth_gate_poses()
env.start_image_callback_thread()
env.start_odometry_callback_thread()
env.start_pass_callback_thread()
env.MAX_ep_step = 1000
env.control_mode = args.control_mode

kp = 0.45
kd = 0.05


for ep in range(20):
    ob = env.reset()
    throttle = 0
    old_time = time.time()
    time_step = 0
    while True:
        env.ep_time_step = time_step
        vx_t, vy_t, vz_t, w_yaw = 0,0,0,0
        key_code = cv2.waitKey(1)
        if key_code == ord('w'):
            vx_t = 1
        elif key_code == ord('s'):
            vx_t = -1
        elif key_code == ord('a'):
            vy_t = 1
        elif key_code == ord('d'):
            vy_t = -1
        elif key_code == ord('q'):
            w_yaw = 1
        elif key_code == ord('e'):
            w_yaw = -1
        elif key_code == ord('o'):
            vz_t -= 1
        elif key_code == ord('p'):
            vz_t += 1
        elif key_code == ord('t'):
            break

        action = np.array([vx_t, vy_t, vz_t, w_yaw])
        next_ob, reward, done, info = env.step(action)
        ob = next_ob
        time_step+=1

cv2.destroyWindow("image")
env.reset_race()
env.stop_image_callback_thread()
env.stop_odometry_callback_thread()
env.stop_pass_callback_thread()
