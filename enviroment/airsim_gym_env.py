from argparse import ArgumentParser
from . import utils
#import utils
import airsimneurips as airsim
import cv2
import threading
import time
import numpy as np
import math
import sys
import traceback
import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_high = 120
image_width = 160
image_size = image_high*image_width
h_dim_1 = 1000
h_dim_2 = 500
h_dim_3 = 250
z_dim = 20
num_epochs = 40
ep_max_step = 10000
batch_size = 32
learning_rate = 5e-4



# VAE model
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# Set folder
sample_dir = 'dronet_full_convolution_samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Got data & Data loader
# img_buffer = utils.ImgBuffer(8, 120, 160)
# img_buffer.load("/home/pc3396/newrtd3_test/NewRTD3/Weight/fullinfo_vel_maxv5_6/ImgBuffer/TD3_Soccer_Field_Easy_0_pretrained_fullinfo_maxV5.npz")

# Set Hyper parameters
# image_high = img_buffer.img_high
# image_width = img_buffer.img_width
# image_size = image_high*image_width*3
h_dim_1 = 1000
h_dim_2 = 500
h_dim_3 = 250
z_dim = 10
num_epochs = 400
ep_max_step = 10000
batch_size = 32
learning_rate = 5e-4


INPUT_CHANNELS = 3
CHANNEL_FACTOR = 0.25
DROPOUT_PROBABILITY = 0.5
DATA_BATCH_SIZE = 10

# VAE model
class VAE(nn.Module):
    def __init__(self, image_size, h_dim_1, h_dim_2, h_dim_3, z_dim, input_channels=INPUT_CHANNELS, channel_factor=CHANNEL_FACTOR, dropout_probability=DROPOUT_PROBABILITY, batch_size=DATA_BATCH_SIZE):
        super(VAE, self).__init__()
        '''
        self.fc1 = nn.Linear(image_size, h_dim_1)        # Input layer
        self.fc2_1 = nn.Linear(h_dim_1, h_dim_2)
        self.fc2_2 = nn.Linear(h_dim_2, h_dim_3)
        self.fc3 = nn.Linear(h_dim_3, z_dim)             # Hidden layer(mean)
        self.fc4 = nn.Linear(h_dim_3, z_dim)             # Hidden layer(var)
        '''

        # DroNet Architecture
        ## Encorder
        ### Layer 0
        self.layer_0_conv1 = nn.Conv2d(input_channels, int(32*channel_factor), 5, stride=2, padding=2)
        self.layer_0_maxpool1 = nn.MaxPool2d(3, stride=2)

        ### Layer 1
        self.layer_1_1_conv1 = nn.Conv2d(int(32*channel_factor), int(32*channel_factor), 3, stride=2, padding=1)
        self.layer_1_1_conv2 = nn.Conv2d(int(32*channel_factor), int(32*channel_factor), 3, padding=1)
        self.layer_1_2_conv1 = nn.Conv2d(int(32*channel_factor), int(32*channel_factor), 1, stride=2)

        ### Layer 2
        self.layer_2_1_conv1 = nn.Conv2d(int(32*channel_factor), int(64*channel_factor), 3, stride=2, padding=1)
        self.layer_2_1_conv2 = nn.Conv2d(int(64*channel_factor), int(64*channel_factor), 3, padding=1)
        self.layer_2_2_conv1 = nn.Conv2d(int(32*channel_factor), int(64*channel_factor), 1, stride=2)

        ### Layer 3
        self.layer_3_1_conv1 = nn.Conv2d(int(64*channel_factor), int(128*channel_factor), 3, stride=2, padding=1)
        self.layer_3_1_conv2 = nn.Conv2d(int(128*channel_factor), int(128*channel_factor), 3, padding=1)
        self.layer_3_2_conv1 = nn.Conv2d(int(64*channel_factor), int(128*channel_factor), 1, stride=2)

        ### Layer 4
        self.layer_4_dropout = nn.Dropout(dropout_probability)
        self.layer_4_linear = nn.Linear(4 * 5 * int(128*channel_factor), int(256*channel_factor))

        ### Layer 5 & 6
        self.layer_5_linear = nn.Linear(int(256*channel_factor), z_dim)
        self.layer_6_linear = nn.Linear(int(256*channel_factor), z_dim)
        
        # Decorder
        ## latent space
        '''
        self.fc5 = nn.Linear(z_dim, h_dim_3)
        self.fc6_1 = nn.Linear(h_dim_3, h_dim_2)
        self.fc6_2 = nn.Linear(h_dim_2, h_dim_1)
        self.fc7 = nn.Linear(h_dim_1, image_size)
        '''
        self.layer_7_deconv1 = nn.ConvTranspose2d(z_dim, 128, kernel_size=(3,4), stride=1)
        #self.layer_7_deconv1 = nn.ConvTranspose2d(z_dim, 128, kernel_size=4, stride=1)
        #self.layer_8_deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, dilation=3)
        #self.layer_8_deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=(5,6), stride=1, dilation=3)
        self.layer_8_deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=1, dilation=(3,4))
        self.layer_9_deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=6, stride=2, dilation=1, padding = 2)
        self.layer_10_deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, dilation=1, padding = 2, output_padding = 1)
        self.layer_11_deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1, dilation=1, padding = 2)
        self.layer_12_deconv1 = nn.ConvTranspose2d(16, input_channels, kernel_size=5, stride=2, dilation=1, padding = 2, output_padding = 1)

    # Encoder
    def encode(self, img):
        # h = F.relu(self.fc1(img))
        # h = F.relu(self.fc2_1(h))
        # h = F.relu(self.fc2_2(h))
        # mean = self.fc3(h)
        # var = self.fc4(h)

        # DroNet architecture
        # Layer 0
        x0 = self.layer_0_conv1(img)
        x0 = self.layer_0_maxpool1(x0)

        ##########
        # Layer 1
        ##########
        # Layer 1_1
        x11 = F.relu(x0)
        x11 = self.layer_1_1_conv1(x11)
        x11 = F.relu(x11)
        x11 = self.layer_1_1_conv2(x11)

        # Layer 1_2
        x12 = self.layer_1_2_conv1(x0)

        # Layer 1 Total
        x11.add(x12)
        x1 = x11

        ##########
        # Layer 2
        ##########
        # Layer 2_1
        x21 = F.relu(x1)
        x21 = self.layer_2_1_conv1(x21)
        x21 = F.relu(x21)
        x21 = self.layer_2_1_conv2(x21)

        # Layer 2_2
        x22 = self.layer_2_2_conv1(x1)

        # Layer 2 Total
        x21.add(x22)
        x2 = x21

        ##########
        # Layer 3
        ##########
        # Layer 3_1
        x31 = F.relu(x2)
        x31 = self.layer_3_1_conv1(x31)
        x31 = F.relu(x31)
        x31 = self.layer_3_1_conv2(x31)

        # Layer 2_2
        x32 = self.layer_3_2_conv1(x2)

        # Layer 2 Total
        x31.add(x32)
        x3 = x31

        ##########
        # Layer 4
        ##########
        x4 = torch.reshape(x3, (x3.shape[0], -1,))
        
        x4 = self.layer_4_dropout(x4)
        #print('x4.shape',x4.shape)
        x4 = self.layer_4_linear(x4)

        ##########
        # Layer 5
        ##########
        mean = self.layer_5_linear(F.relu(x4))
        var = self.layer_6_linear(F.relu(x4))

        # return x5

        return mean, var

    # mix m, var, noise
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)

        return mu + eps * std

    # Decoder
    def decode(self, z):
        # h = F.relu(self.fc5(z))
        # h = F.relu(self.fc6_1(h))
        # h = F.relu(self.fc6_2(h))
        # out_img = F.sigmoid(self.fc7(h))
        size = 10
        y0 = z.view(z.size(0), size, 1, 1)
        #print('y0.shape',y0.shape)
        y1 = self.layer_7_deconv1(y0)
        #y1 = nn.ReLU(self.layer_7_deconv1(y0))
        #y1 = nn.ReLU(y1)
        y1 = F.relu(y1)
        #print('y1.shape,after layer_7_deconv1:',y1.shape)
        y2 = self.layer_8_deconv1(y1)
        #y2 = nn.ReLU(y2)
        y2 = F.relu(y2)
        #print('y2.shape,after layer_8_deconv1:',y2.shape)

        #y2 = nn.ReLU(self.layer_8_deconv1(y1))
        #y3 = nn.ReLU(self.layer_9_deconv1(y2))
        y3 = self.layer_9_deconv1(y2)
        #y3 = nn.ReLU(y3)
        y3 = F.relu(y3)
        #print('y3.shape,after layer_9_deconv1:',y3.shape)

        #y4 = nn.ReLU(self.layer_10_deconv1(y3))
        y4 = self.layer_10_deconv1(y3)
        #y4 = nn.ReLU(y4)
        y4 = F.relu(y4)
        #print('y4.shape,after layer_10_deconv1:',y4.shape)

        #y5 = nn.ReLU(self.layer_11_deconv1(y4))
        y5 = self.layer_11_deconv1(y4)
        #y5 = nn.ReLU(y5)
        y5 = F.relu(y5)
        #print('y5.shape,after layer_11_deconv1:e',y5.shape)

        #out_img = nn.Tanh(self.layer_12_deconv1(y4))
        out_img = self.layer_12_deconv1(y5)
        #out_img = nn.Tanh(out_img)
        out_img = torch.tanh(out_img)
        #print('ou.shape,after layer_12_deconv1:',out_img.shape) # batch,3,120,160

        return out_img

    # forward
    def forward(self, img):
        m, var = self.encode(img)
        z = self.reparameterize(m, var)
        img_reconst = self.decode(z)

        return img_reconst, m, var

    def get_latent_z(self, img):
        m, var = self.encode(img)
        z = self.reparameterize(m, var)

        return z


# Define the model & Optimizer


# drone_name should match the name in ~/Document/AirSim/settings.json
class AirSimEnviroment(object):
    def __init__(self, drone_name = "drone_1", viz_traj=False, viz_traj_color_rgba=[1.0, 0.0, 0.0, 1.0], viz_image_cv2=False):
        # Information of drone
        self.oldtime = time.time()
        self.old_control_time = time.time()
        self.drone_name = drone_name
        self.drone_position = None
        self.drone_orientation = None
        self.drone_linear_velocity = None
        self.drone_angular_velocity = None
        self.drone_linear_acceleration = None
        self.drone_angular_acceleration = None
        self.drone_roll = None
        self.drone_pitch = None
        self.drone_yaw = None
        self.real_drone_roll = None
        self.real_drone_pitch = None
        self.real_drone_yaw = None
        self.x_pos = 0
        self.y_pos = 0
        self.z_pos = 0
        self.old_x_pos = 0
        self.old_y_pos = 0
        self.old_z_pos = 0
        self.x_vel = 0
        self.y_vel = 0
        self.z_vel = 0
        self.old_x_vel = 0
        self.old_y_vel = 0
        self.old_z_vel = 0
        self.x_acc = 0
        self.y_acc = 0
        self.z_acc = 0
        self.stop_motion_c = 0
        self.design_vel = 4

        # Information of gate
        self.gate_poses_ground_truth = None
        self.gate_num = None
        self.gate_counter = 0
        self.old_gate_counter = 0

        # Information of img_g
        self.Img_g = None
        self.Img_rgb = None

        # Information of error
        self.x_error = 0
        self.y_error = 0
        self.z_error = 0
        self.error = 0
        self.yaw_error = 0
        self.old_error = 0
        self.old_yaw_error = 0

        # Setting of airsim View
        self.viz_traj = viz_traj
        self.viz_traj_color_rgba = viz_traj_color_rgba
        self.viz_image_cv2 = viz_image_cv2

        # Setting of Gym Like Parameter
        # self.observation_space = gym.spaces.Box(-30, 30, (8,))
        print('we use latent as state')
        self.observation_space = gym.spaces.Box(-10, 10, (10,))
        self.action_space = gym.spaces.Box(-1, 1, (4,))
        self._max_episode_steps = 99999
        self.ep_time_step = 0
        self.done = False
        self.MAX_ep_step = 300
        self.max_acc = 150
        self.control_mode = "rpyt"
        self.reward_parameter = {"pass_Gate":150,
                                 "error_punish":-2.0,
                                 "yaw_error_punish":-0.5,
                                 "vel_punish":-0.01,
                                 "action_punish":-0.0,
                                 "step_punish":-100,
                                 "no_motion_punish":-100}
        self.encoder = model = VAE(image_size, h_dim_1, h_dim_2, h_dim_3, z_dim).to(device)
        print("load encorder :/home/pc3433/aidrone-airsim/Airsim-Dreamer/simple_vae/dronet_full_convolution_model_folder/epoch19.pth")
        self.encoder.load_state_dict(torch.load("/home/pc3433/aidrone-airsim/Airsim-Dreamer/simple_vae/dronet_full_convolution_model_folder/epoch19.pth"))
        self.encoder.eval()



        # we need three airsim MultirotorClient objects because the comm lib we use (rpclib) is not thread safe
        # Drone in a thread using one airsim MultirotorClient object
        # Images in a thread using one airsim MultirotorClient object
        # Odometry in a thread using one airsim MultirotorClient object (querying state commands)
        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()
        self.airsim_client_images = airsim.MultirotorClient()
        self.airsim_client_images.confirmConnection()
        self.airsim_client_odom = airsim.MultirotorClient()
        self.airsim_client_odom.confirmConnection()
        self.airsim_client_pass = airsim.MultirotorClient()
        self.airsim_client_pass.confirmConnection()

        self.level_name = None

        # Set the "image, odometry, pass" thread
        self.image_callback_thread = threading.Thread(target=self.repeat_timer_image_callback, args=(self.image_callback, 0.03))
        self.odometry_callback_thread = threading.Thread(target=self.repeat_timer_odometry_callback, args=(self.odometry_callback, 0.02))
        self.pass_callback_thread = threading.Thread(target=self.repeat_timer_pass_callback, args=(self.pass_callback, 0.03))
        self.is_image_thread_active = False
        self.is_odometry_thread_active = False
        self.is_pass_thread_active = False
        self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS = 10 # see https://github.com/microsoft/AirSim-NeurIPS2019-Drone-Racing/issues/38

    # Loads desired level
    def load_level(self, level_name, sleep_sec = 3.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        self.airsim_client.confirmConnection() # failsafe
        time.sleep(sleep_sec) # let the environment load completely

    # Starts an instance of a race in your given level, if valid
    def start_race(self, tier=1):
        self.airsim_client.simStartRace(tier)

    # arms drone, enable APIs, set default traj tracker gains
    def initialize_drone(self):
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)

        # set default values for trajectory tracker gains
        traj_tracker_gains = airsim.TrajectoryTrackerGains(kp_cross_track = 5.0, kd_cross_track = 0.0,
                                                            kp_vel_cross_track = 3.0, kd_vel_cross_track = 0.0,
                                                            kp_along_track = 0.4, kd_along_track = 0.0,
                                                            kp_vel_along_track = 0.04, kd_vel_along_track = 0.0,
                                                            kp_z_track = 2.0, kd_z_track = 0.0,
                                                            kp_vel_z = 0.4, kd_vel_z = 0.0,
                                                            kp_yaw = 3.0, kd_yaw = 0.1)

        self.airsim_client.setTrajectoryTrackerGains(traj_tracker_gains, vehicle_name=self.drone_name)
        time.sleep(0.2)

    # (Unused)like "takeoff_with_moveOnSpline"
    def takeoffAsync(self):
        self.airsim_client.takeoffAsync().join()

    # like takeoffAsync(), but with moveOnSpline()
    def takeoff_with_moveOnSpline(self, takeoff_height = 1.5):
        start_position = self.airsim_client.simGetVehiclePose(vehicle_name=self.drone_name).position
        takeoff_waypoint = airsim.Vector3r(start_position.x_val, start_position.y_val+0.5, start_position.z_val-takeoff_height)

        self.airsim_client.moveOnSplineAsync([takeoff_waypoint],
                                             vel_max=15.0,
                                             acc_max=5.0,
                                             add_position_constraint=True,
                                             add_velocity_constraint=False,
                                             add_acceleration_constraint=False,
                                             viz_traj=self.viz_traj,
                                             viz_traj_color_rgba=self.viz_traj_color_rgba,
                                             vehicle_name=self.drone_name).join()
        self.done = False

    # stores gate ground truth poses as a list of airsim.Pose() objects in "self.gate_poses_ground_truth"
    def get_ground_truth_gate_poses(self):
        gate_original_list = self.airsim_client.simListSceneObjects("Gate.*")
        gate_names_sorted_bad = sorted(gate_original_list)
        gate_indices_bad = [int(gate_name.split('_')[0][4:]) for gate_name in gate_names_sorted_bad]
        gate_indices_correct = sorted(range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k])
        gate_names_sorted = [gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct]
        self.gate_poses_ground_truth = []
        self.gate_num = len(gate_names_sorted)
        for gate_name in gate_names_sorted:
            curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            counter = 0
            # Make sure to get the coordinates
            while (math.isnan(curr_pose.position.x_val) or math.isnan(curr_pose.position.y_val) or math.isnan(curr_pose.position.z_val)) and (counter < self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS):
                print(f"DEBUG: {gate_name} position is nan, retrying...")
                counter += 1
                curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            assert not math.isnan(curr_pose.position.x_val), f"ERROR: {gate_name} curr_pose.position.x_val is still {curr_pose.position.x_val} after {counter} trials"
            assert not math.isnan(curr_pose.position.y_val), f"ERROR: {gate_name} curr_pose.position.y_val is still {curr_pose.position.y_val} after {counter} trials"
            assert not math.isnan(curr_pose.position.z_val), f"ERROR: {gate_name} curr_pose.position.z_val is still {curr_pose.position.z_val} after {counter} trials"
            self.gate_poses_ground_truth.append(curr_pose)

    # Resets a current race: moves players to start positions, timer and penalties reset
    def reset_race(self):
        self.airsim_client.simResetRace()

    #**************************  Information Callback *************************#
    # call task() method every "period" seconds.
    def repeat_timer_image_callback(self, task, period):
        # task() is image_callback()
        while self.is_image_thread_active:
            task()
            time.sleep(period)

    # call task() method every "period" seconds.
    def repeat_timer_odometry_callback(self, task, period):
        # task() is odometry_callback()
        while self.is_odometry_thread_active:
            task()
            time.sleep(period)

    # call task() method every "period" seconds.
    def repeat_timer_pass_callback(self, task, period):
        # task() is pass_callback()
        while self.is_pass_thread_active:
            task()
            time.sleep(period)

    # Get image
    def image_callback(self):
        # get uncompressed fpv cam image
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_images.simGetImages(request)
        img_rgb_1d = np.fromstring(response[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        if self.viz_image_cv2:
            cv2.imshow("img_rgb", img_rgb)
            cv2.waitKey(1)
        img_g = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        self.Img_g = cv2.resize(img_g,(160,120),interpolation=cv2.INTER_CUBIC)
        self.Img_rgb = cv2.resize(img_rgb,(160,120),interpolation=cv2.INTER_CUBIC)


    # Get odometry
    def odometry_callback(self):
        drone_state = self.airsim_client_odom.getMultirotorState()
        # in world frame:
        self.drone_position = drone_state.kinematics_estimated.position
        self.drone_orientation = drone_state.kinematics_estimated.orientation
        self.drone_linear_velocity = drone_state.kinematics_estimated.linear_velocity
        self.drone_angular_velocity = drone_state.kinematics_estimated.angular_velocity
        self.drone_linear_acceleration = drone_state.kinematics_estimated.linear_acceleration
        self.drone_angular_acceleration = drone_state.kinematics_estimated.angular_velocity
        if self.drone_orientation != None:
            self.drone_roll, self.drone_pitch, self.drone_yaw =  utils.Quaternion2Euler(self.drone_orientation)
            self.real_drone_roll = self.drone_roll*(180/math.pi)
            self.real_drone_pitch = -self.drone_pitch*(180/math.pi)
            self.real_drone_yaw = -self.drone_yaw*(180/math.pi)

    # Get pass information
    def pass_callback(self):
        drone_collition_info = self.airsim_client_pass.simGetCollisionInfo(vehicle_name = "drone_1")
        if self.drone_orientation != None and self.drone_position != None and self.gate_poses_ground_truth[self.gate_counter] != None:
            goal_gate_info = self.gate_poses_ground_truth[self.gate_counter]
            self.x_error, self.y_error, self.z_error, self.yaw_error = self._cal_drone_error(goal_gate_info)
            self.error = (self.x_error**2+self.y_error**2+self.z_error**2)**0.5
            if self.error <= 1 and self.gate_counter < len(self.gate_poses_ground_truth)-1 :
                self.gate_counter+=1
            self._cal_VelAcc()
            acceleration = (self.x_acc**2+self.y_acc**2+self.z_acc**2)**0.5
            drone_vel = (self.x_vel**2 + self.y_vel**2 + self.z_vel**2)**0.5

            if drone_vel < 0.5:
                self.stop_motion_c+=1
            else:
                self.stop_motion_c = 0

            if drone_collition_info.has_collided:
                self.done = True
                #print("Done reason: Drone Crash")
            elif self.error > 60:
                self.done = True
                #print("Done reason: Error too big")
            elif self.x_error <= 0:
                self.done = True
                #print("Didn't pass gate")
            elif self.gate_counter >= 10:
                self.done = True
                #print("Done reason: Finish 10 Gate")
            elif self.ep_time_step > self.MAX_ep_step:
                self.done = True
                #print("Done reason: Out of step")
            elif self.stop_motion_c > 150:
                self.done = True
                #print("Done reason: No motion")

            '''
            fmt = '{:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5} {:<5}'
            print(fmt.format('X_e:', str(round(self.x_error,2)),
                             'Y_e:', str(round(self.y_error,2)),
                             'Z_e:', str(round(self.z_error,2)),
                             'YW_e:', str(round(self.yaw_error, 2)),
                             'D_roll:', str(round(self.real_drone_roll, 2)),
                             'D_pitch:', str(round(self.real_drone_pitch, 2)),
                             'x_vel:', str(round(self.x_vel, 2)),
                             'y_vel:', str(round(self.y_vel, 2)),
                             'z_vel:', str(round(self.z_vel, 2)),
                             'Passed:', str(self.gate_counter)),end="\r")
            '''

    def start_image_callback_thread(self):
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()
            print("Started image callback thread")

    def stop_image_callback_thread(self):
        if self.is_image_thread_active:
            self.is_image_thread_active = False
            self.image_callback_thread.join()
            print("Stopped image callback thread.")

    def start_odometry_callback_thread(self):
        if not self.is_odometry_thread_active:
            self.is_odometry_thread_active = True
            self.odometry_callback_thread.start()
            print("Started odometry callback thread")

    def stop_odometry_callback_thread(self):
        if self.is_odometry_thread_active:
            self.is_odometry_thread_active = False
            self.odometry_callback_thread.join()
            print("Stopped odometry callback thread.")

    def start_pass_callback_thread(self):
        if not self.is_pass_thread_active:
            self.is_pass_thread_active = True
            self.pass_callback_thread.start()
            print("Started pass callback thread")

    def stop_pass_callback_thread(self):
        if self.is_pass_thread_active:
            self.is_pass_thread_active = False
            self.pass_callback_thread.join()
            print("Stopped pass callback thread.")

    def _cal_drone_error(self, goal_gate_info):
        x = goal_gate_info.position.x_val-self.drone_position.x_val
        y = goal_gate_info.position.y_val-self.drone_position.y_val
        z = goal_gate_info.position.z_val-self.drone_position.z_val
        _, _, gate_yaw = utils.Quaternion2Euler(goal_gate_info.orientation)
        x_error, y_error, z_error = utils.World2BodyFrame(self.drone_yaw, x, y, z)
        yaw_error = gate_yaw + 0.5*math.pi - self.drone_yaw
        if yaw_error  > math.pi:
            yaw_error  -= 2*math.pi
        elif yaw_error  < -math.pi:
            yaw_error  += 2*math.pi
        else:
            pass
        yaw_error = -yaw_error*(180/math.pi)
        #yaw_error = -yaw_error

        return x_error, y_error, z_error, yaw_error

    def _cal_VelAcc(self):
        dt = (time.time()-self.oldtime)
        self.x_pos = self.drone_position.x_val
        self.y_pos = self.drone_position.y_val
        self.z_pos = self.drone_position.z_val
        self.x_vel, self.y_vel, self.z_vel = utils.World2BodyFrame(self.drone_yaw,
                                                                   (self.x_pos-self.old_x_pos)/dt,
                                                                   (self.y_pos-self.old_y_pos)/dt,
                                                                   (self.z_pos-self.old_z_pos)/dt)
        self.x_acc = (self.x_vel-self.old_x_vel)/dt
        self.y_acc = (self.y_vel-self.old_y_vel)/dt
        self.z_acc = (self.z_vel-self.old_z_vel)/dt
        self.old_x_pos, self.old_y_pos, self.old_z_pos = self.x_pos, self.y_pos, self.z_pos
        self.old_x_vel, self.old_y_vel, self.old_z_vel = self.x_vel, self.y_vel, self.z_vel
        self.oldtime = time.time()

    #************************* Control Drone **********************************#
    # Hight level Control
    def fly_through_all_gates_at_once_with_moveOnSpline(self):
        if self.level_name in ["Soccer_Field_Medium", "Soccer_Field_Easy", "ZhangJiaJie_Medium"] :
            vel_max = 30.0
            acc_max = 15.0

        if self.level_name == "Building99_Hard":
            vel_max = 4.0
            acc_max = 1.0
        return self.airsim_client.moveOnSplineAsync([gate_pose.position for gate_pose in self.gate_poses_ground_truth],
                                                    vel_max=vel_max,
                                                    acc_max=acc_max,
                                                    add_position_constraint=True,
                                                    add_velocity_constraint=False,
                                                    add_acceleration_constraint=False,
                                                    viz_traj=self.viz_traj,
                                                    viz_traj_color_rgba=self.viz_traj_color_rgba,
                                                    vehicle_name=self.drone_name).join()

    def fly_through_all_gates_at_once_with_RPYT_control(self):
        self.airsim_client.moveByAngleRatesThrottleAsync(roll_rate=0.0,
                                                         pitch_rate=0.1,
                                                         yaw_rate=0.0,
                                                         throttle=0.6,
                                                         duration=0.01,
                                                         vehicle_name=self.drone_name).join()

    #************************* Gym Like Function ******************************#
    def reset(self):
        self.airsim_client.reset()
        self.airsim_client.enableApiControl(self.drone_name)
        self.airsim_client.arm(self.drone_name)

        self.airsim_client.simResetRace()
        time.sleep(1.0)
        self.start_race(2)
        self.takeoff_with_moveOnSpline()
        self.stop_motion_c = 0
        self.done = False

        self.x_error = 0
        self.y_error = 0
        self.z_error = 0
        self.error = 0
        self.yaw_error = 0
        self.old_error = 0
        self.old_yaw_error = 0

        self.gate_counter = 0
        self.old_gate_counter = 0
        '''
        state = np.array([self.x_error,
                          self.y_error,
                          self.z_error,
                          self.yaw_error,
                          self.x_vel,
                          self.y_vel,
                          self.z_vel,
                          -1*self.drone_angular_velocity.z_val])
        print("state.shape old",state.shape)
        print("self.Img_rgb.shape",self.Img_rgb.shape)
        '''
        img_tensor = torch.FloatTensor(self.Img_rgb).to(device) 
        x_img = torch.unsqueeze(img_tensor/255,dim=0).permute(-4,-1,-3,-2)
        #print("x_img.shape",x_img.shape)
        #stop()
        state = self.encoder.get_latent_z(x_img)
        state = torch.squeeze(state).cpu().detach().numpy()
        #print('state.shape',state.shape)
        
        #a = self.Img_rgb.reshape(-1, image_size)/255
        # img_tensor = torch.FloatTensor(a).to(device)  
        # state, _ = self.encoder.encode(img_tensor)
        # state = state.cpu().detach().numpy()
        return state

    def random_action(self):
        action = np.random.normal(0, 0.5, size=4)+np.array([0.0, 0, 0.05, 0])
        return action

    def step(self, action):
        action = np.array(action).reshape(self.action_space.shape[0],-1)
        if self.control_mode == "rpyt":
            roll_rate = -float(action[1,])*0.2
            pitch_rate = float(action[0,])*0.2
            yaw_rate = float(action[3,])*0.5
            throttle = float(action[2,])*0.5

        elif self.control_mode == "vel_rpyt":
            kp = 0.2
            kd = 0.0075
            # vx_t, vy_t, vz_t, w_yaw = float(action[0,])*10, float(action[1,])*10, float(action[2,])*15, float(action[3,])*5
            # we set max vel here, we set max vel = 5
            vx_t, vy_t, vz_t, w_yaw = float(action[0,])*5, float(action[1,])*5, float(action[2,])*5, float(action[3,])*5
            vx_r, vy_r, vz_r = self.x_vel, self.y_vel, self.z_vel
            eVx, eVy, eVz = vx_t-vx_r, vy_t-vy_r, vz_t-vz_r
            eVyaw = w_yaw+self.drone_angular_velocity.z_val
            dt = time.time()-self.old_control_time
            self.old_control_time = time.time()

            roll_rate  = -1*(kp*eVy + kd*(eVy/dt))-(kp*self.real_drone_roll+kd*(self.real_drone_roll/dt))*0.15
            pitch_rate = kp*eVx + kd*(eVx/dt)-(kp*self.real_drone_pitch+kd*(self.real_drone_pitch/dt))*0.15
            yaw_rate = kp*eVyaw + kd*(eVyaw/dt)
            throttle = kp*eVz + kd*(eVz/dt)

        elif self.control_mode =="acc_rpyt":
            kp = 0.2
            kd = 0.0075
            ax_t, ay_t, az_t, a_yaw = float(action[0,])*1.5, float(action[1,])*1.5, float(action[2,])*10, float(action[3,])*5
            ax_r, ay_r, az_r = self.drone_linear_acceleration.x_val, -self.drone_linear_acceleration.y_val, -self.drone_linear_acceleration.z_val
            eAx, eAy, eAz = ax_t-ax_r, ay_t-ay_r, az_t-az_r
            eAyaw = a_yaw+self.drone_angular_acceleration.z_val
            dt = time.time()-self.old_control_time
            self.old_control_time = time.time()

            roll_rate  = -1*(kp*eAy + kd*(eAy/dt))-(kp*self.real_drone_roll+kd*(self.real_drone_roll/dt))*0.1
            pitch_rate = kp*eAx + kd*(eAx/dt)-(kp*self.real_drone_pitch+kd*(self.real_drone_pitch/dt))*0.1
            yaw_rate = kp*eAyaw + kd*(eAyaw/dt)
            throttle = kp*eAz + kd*(eAz/dt)

        elif self.control_mode =="new_rpyt":
            kp = 0.05
            kd = 0.005
            p_t, r_t = float(action[0,])*12, -float(action[1,])*12
            p_r, r_r = self.real_drone_pitch, self.real_drone_roll
            eP, eR = p_t-p_r, r_t-r_r
            dt = time.time()-self.old_control_time
            self.old_control_time = time.time()

            roll_rate = kp*eR + kd*(eR/dt)
            pitch_rate = kp*eP + kd*(eP/dt)
            yaw_rate = float(action[3,])
            throttle = float(action[2,])*0.5

        self.airsim_client.moveByAngleRatesThrottleAsync(roll_rate = roll_rate,
                                                         pitch_rate = pitch_rate,
                                                         yaw_rate = yaw_rate,
                                                         throttle = 0.58+throttle,
                                                         duration = 0.1,
                                                         vehicle_name = self.drone_name).join()
        '''
        Next_state = np.array([self.x_error,
                               self.y_error,
                               self.z_error,
                               self.yaw_error,
                               self.x_vel,
                               self.y_vel,
                               self.z_vel,
                               -1*self.drone_angular_velocity.z_val])
        '''
        Next_img_tensor = torch.FloatTensor(self.Img_rgb).to(device) 
        Next_x_img = torch.unsqueeze(Next_img_tensor/255,dim=0).permute(-4,-1,-3,-2)
        #print("x_img.shape",x_img.shape)
        #stop()
        Next_state = self.encoder.get_latent_z(Next_x_img)
        Next_state = torch.squeeze(Next_state).cpu().detach().numpy()
        #print("self.Img_g",self.Img_g)
        # b = self.Img_g.reshape(-1, image_size)/255
        # img_tensor = torch.FloatTensor(b).to(device)  
        # Next_state, _ = self.encoder.encode(img_tensor)
        # Next_state = Next_state.cpu().detach().numpy()
    

        Reward = self._get_reward(action)
        Done = self.done
        Info = None
        return Next_state, Reward, Done, Info

    def _get_reward(self, action):
        if self.gate_counter == self.old_gate_counter+1:
            Reward_pass_gate = self.reward_parameter["pass_Gate"]*1
            Reward_error = 0
            Reward_yaw_error = 0
        else:
            Reward_pass_gate = 0
            if self.ep_time_step > 1:
                Reward_error = self.reward_parameter["error_punish"]*(abs(self.error) - self.old_error)
                Reward_yaw_error = self.reward_parameter["yaw_error_punish"]*(abs(self.yaw_error) - self.old_yaw_error)
                if Reward_error <  0 :
                    pass
                else:
                    Reward_error = 0
                if Reward_yaw_error < 0:
                    pass
                else:
                    Reward_yaw_error = 0
            else:
                Reward_error = 0
                Reward_yaw_error = 0

        Reward_action = self.reward_parameter["action_punish"]*float((action**2).sum())
        Reward_vel = self.reward_parameter["vel_punish"]*(abs(self.x_vel-self.design_vel))

        if self.stop_motion_c > self.MAX_ep_step:
            Reward_no_motion = self.reward_parameter["no_motion_punish"]
        else:
            Reward_no_motion = 0
        if self.done:
            Reward_step = self.reward_parameter["step_punish"]*(math.e**(-0.05*self.ep_time_step))
        else:
            Reward_step = 0

        self.old_error = abs(self.error)
        self.old_yaw_error = abs(self.yaw_error)
        self.old_gate_counter = self.gate_counter
        #print(self.ep_time_step)
        #print(Reward_pass_gate , Reward_error , Reward_yaw_error , Reward_action , Reward_step , Reward_no_motion)
        #print(Reward_pass_gate + Reward_error + Reward_yaw_error + Reward_action + Reward_step + Reward_vel + Reward_no_motion)
        return (Reward_pass_gate + Reward_error + Reward_yaw_error + Reward_action + Reward_step + Reward_vel + Reward_no_motion)

def main(args):
    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    baseline_racer = AirSimEnviroment(drone_name="drone_1", viz_traj=args.viz_traj, viz_traj_color_rgba=[1.0, 1.0, 0.0, 1.0], viz_image_cv2=args.viz_image_cv2)
    baseline_racer.load_level(args.level_name)
    baseline_racer.initialize_drone()
    baseline_racer.get_ground_truth_gate_poses()
    baseline_racer.start_image_callback_thread()
    baseline_racer.start_odometry_callback_thread()
    baseline_racer.start_pass_callback_thread()

    if args.planning_and_control_api == "moveOnSpline":
        baseline_racer.start_race(args.race_tier)
        baseline_racer.takeoff_with_moveOnSpline()
        baseline_racer.fly_through_all_gates_at_once_with_moveOnSpline()  # Currently using this
        baseline_racer.reset_race()
    if args.planning_and_control_api == "RPYT_control":
        for e in range (3):
            baseline_racer.start_race(args.race_tier)
            baseline_racer.takeoff_with_moveOnSpline()
            for i in range(300):
                baseline_racer.fly_through_all_gates_at_once_with_RPYT_control()
                i+=1
            e+=1
            baseline_racer.reset_race()
            time.sleep(3)

    # Comment out the following if you observe the python script exiting prematurely, and resetting the race
    baseline_racer.stop_image_callback_thread()
    baseline_racer.stop_odometry_callback_thread()
    baseline_racer.stop_pass_callback_thread()
    time.sleep(3)

if __name__ == "__main__":
    parser = ArgumentParser()
    # Environment Setting
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy",
                                                           "Soccer_Field_Medium",
                                                           "ZhangJiaJie_Medium",
                                                           "Building99_Hard"],
                                                           default="ZhangJiaJie_Medium")
    parser.add_argument('--planning_and_control_api', type=str, choices=["moveOnSpline", "RPYT_control"], default="moveOnSpline")  # How to Control UAV (Which we need to design)
    parser.add_argument('--enable_viz_traj', dest='viz_traj', action='store_true', default=False)  # Show the baseline trajectory
    parser.add_argument('--enable_viz_image_cv2', dest='viz_image_cv2', action='store_true', default=False)  # Show the camera view in front of UAV
    parser.add_argument('--race_tier', type=int, choices=[1,2,3], default=1)  # racing setting
    args = parser.parse_args()
    main(args)

'''
# NOTE:
State:
self.x_error
self.y_error
self.z_error
self.yaw_error
self.real_drone_roll
self.real_drone_pitch
self.x_vel
self.y_vel
self.z_vel
'''
