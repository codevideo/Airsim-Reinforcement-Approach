import numpy as np
import torch
import gym
import argparse
import os
import utils
import math
import json
import time
import torchvision

from agent.TD3 import TD3, SimpleFly, RTD3

from enviroment import airsim_gym_env
from torch.utils.tensorboard import SummaryWriter

os.environ[ " CUDA_VISIBLE_DEVICES "] = " 0 "

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, eval_env, seed, eval_episodes=1):
	avg_reward = 0.
	episode_timesteps = 0
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			episode_timesteps +=1
			eval_env.ep_time_step = episode_timesteps

			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

if __name__ == "__main__":
	# Define parser
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                       # Policy name (TD3, SimpleFly)
	parser.add_argument("--env", choices=["Soccer_Field_Easy",
                        				  "Soccer_Field_Medium",
                        				  "ZhangJiaJie_Medium",
                        				  "Building99_Hard"],
										  default="Soccer_Field_Easy")  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              	 # Sets Gym, PyTorch and Numpy seeds
	# parser.add_argument("--start_timesteps", default=5e4, type=int) 	 # Time steps initial random policy is used
	parser.add_argument("--start_timesteps", default=1e2, type=int) 	 # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=50, type=int)        	 # How often (episode) we evaluate
	parser.add_argument("--max_timesteps", default=1e7, type=int)   	 # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.9, type=float)         # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=2048, type=int)      	 # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.999)                	 # Discount factor
	parser.add_argument("--tau", default=0.005)                     	 # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              	 # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                	 # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=3, type=int)       	 # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")       	     # Save model and optimizer parameters
	parser.add_argument("--save_img", action="store_true", default=False)
	parser.add_argument("--load_model", default="")                 	 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--load_RB", default="")                  	     # ReplayBuffer load file name, "" doesn't load.
	parser.add_argument("--load_IB", default="")
	parser.add_argument("--freeze_actor", default=3000, type=int)		 # Freeze actor (time step)
	parser.add_argument("--name", default="RLtrain")					 # Folder name
	parser.add_argument("--control_mode", default="new_rpyt", choices=["acc_rpyt", "vel_rpyt", "rpyt", "new_rpyt"])
	parser.add_argument("--validation", action="store_true")
	parser.add_argument("--rnn", action="store_true")
	args = parser.parse_args()

	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	# Set file path(result, models, replay buffer)
	weight_dir = "Weight"
	save_path = weight_dir+'/'+args.name
	save_path_serial = 0
	if os.path.exists(save_path):
		save_path_serial = 1
		while os.path.exists(save_path + '_%d' % save_path_serial):
			save_path_serial += 1
		save_path = save_path + '_%d' % save_path_serial

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	if not os.path.exists(save_path+"/results"):
		os.makedirs(save_path+"/results")
	if args.save_model and not os.path.exists(save_path+"/models"):
		os.makedirs(save_path+"/models")
	if not os.path.exists(save_path+"/ReplayBuffer"):
		os.makedirs(save_path+"/ReplayBuffer")
	if not os.path.exists(save_path+"/ImgBuffer"):
		os.makedirs(save_path+"/ImgBuffer")
	if not os.path.exists(save_path+"/runs"):
		os.makedirs(save_path+"/runs")

	# Save records
	with open(save_path+"/records.txt", "w") as f:
		args_dict = vars(args)
		json.dump({'args': args_dict}, sort_keys=True, indent=4, fp=f)

	# Set Enviroment
	env = airsim_gym_env.AirSimEnviroment()
	env.control_mode = args.control_mode
	print("Control Mode: "+args.control_mode)

	# Set seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	# Set state action dimantion
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	# Set policy parameter
	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"freeze_actor": args.freeze_actor
	}

	# Initialize policy
	if args.policy == "TD3":
		if args.rnn:
			print('use RTD3')
			# Target policy smoothing is scaled wrt the action scale
			kwargs["policy_noise"] = args.policy_noise * max_action
			kwargs["noise_clip"] = args.noise_clip * max_action
			kwargs["policy_freq"] = args.policy_freq
			policy = RTD3.TD3(**kwargs)

		else:
			print('use normal TD3')
			# Target policy smoothing is scaled wrt the action scale
			kwargs["policy_noise"] = args.policy_noise * max_action
			kwargs["noise_clip"] = args.noise_clip * max_action
			kwargs["policy_freq"] = args.policy_freq
			policy = TD3.TD3(**kwargs)

		

	elif args.policy == "SimpleFly":
		policy = SimpleFly.SimpleFly(**kwargs)
	if args.load_model != "":
		policy.load(args.load_model)

	# Initial ReplayBuffer
	if args.rnn :
		print('use recurrent version reply buffer')
		replay_buffer = utils.HistoricalReplayBuffer(state_dim, action_dim)
	else:
		print('use normal replay buffer')
		replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	if args.load_RB != "":
		replay_buffer.load(args.load_RB)

	# Initial ImgBuffer
	img_buffer = utils.ImgBuffer(state_dim, 120, 160)
	if args.load_IB != "":
		img_buffer.load(args.load_IB)

	# Set TensorBoard
	writer = SummaryWriter(save_path+"/runs/")

	# Start enviroment
	env.load_level(args.env)
	env.initialize_drone()
	env.get_ground_truth_gate_poses()
	env.start_image_callback_thread()
	env.start_odometry_callback_thread()
	env.start_pass_callback_thread()


	# Evaluate untrained policy
	evaluations = [eval_policy(policy, env, args.seed)]

	# Reset training information
	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	expl_discount_c = 0
	noise = 1
	fifty_ep_gate = np.zeros(50)


#************************* MAIN FUNCTION START ********************************#
	for t in range(int(args.max_timesteps)):
		episode_timesteps += 1
		env.ep_time_step = episode_timesteps

		# Select action randomly or according to policy
		if args.validation:
			action = (policy.select_action(np.array(state))).clip(-max_action, max_action)

		elif t < args.start_timesteps:
			if args.load_model != "" and args.load_RB == "":
				noise = (0.1+args.expl_noise*(math.exp(-0.00001*expl_discount_c)))
				action = (policy.select_action(np.array(state))
					     + np.random.normal(0, max_action * noise, size=action_dim)
						 ).clip(-max_action, max_action)
			else:
				action = (env.random_action()).clip(-max_action, max_action)
				# print('action',action)

		else:
			noise = (0.1+args.expl_noise*(math.exp(-0.00001*expl_discount_c)))
			action = (policy.select_action(np.array(state))
				     + np.random.normal(0, max_action * noise, size=action_dim)
					 ).clip(-max_action, max_action)
			expl_discount_c+=1
			# print('noise',noise)
			# print('action',action)

		# Perform action
		next_state, reward, done, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)
		# print('env.Img_rgb.shape',env.Img_rgb.shape)
		# print('env.Img_g.shape',env.Img_g.shape)

		# img_buffer.add(state, env.Img_g)
		img_buffer.add(state, env.Img_rgb)
		# print('env.Img_rgb.shape',env.Img_rgb.shape)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps and args.validation == False:
			if not args.rnn:
				policy.train(replay_buffer, args.batch_size)
			



		# TensorBoard record (each step)
		writer.add_scalar("Reward/StepReward", reward, t)
		writer.add_scalar("Error/PositionError", env.error, t)
		writer.add_scalar("Error/YawError", abs(env.yaw_error), t)
		writer.add_scalar("Action_Space/X", action[0], t)
		writer.add_scalar("Action_Space/Y", action[1], t)
		writer.add_scalar("Action_Space/Z", action[2], t)
		writer.add_scalar("Action_Space/Yaw", action[3], t)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.4f} n_Gate: { env.gate_counter}")
			fifty_ep_gate[int(episode_num % 50)] = env.gate_counter

			# TensorBoard record (each episode)
			if writer is not None:
				if t >= args.start_timesteps:
					writer.add_scalar("Loss/CriticLoss", policy.critic_loss, episode_num)
					writer.add_scalar("Loss/ActorLoss", policy.actor_loss, episode_num)
				writer.add_scalar("Performance/PassGateNumber", env.gate_counter, episode_num)
				writer.add_scalar("Reward/EpisodeAvgReward", episode_reward/episode_timesteps, episode_num)
				writer.add_scalar("Performance/Noise", noise, episode_num)
				writer.add_scalar("Performance/nGatePerEpisodeRate", fifty_ep_gate.sum()/50, episode_num)
            # Update RNN TD3
			if args.rnn:
				# if done:
				if t >= args.start_timesteps and args.validation == False:
					# print('start update policy')
					# with torch.autograd.set_detect_anomaly(True):
					policy.train(replay_buffer, args.batch_size)
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode & save result, models, replay buffer
		if (episode_num + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, env, args.seed))
			np.save(save_path+"/results/"+f"{file_name}", evaluations)
			replay_buffer.save(save_path+"/ReplayBuffer/"+f"{file_name}")
			# if args.save_img:img_buffer.save(save_path+"/ImgBuffer/"+str(time.time())+f"{file_name}")
			if args.save_img:img_buffer.save(save_path+"/ImgBuffer/"+f"{file_name}")
			if args.save_model: policy.save(save_path+"/models/"+f"{file_name}"+"_ep"+str(episode_num + 1))

#************************* MAIN FUNCTION STOP *********************************#
	# Close enviroment
	env.reset_race()
	env.stop_image_callback_thread()
	env.stop_odometry_callback_thread()
	env.stop_pass_callback_thread()
