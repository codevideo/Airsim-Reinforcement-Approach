import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


# from .model import Actor, Critic, lstmActor, lstmCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class lstmActor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action=None, hidden_dim=256, lstm_layer=1, GRU=False, **kwargs):
		# state_dim = observation_dim + action_dim
		super(lstmActor, self).__init__()
		self.state_dim = state_dim + action_dim
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim
		dropout = 0 if lstm_layer == 1 else 0.2

		rnn_type = nn.LSTM if not GRU else nn.GRU

		self.l1 = rnn_type(input_size=self.state_dim,  
						  hidden_size=hidden_dim, 
						  num_layers=lstm_layer,
						  bias=True, batch_first=True, dropout=dropout)

		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		# self.l3_throttle = nn.Linear(hidden_dim, 1)
		# self.l3_else = nn.Linear(hidden_dim, action_dim-1)
		self.l3 = nn.Linear(hidden_dim, action_dim)


		self.max_action = max_action or torch.ones(action_dim).to("cuda")



	def forward(self, *args, **kwargs):
		a, hc = self._forward(*args, **kwargs)
		return a

	def _forward(self, x, hc=None):
		
		# lstm_out, hc = self.l1(x, hc)
		lstm_out, hc = self.l1(x, hc)

		if type(x) is torch.Tensor:
			lstm_out = lstm_out.view((-1, self.hidden_dim))
		else:
			lstm_out_tmp, out_len = rnn_utils.pad_packed_sequence(lstm_out, batch_first=True)
			bs = lstm_out_tmp.shape[0]
			lstm_out = torch.zeros((bs, self.hidden_dim)).cuda()
			for i, length in enumerate(out_len):
				lstm_out[i] = lstm_out_tmp[i, length-1, :].clone()

		# lstm_out = F.relu(lstm_out)
		lstm_out = F.relu(lstm_out.clone())
		a = F.relu(self.l2(lstm_out))
		# output = torch.cat( [F.softsign(self.l3_else(a)), torch.sigmoid(self.l3_throttle(a))], 1)
		# output = torch.tanh(self.l3(a))
		# print('original out_put',output)

		# output = self.max_action * output
		# print('max_action',self.max_action)
		# print('output:',output)
		output = self.max_action * torch.tanh(self.l3(a))
		return output, hc

	def flatten_parameters(self):
		self.l1.flatten_parameters()


class lstmCritic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim=256, lstm_layer=1, dropout=0., GRU=False, **kwargs):
		super(lstmCritic, self).__init__()
		self.state_dim = state_dim + action_dim
		self.action_dim = action_dim
		self.hidden_dim = hidden_dim
		
		
		rnn_type = nn.LSTM if not GRU else nn.GRU
		self.q1l1 = rnn_type(input_size=self.state_dim,  
						  hidden_size=hidden_dim, 
						  num_layers=lstm_layer,
						  bias=True, batch_first=True, dropout=dropout)
		self.q1ax = nn.Linear(action_dim, hidden_dim)
		self.q1l2 = nn.Linear(hidden_dim, hidden_dim)
		self.q1l3 = nn.Linear(hidden_dim, 1)


		self.q2l1 = rnn_type(input_size=self.state_dim,  
						  hidden_size=hidden_dim, 
						  num_layers=lstm_layer,
						  bias=True, batch_first=True, dropout=dropout)
		self.q2ax = nn.Linear(action_dim, hidden_dim)
		self.q2l2 = nn.Linear(hidden_dim, hidden_dim)
		self.q2l3 = nn.Linear(hidden_dim, 1)
		
	def forward(self, *args, **kwargs):
		q1, q2, hc1, hc2 = self._forward(*args, **kwargs)
		return q1, q2

	def _forward(self, x, a, hc1=None, hc2=None):
		q1, hc1 = self.Q1(x, a, hc1)
		q2, hc2 = self.Q2(x, a, hc2)
		return q1, q2, hc1, hc2

	def flatten_parameters(self):
		self.q1l1.flatten_parameters()
		self.q2l1.flatten_parameters()

	def Q1(self, x, a, hc=None):
		# print('hc.shape',hc.shape)
		q1, hc = self.q1l1(x, hc)
		if type(x) is torch.Tensor:
			q1 = q1.view((-1, self.hidden_dim))
		else:
			lstm_out_tmp, out_len = rnn_utils.pad_packed_sequence(q1, batch_first=True)
			bs = lstm_out_tmp.shape[0]
			q1 = torch.zeros((bs, self.hidden_dim)).cuda()
			for i, length in enumerate(out_len):
				q1[i] = lstm_out_tmp[i, length-1, :].clone()

		q1a = self.q1ax(a)
		# q1 = F.relu(q1 + q1a)
		q1 = F.relu(q1.clone() + q1a)
		# q1 = F.relu(self.q1l2(q1))
		q1_f = q1.clone()
		q1 = F.relu(self.q1l2(q1_f))
		# q1 = self.q1l3(q1)
		q1_ff = q1.clone()
		q1 = self.q1l3(q1_ff)

		return q1, hc

	def Q2(self, x, a, hc=None):
		q2, hc = self.q2l1(x, hc)
		if type(x) is torch.Tensor:
			q2 = q2.view((-1, self.hidden_dim))
		else:
			lstm_out_tmp, out_len = rnn_utils.pad_packed_sequence(q2, batch_first=True)
			bs = lstm_out_tmp.shape[0]
			q2 = torch.zeros((bs, self.hidden_dim)).cuda()
			for i, length in enumerate(out_len):
				q2[i] = lstm_out_tmp[i, length-1, :].clone()

		q2a = self.q2ax(a)
		# q2 = F.relu(q2) + F.relu(q2a)
		q2 = F.relu(q2.clone()) + F.relu(q2a)
		# q2 = F.relu(self.q2l2(q2))
		q2 = F.relu(self.q2l2(q2.clone()))
		# q2 = self.q2l3(q2)
		q2 = self.q2l3(q2.clone())

		return q2, hc




class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		hidden_dim=256,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		rnn = True,
		lr_a=1e-3,
		lr_c=1e-3,
		J_lambda=1,
		weight_decay=0,
		freeze_actor=False 
	):

		self.rnn = rnn

		actor_model = Actor if not self.rnn else lstmActor 
		critic_model = Critic if not self.rnn else lstmCritic

		self.actor = actor_model(state_dim, action_dim, max_action, hidden_dim=hidden_dim, GRU=self.rnn in ['gru', 'GRU', 'Gru']).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_a, weight_decay=weight_decay)

		self.critic = critic_model(state_dim, action_dim, hidden_dim=hidden_dim, GRU=self.rnn in ['gru', 'GRU', 'Gru']).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c, weight_decay=weight_decay)

		self.action_dim = action_dim
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.J_lambda = J_lambda

		self.total_it = 0
		self.reset()
		self.actor_loss = 0
		self.critic_loss = 0


	def reset(self):
		if self.rnn:
			self.hc = None
			self.last_action = torch.zeros(1, self.action_dim).to(device)

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		if not self.rnn:
			return self.actor(state).cpu().data.numpy().flatten()
		else:
			state = torch.cat([state, self.last_action], dim=1).reshape(1, 1, -1)
			action, self.hc = self.actor._forward(state, self.hc)
			self.last_action = action.clone()
			return action.cpu().data.numpy().flatten()


	def get_loss(self, replay_buffer, batch_size=100, replay_expert=False):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		batch_size = reward.shape[0]

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			if not self.rnn:
				next_action = (
					self.actor_target(next_state) + noise
				).clamp(-self.max_action, self.max_action)
			else:
				self.actor_target.flatten_parameters()
				next_action, hc = self.actor_target._forward(state)
				self.actor_target.flatten_parameters()
				next_action = (
					self.actor_target(next_state.reshape(batch_size, 1, -1), hc) + noise
				).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			if not self.rnn:
				target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			else:
				self.critic_target.flatten_parameters()
				target_Q1, target_Q2, hc1, hc2 = self.critic_target._forward(state, action)  # actions doesnot matter here.
				target_Q1, target_Q2 = self.critic_target(next_state.reshape(batch_size, 1, -1), next_action, hc1, hc2)

			target_Q = torch.min(target_Q1, target_Q2)
			# target_Q = reward + not_done * self.discount * target_Q
			target_Q = reward + not_done * self.discount * target_Q.clone()

		# Get current Q estimates
		if self.rnn:
			self.critic.flatten_parameters()
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		if not replay_expert:
			
			# Compute actor losses
			if not self.rnn:
				actor_loss = - self.critic.Q1(state, self.actor(state)).mean() * self.J_lambda
			else:
				self.critic.flatten_parameters()
				actor_loss = - self.critic.Q1(state, self.actor(state))[0].mean() * self.J_lambda
				# print('actor_loss.shape',self.critic.Q1(state, self.actor(state))[0].shape)
				# actor_loss_ = -self.critic.Q1(state, self.actor(state))#.mean()
				# print('actor_loss_:')
				# print(self.critic.Q1(state, self.actor(state)))



			return critic_loss, actor_loss

		else:
			# Compute Behavior Cloning Loss if needed
		
			state, action, next_state, reward, not_done = replay_buffer.sample(self.buffer_size_expert)
			batch_size = reward.shape[0]
			# action = torch.clamp(action, -1, 1)
			action = torch.clamp(action.clone(), -1, 1)
			if self.rnn: self.actor_target.flatten_parameters()
			pred_action = self.actor_target(state)

			# Q Filter 
			if self.rnn: self.critic_target.flatten_parameters()
			Q_expert = torch.min(*self.critic_target(state, action))
			if self.rnn: self.critic_target.flatten_parameters()
			Q_pred = torch.min(*self.critic_target(state, pred_action))
			mask = (Q_expert<Q_pred).clone()
			Q_expert[mask] = 0
			Q_pred[mask] = 0

			bc_loss = F.mse_loss(Q_expert, Q_pred) * self.bc_lambda

			return critic_loss, bc_loss, torch.sum(mask==1).item()/self.buffer_size_expert

	def train(self, replay_buffer, batch_size=100, freeze_actor_update=False):
		# with torch.autograd.set_detect_anomaly(True):

		self.total_it += 1

		if hasattr(self, 'demonstration_buffer'):
			critic_loss_, actor_loss_ = self.get_loss(replay_buffer, batch_size)	
			exp_critic_loss, bc_loss, Qfilted_ratio = self.get_loss(self.demonstration_buffer, self.buffer_size_expert, replay_expert=True)
			loss_dict = OrderedDict({
							'actorLoss': actor_loss_.item(), 
							'bcLoss': bc_loss.item(), 
							'actorLoss_total': actor_loss_.item() + bc_loss.item(),
							'actorQfiltedRatio': Qfilted_ratio,
							'criticLoss': critic_loss_.item(),
							'criticLossExpert': exp_critic_loss.item(),
							'criticLoss_total': critic_loss_.item() + exp_critic_loss.item()
						})
			actor_loss = actor_loss_ + bc_loss
			critic_loss = critic_loss_ + exp_critic_loss
		else:
			critic_loss, actor_loss = self.get_loss(replay_buffer, batch_size)
			loss_dict = OrderedDict({'actorLoss': actor_loss.item() ,'criticLoss': critic_loss.item()})
		self.actor_loss = actor_loss
		self.critic_loss = critic_loss
		# # Optimize the critic
		# self.critic_optimizer.zero_grad()
		# # We set retain grad = true here
		# # critic_loss.backward()
		# critic_loss.backward()
		# self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			if not freeze_actor_update:
				# Optimize the critic
				self.critic_optimizer.zero_grad()
				self.actor_optimizer.zero_grad()

				# We set retain grad = true here
				# critic_loss.backward()
				# critic_loss.backward(retain_graph = True)
				critic_loss.backward()
				# Optimize the actor 

				actor_loss.backward()
				self.critic_optimizer.step()
				self.actor_optimizer.step()

				# Update the frozen target models
				for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		else:
			# Optimize the critic
			self.critic_optimizer.zero_grad()
			# We set retain grad = true here
			# critic_loss.backward()
			critic_loss.backward()
			self.critic_optimizer.step()


		return loss_dict


	def load_demonstration_buffer(self, demon_buffer, buffer_size_expert, bc_lambda):
		self.demonstration_buffer = demon_buffer
		self.buffer_size_expert = buffer_size_expert
		self.bc_lambda = bc_lambda

	def pretrain(self, replay_buffer, batch_size=4096, no_update=False):
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		batch_size = reward.shape[0]
		action = torch.clamp(action, -1, 1)

		if self.rnn:
			self.actor.flatten_parameters()
	
		pred_action = self.actor(state)
	
		actor_loss = F.mse_loss(pred_action, action)

		if not no_update:
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

		return actor_loss.item()

	def hard_update_target(self):
		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(param.data)

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)








'''
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)

		self.max_action = max_action


	def forward(self, state):
		a = F.tanh(self.l1(state))
		a = F.tanh(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		freeze_actor = 15000
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0
		self.freeze_actor = freeze_actor

		# TensorBoard Record parameter
		self.actor_loss = 0
		self.critic_loss = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=100):
		self.total_it += 1

		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)

			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		self.critic_loss = critic_loss

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0 and self.total_it > self.freeze_actor:

			# Compute actor losse
			actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
			self.actor_loss = actor_loss

			# Optimize the actor
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		print("Load model: "+filename)

'''

