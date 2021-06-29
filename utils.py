import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import copy
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(5e5)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

	def save(self, file):
		np.savez(file, max_size = self.max_size,
					   ptr = self.ptr,
					   size = self.size,
					   state = self.state,
					   action = self.action,
					   next_state = self.next_state,
					   reward = self.reward,
					   not_done = self.not_done)

	def load(self, file):
		npReplayBuffer = np.load(file)
		self.max_size = npReplayBuffer["max_size"]
		self.ptr = npReplayBuffer["ptr"]
		self.size = npReplayBuffer["size"]
		self.state = npReplayBuffer["state"]
		self.action = npReplayBuffer["action"]
		self.next_state = npReplayBuffer["next_state"]
		self.reward = npReplayBuffer["reward"]
		self.not_done = npReplayBuffer["not_done"]
		print("Load ReplayBuffer: "+file)
		print("There are "+str(self.size)+" data in ReplayBuffer")

class ImgBuffer(object):
	def __init__(self, state_dim, img_high, img_width, max_size=int(2e4)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.img_high = img_high
		self.img_width = img_width

		self.state = np.zeros((max_size, state_dim))
		# self.img = np.zeros((max_size, 1, self.img_high, self.img_width))
		self.img = np.zeros((max_size, 1, self.img_high, self.img_width, 3))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, img):
		self.state[self.ptr] = state
		self.img[self.ptr][0] = img

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind][0]).to(self.device),
			torch.FloatTensor(self.img[ind]).to(self.device),
		)

	def save(self, file):
		np.savez(file, max_size = self.max_size,
					   ptr = self.ptr,
					   size = self.size,
					   state = self.state,
					   img = self.img)

	def load(self, file):
		npReplayBuffer = np.load(file)
		self.max_size = npReplayBuffer["max_size"]
		self.ptr = npReplayBuffer["ptr"]
		self.size = npReplayBuffer["size"]
		self.state = npReplayBuffer["state"]
		self.img = npReplayBuffer["img"]
		print("Load ReplayBuffer: "+file)
		print("There are "+str(self.size)+" data in ReplayBuffer")


####################### For Recurrent Policies #######################


class HistoricalReplayBuffer():
	def __init__(self, observation_dim, action_dim, max_size=int(1e6)):
		self.observation_dim = observation_dim
		self.action_dim = action_dim
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		
		self.h = History()
		self.new_episode = True
		self.transitions = np.zeros((max_size, ), dtype=History)		

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	def split_validation(self, propotion=0.2):
		self.size_val = int(self.size*propotion)
		self.size = self.size - self.size_val
		self.validation_buffer = HistoricalReplayBuffer(self.observation_dim, self.action_dim, max_size=self.size_val)
		self.validation_buffer.size = self.size_val
		self.validation_buffer.transitions = self.transitions[:self.size_val].copy()
		self.transitions = self.transitions[self.size_val:]
		self.max_size = self.size


	def add(self, observation, action, next_observation, reward, is_done):
		if self.new_episode:
			self.new_episode = not self.new_episode
			self.h.init(observation, action.shape)

		self.h.add(action, reward, next_observation, is_done)

		if is_done:
			if len(self.h) > 3:
				self.transitions[self.ptr] = self.h.copy()
				self.ptr = (self.ptr + 1) % self.max_size
				self.size = min(self.size + 1, self.max_size)

			self.h = History()
			self.new_episode = True
			

	def sample(self, batch_size):
		if batch_size == -1:
			indices = np.array([i for i in range(self.size-1)])
			batch_size = self.size-1
		else:
			indices = np.random.randint(0, self.size-1, size=batch_size)

		rand_length = []

		return_h = []
		return_a = np.zeros((batch_size, self.action_dim))
		return_obn = np.zeros((batch_size, self.observation_dim+self.action_dim))
		return_r = np.zeros((batch_size, 1))
		return_ndones = np.zeros((batch_size, 1))

		for i, idx in enumerate(indices):
			try:
				if np.random.rand() < 0.1:
					sampled = self.transitions[idx].get(length=-1)  # sample the whole traj
				else:
					sampled = self.transitions[idx].get(rand_clamp=True)
			except Exception as e:
				print(e)
				print(idx)
				print(type(self.transitions[idx]))
				assert False

			o_i = np.array(sampled.h['o_i'])  # (h_length, observation_dim)
			a_i = np.array(sampled.h['a_i'])  # (h_length, action_dim)
			cat = torch.FloatTensor(np.concatenate((o_i, a_i), axis=1)).to(self.device)  # (h_length, ob_dim+act_dim)
			return_h.append(cat.clone())

			rand_length.append(len(sampled))
			return_a[i] = sampled.action_n
			return_obn[i] = np.concatenate([sampled.observation_n.reshape(-1, ), sampled.action_n.reshape(-1, )])
			return_r[i] = sampled.reward
			return_ndones[i] = 1. - sampled.is_done

		return_h = rnn_utils.pad_sequence(return_h, batch_first=True)
		return_h = rnn_utils.pack_padded_sequence(return_h, rand_length, batch_first=True, enforce_sorted=False)

		return (
			return_h,
			torch.FloatTensor(return_a).to(self.device),
			torch.FloatTensor(return_obn).to(self.device),
			torch.FloatTensor(return_r).to(self.device),
			torch.FloatTensor(return_ndones).to(self.device)
		)
	def save(self, file):
		np.savez(file, max_size = self.max_size,
					   ptr = self.ptr,
					   size = self.size,
					   h = self.h)

	def load(self, file):
		npReplayBuffer = np.load(file)
		self.max_size = npReplayBuffer["max_size"]
		self.ptr = npReplayBuffer["ptr"]
		self.size = npReplayBuffer["size"]
		self.h = npReplayBuffer["h"]
		print("Load ReplayBuffer: "+file)
		print("There are "+str(self.size)+" data in ReplayBuffer")


class History:  # History itself is a transition composed of (h_{i-1}, a_i, o_i, r_i, done_i)
	def __init__(self, maxlen=200, h:dict=None, action_n=None, observation_n=None, reward=None, is_done=False):

		if h is not None: 
			assert type(h) is dict
			assert 'o_i' in h.keys() and 'a_i' in h.keys()

		self.maxlen = maxlen
		self.action_n = action_n  # next action
		self.observation_n = observation_n  # next observation
		self.reward = reward
		self.is_done = is_done

		self.h = h or { 'o_i': [], 'a_i': [], 'r_i': [] }

	def __len__(self):
		return len(self.h['o_i'])

	def init(self, observation_0, action_shape):
		assert type(observation_0) == np.ndarray
		self.h['o_i'].append(observation_0.copy())
		self.h['a_i'].append(np.zeros(action_shape))

	def add(self, action, reward, next_observation, is_done): 
		assert type(action) == np.ndarray and type(next_observation) == np.ndarray
		if self.observation_n is not None:
			assert self.action_n is not None and self.reward is not None
			self.h['o_i'].append(self.observation_n.copy())
			self.h['a_i'].append(self.action_n.copy())
			self.h['r_i'].append(self.reward)
		self.observation_n = next_observation.copy()
		self.action_n = action.copy()
		self.reward = reward
		self.is_done = is_done

	def _get_length(self, length=-1):
		# Return type: History
		# 長度最小為1，返回 initial observation 與一個timestep的資料，len(self)=1
		# 若length < 1 則視為抽取該比例的長度
		# 若length == -1 則抽取整條軌跡
		if length == -1:
			length  = len(self)
		else:
			length = int(np.round(len(self)*length)) if length < 1 else int(length)
			length = int(np.clip(length, 1, len(self)))

		new_h = {
			'o_i': self.h['o_i'][:length].copy(),
			'a_i': self.h['a_i'][:length].copy(),
			'r_i': self.h['r_i'][:length-1].copy()
		}
		whole = length == len(self)
		observation_n = self.observation_n.copy() if whole else self.h['o_i'][length]
		action_n = self.action_n.copy() if whole else self.h['a_i'][length]
		reward = self.reward if whole else self.h['r_i'][length-1]
		is_done = self.is_done if whole else False

		return History(maxlen=self.maxlen, h=new_h, action_n=action_n, observation_n=observation_n, reward=reward, is_done=is_done)

	def copy(self):
		return History(maxlen=self.maxlen, h=self.h, action_n=self.action_n.copy(), observation_n=self.observation_n.copy(), reward=self.reward, is_done=self.is_done)

	def get(self, length=None, rand_clamp=False):
		if rand_clamp and length is None: length = np.random.rand()
		return self.copy() if length is None else self._get_length(length)
		

	def __str__(self):
		return '<class History> \n\th=%s, \n\taction_n=%s, \n\tobservation_n=%s, \n\treward=%s, \n\tis_done=%s, \n\tlen=%d' % (
					str(self.h), str(self.action_n), str(self.observation_n), str(self.reward), str(self.is_done), len(self)
				)

def gen_fake_history(n=40, ob_dim=6 , action_dim=4 ):
	rand_ob = lambda: np.around(np.random.normal(100, 10, ob_dim), -1)
	rand_ac = lambda: np.around(np.random.normal(1, 0.2, action_dim), 2)
	rand_r = lambda: np.around(np.random.rand(), 2)
	h = History(n*2)
	init_ob = rand_ob()
	h.init(init_ob, action_dim)
	#print('init_ob=', init_ob)

	for i in range(n):
		a_i, r_i, o_i = rand_ac(), rand_r(), rand_ob()
		h.add(a_i, r_i, o_i, False)
		#print('a[%d], r[%d], o[%d] = ' % (i, i, i) , a_i, r_i, o_i )

	return h

def gen_fake_transition(n_ep=10, n_step_top=500, ob_dim=(6, ), action_dim=(2, )):
	rand_ob = lambda: np.around(np.random.normal(10, 1, ob_dim), 1)
	rand_ac = lambda: np.around(np.random.normal(1, 0.2, action_dim), 2)
	rand_r = lambda: np.around(np.random.rand(), 2)
	b = HistoricalReplayBuffer(ob_dim, action_dim)
	ep_len = []
	for ep in range(n_ep):
		observation = rand_ob()
		ep_length = int(np.round(np.random.rand()*n_step_top))
		ep_len.append(ep_length)
		for step in range( ep_length ):
			action = rand_ac()
			next_observation = rand_ob()
			reward = rand_r()
			is_done = False if step < ep_length - 1 else True
			b.add(observation, action, next_observation, reward, is_done)
			observation = next_observation.copy()
	print('buffer length:', b.size)
	print('memory_length:', ep_len)
	return b

