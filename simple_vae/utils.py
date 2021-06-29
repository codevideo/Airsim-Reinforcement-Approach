import numpy as np
import torch


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
		self.img = np.zeros((max_size, 1, self.img_high, self.img_width))

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
