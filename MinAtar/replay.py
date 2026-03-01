import torch
from collections import deque
import random
import numpy as np

class expReplay():
	def __init__(self, batch_size, device, max_size=100000):
		self.memory = deque(maxlen=max_size)
		self.batch_size = batch_size
		self.device = device

	def store(self, obs, action, next_obs, reward, done):
		state = np.moveaxis(obs, 2, 0)
		state = torch.tensor(state, dtype=torch.float)
		next_state = np.moveaxis(next_obs, 2, 0)
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor([action], dtype=torch.int64)
		reward = torch.tensor([reward], dtype=torch.float)
		done = torch.tensor([float(done)], dtype=torch.float)

		self.memory.append((state, action, next_state, reward, done))

	def sample(self):
		if len(self.memory) >= self.batch_size:
			batch = random.sample(self.memory, self.batch_size)
		else:
			batch = list(self.memory)
		state, action, next_state, reward, done = map(torch.stack, zip(*batch))
		return state.to(self.device), action.to(self.device), next_state.to(self.device), reward.to(self.device), done.to(self.device)

	def size(self):
		return len(self.memory)

	def delete(self):
		self.memory.clear()




class expReplay_Meta():
	def __init__(self, batch_size, device, max_size=100000):
		self.memory = deque(maxlen=max_size)
		self.batch_size = batch_size
		self.device = device

	# [state, acton]
	def store(self, obs, action):
		state = np.moveaxis(obs, 2, 0)
		state = torch.tensor(state, dtype=torch.float)
		action = torch.tensor([action], dtype=torch.int64)
		self.memory.append((state, action))

	def sample(self):
		if len(self.memory) >= self.batch_size:
			batch = random.sample(self.memory, self.batch_size)
		else:
			batch = list(self.memory)
		state, action = map(torch.stack, zip(*batch))
		return state.to(self.device), action.to(self.device)

	def size(self):
		return len(self.memory)

	def delete(self):
		self.memory.clear()

	def copy_to(self, target_buffer):
		"""
		Copy all elements from current buffer to target_buffer.
		"""
		for item in self.memory: # (state, action)
			target_buffer.memory.append(item)
		return target_buffer

class expReplay_PM():
	def __init__(self, max_size, batch_size, device):
		self.memory = deque(maxlen=max_size)
		self.batch_size = batch_size
		self.device = device

	# [state, acton, val_p]
	def store(self, obs, action, val_p):
		state = np.moveaxis(obs, 2, 0)
		state = torch.tensor(state, dtype=torch.float)
		action = torch.tensor([action], dtype=torch.int64)
		val_p = torch.tensor([val_p], dtype=torch.float)

		self.memory.append((state, action, val_p))

	def sample(self):
		if len(self.memory) >= self.batch_size:
			batch = random.sample(self.memory, self.batch_size)
		else:
			batch = list(self.memory)
		state, action, val_p = map(torch.stack, zip(*batch))
		return state.to(self.device), action.to(self.device), val_p.to(self.device)

	def size(self):
		return len(self.memory)

	def delete(self):
		self.memory.clear()

class expReplay_Large():
	def __init__(self, batch_size, device):
		self.memory = deque(maxlen=500000)
		self.batch_size = batch_size
		self.device = device

	def store(self, obs, action, next_obs, reward, done):
		state = np.moveaxis(obs, 2, 0)
		state = torch.tensor(state, dtype=torch.float)
		next_state = np.moveaxis(next_obs, 2, 0)
		next_state = torch.tensor(next_state, dtype=torch.float)
		action = torch.tensor([action], dtype=torch.int64)
		reward = torch.tensor([reward], dtype=torch.float)
		done = torch.tensor([float(done)], dtype=torch.float)

		self.memory.append((state, action, next_state, reward, done))

	def sample(self):
		if len(self.memory) >= self.batch_size:
			batch = random.sample(self.memory, self.batch_size)
		else:
			batch = list(self.memory)
		state, action, next_state, reward, done = map(torch.stack, zip(*batch))
		return state.to(self.device), action.to(self.device), next_state.to(self.device), reward.to(self.device), done.to(self.device)

	def size(self):
		return len(self.memory)

	def delete(self):
		self.memory.clear()