import os.path

import torch
from collections import deque
import random
import numpy as np
import pickle

class expReplay():
	def __init__(self, batch_size, device, max_size=100000, buffer_file='data_FAME/Freeway_buffer_fast.pkl'):
		self.memory = deque(maxlen=max_size)
		self.batch_size = batch_size
		self.device = device
		self.buffer_file = buffer_file

		if os.path.exists(self.buffer_file):
			self.load()
			print(f"[INFO] Loaded replay buffer from {self.buffer_file} with {len(self.memory)} transitions.")
		else:
			print(f"[INFO] No replay buffer found at {self.buffer_file}. Starting with an empty buffer.")

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

	def save(self):
		with open(self.buffer_file, 'wb') as f:
			pickle.dump(self.memory, f)
		print(f"[INFO] Saved replay buffer to {self.buffer_file}.")

	def load(self):
		with open(self.buffer_file, 'rb') as f:
			self.memory = pickle.load(f)


class expReplay_Meta():
	def __init__(self, batch_size, device, max_size=100000):
		self.memory = deque(maxlen=max_size)
		self.batch_size = batch_size
		self.device = device
		# self.buffer_file = buffer_file

		# if os.path.exists(self.buffer_file):
		# 	self.load()
		# 	print(f"[INFO] Loaded replay buffer from {self.buffer_file} with {len(self.memory)} transitions.")
		# else:
		# 	print(f"[INFO] No replay buffer found at {self.buffer_file}. Starting with an empty buffer.")

	# [state, acton]
	def store(self, obs, action):
		# state = np.moveaxis(obs, 2, 0)
		# state = torch.tensor(state, dtype=torch.float)
		# action = torch.tensor([action], dtype=torch.int64)

		for i in range(obs.shape[0]):
			self.memory.append((obs[i], action[i]))

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

	# def save(self):
	# 	if not os.path.exists('data_FAME'):
	# 		os.makedirs('data_FAME')
	# 	with open(self.buffer_file, 'wb') as f:
	# 		pickle.dump(self.memory, f)
	# 	print(f"[INFO] Saved replay buffer to {self.buffer_file}.")
	#
	# def load(self):
	# 	with open(self.buffer_file, 'rb') as f:
	# 		self.memory = pickle.load(f)
