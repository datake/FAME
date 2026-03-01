import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class CNN(nn.Module):
	def __init__(self, in_channels, num_actions):
		super().__init__()

		# the same as https://github.com/initial-h/CEER/blob/main/networks.py except (1) out_channels = 16 / out_features=256 (2) intialization
		self.conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)

		def size_linear_unit(size, kernel_size=3, stride=1):
			return (size - (kernel_size - 1) - 1) // stride + 1
		num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 32
		self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=256)
		self.output = nn.Linear(in_features=256, out_features=num_actions)

		self._init_weights()

	def _init_weights(self):
		torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.fc_hidden.weight, nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.output.weight, nonlinearity='relu')

	def forward(self, x):
		x = F.relu(self.conv(x))
		x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))
		return self.output(x)

	# def get_action_distribution(self, x):
	# 	logits = self.forward(x)
	# 	action_probs = F.softmax(logits, dim=-1)
	# 	action_dist = Categorical(probs=action_probs)
	# 	return action_dist

class CNN_three_heads(nn.Module):
	def __init__(self, in_channels, num_actions):
		super().__init__()

		self.conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1)

		def size_linear_unit(size, kernel_size=3, stride=1):
			return (size - (kernel_size - 1) - 1) // stride + 1
		num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 32
		self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=256)
		self.output_1 = nn.Linear(in_features=256, out_features=num_actions)
		self.output_2 = nn.Linear(in_features=256, out_features=num_actions)
		self.output_3 = nn.Linear(in_features=256, out_features=num_actions)

		self._init_weights()

	def _init_weights(self):
		torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.fc_hidden.weight, nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.output_1.weight, nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.output_2.weight, nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.output_3.weight, nonlinearity='relu')

	def forward(self, x):
		x = F.relu(self.conv(x))
		x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))
		return self.output_1(x), self.output_2(x), self.output_3(x)

class CNN_half(nn.Module):
	def __init__(self, in_channels, num_actions):
		super().__init__()

		self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

		def size_linear_unit(size, kernel_size=3, stride=1):
			return (size - (kernel_size - 1) - 1) // stride + 1
		num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
		self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128) # [1024, 128]
		self.output = nn.Linear(in_features=128, out_features=num_actions)

		self._init_weights()

	def _init_weights(self):
		torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.fc_hidden.weight, nonlinearity='relu')
		torch.nn.init.kaiming_normal_(self.output.weight, nonlinearity='relu')

	def forward(self, x):
		x = F.relu(self.conv(x))
		x = F.relu(self.fc_hidden(x.view(x.size(0), -1))) # -> [batch, 1024] -> [batch, 128]
		return self.output(x) # [batch, 128] -> [batch, num_actions]

