import os
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from .cnn_encoder import CnnEncoder


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FAMEAgent(nn.Module):
    def __init__(self, envs, fast=True):
        super().__init__()
        self.network = CnnEncoder(hidden_dim=512, layer_init=layer_init)
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.fastmeta = '_fast' if fast else '_meta'

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def forward(self, x):
        hidden = self.network(x)
        logits = self.actor(hidden)
        return logits

    def save(self, dirname):

        os.makedirs(dirname, exist_ok=True)
        torch.save(self.actor, f"{dirname}/actor{self.fastmeta}.pt")
        torch.save(self.network, f"{dirname}/encoder{self.fastmeta}.pt")
        torch.save(self.critic, f"{dirname}/critic{self.fastmeta}.pt")

    def load(self, dirname, envs, load_critic=True, reset_actor=False, map_location=None, ):
        model = FAMEAgent(envs)
        model.network = torch.load(f"{dirname}/encoder{self.fastmeta}.pt", map_location=map_location)
        if not reset_actor:
            model.actor = torch.load(f"{dirname}/actor{self.fastmeta}.pt", map_location=map_location)
        if load_critic:
            model.critic = torch.load(f"{dirname}/critic{self.fastmeta}.pt", map_location=map_location)
        return model
