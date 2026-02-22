import numpy as np
import torch
import torch.nn.functional as F
from agent import utils
import abc
from agent.critic import DoubleQCritic
from agent.actor import DiagGaussianActor
import os

class Agent(object):
    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    @abc.abstractmethod
    def train(self, training=True):
        """Sets the agent in either training or evaluation mode."""

    @abc.abstractmethod
    def update(self, replay_buffer, step):
        """Main function of the agent that performs learning."""

    @abc.abstractmethod
    def act(self, obs, sample=False):
        """Issues an action given an observation."""

class SACAgent(Agent):
    """SAC algorithm."""

    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg=DoubleQCritic,
                 actor_cfg=DiagGaussianActor, discount=0.99, init_temperature=0.1, alpha_lr=1e-4, alpha_betas=[0.9, 0.999],
                 actor_lr=1e-4, actor_betas=[0.9, 0.999], actor_update_frequency=1, critic_lr=1e-4,
                 critic_betas=[0.9, 0.999], critic_tau=0.005, critic_target_update_frequency=1,
                 batch_size=256, learnable_temperature=True,
                 normalize_state_entropy=True,):
        super().__init__()

        self.action_range = action_range
        # print('action_range:',self.action_range)
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic_cfg = critic_cfg
        self.critic_lr = critic_lr
        self.critic_betas = critic_betas
        self.normalize_state_entropy = normalize_state_entropy
        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.alpha_betas = alpha_betas
        self.actor_cfg = actor_cfg
        self.actor_betas = actor_betas
        self.alpha_lr = alpha_lr

        self.critic = self.critic_cfg(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=256,hidden_depth=2).to(self.device)
        self.critic_target = self.critic_cfg(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=256,hidden_depth=2).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor = self.actor_cfg(obs_dim=obs_dim,action_dim=action_dim,hidden_dim=256,hidden_depth=2,log_std_bounds=[-5, 2]).to(self.device)
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True

        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=actor_betas)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=critic_betas)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=alpha_betas)

        # change mode
        self.train()
        self.critic_target.train()

    def eval(self):
        self.train(False)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done):
        # print('obs:',obs.shape,'action:',action.shape,'reward:',reward.shape,'next_obs:',next_obs.shape,'not_done:',not_done.shape)
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        # print('next_log_prob',log_prob[0],next_action[0])
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        # print('target_Q:',target_Q.shape)
        target_Q = target_Q.detach()
        # print('target_Q 1:',target_Q.shape)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        # print('current_Q1:',current_Q1.shape,current_Q2.shape)
        critic_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)) * 0.5

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_loss

    def compute_target_q(self, reward, next_obs, not_done):
        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        return target_Q.detach()

    def update_with_target_q(self, obs, action, target_Q, step):
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)) * 0.5
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if step % self.actor_update_frequency == 0:  # actor_update_frequency = 1
            loss_pi, loss_alpha = self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_frequency == 0:  # critic_target_update_frequency = 2
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        return {"alpha": loss_alpha.item(),
                "actor": loss_pi.item(),
                "critic": critic_loss.item(),
                }

    def save(self, model_dir, model_name):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(self.actor.state_dict(), '%s/%s_actor.pt' % (model_dir, model_name))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pt' % (model_dir, model_name))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pt' % (model_dir, model_name))

    def load(self, model_dir, model_name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pt' % (model_dir, model_name)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pt' % (model_dir, model_name)))
        self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pt' % (model_dir, model_name)))

    def update_actor_and_alpha(self, obs):
        # SAC
        dist = self.actor(obs)
        action = dist.rsample()
        # print('---')
        # print('action',action.shape)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        # print('log_prob:',log_prob.shape)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        # print('actor_Q',actor_Q.shape,log_prob.shape)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        # print('actor_loss:',actor_loss)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss

    def sac_update(self, replay_buffer, step):
        obs, action, reward, success, next_obs, not_done_no_max = replay_buffer.sample(self.batch_size)
        # print('shape:',obs.shape,action.shape,reward.shape,next_obs.shape,not_done_no_max.shape)
        obs, action, reward, success, next_obs, not_done_no_max = replay_buffer.as_torch(obs, action, reward, success, next_obs, not_done_no_max)

        loss_q = self.update_critic(obs, action, reward, next_obs, not_done_no_max)

        if step % self.actor_update_frequency == 0:  # actor_update_frequency = 1
            loss_pi, loss_alpha = self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_frequency == 0:  # critic_target_update_frequency = 2
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        return {"alpha": loss_alpha.item(),
                "actor": loss_pi.item(),
                "critic": loss_q.item(),
                }

    def actor_nll(self, replay_buffer,update_num):
        print_interval = update_num // 5

        for i in range(update_num):
            obs, action, reward, success, next_obs, not_done_no_max = replay_buffer.sample(self.batch_size)
            # print('action:',action[1])
            obs, action, reward, success, next_obs, not_done_no_max = replay_buffer.as_torch(obs, action, reward, success, next_obs, not_done_no_max)
            eps = 1e-6
            action = torch.clamp(action, min=-1.0 + eps, max=1.0 - eps)
            assert not torch.isnan(obs).any(), f'obs nan: {obs}'
            dist = self.actor(obs)
            assert not torch.isnan(dist.loc).any(), f'dist: {dist.loc}'
            assert torch.all(action > -1) and torch.all(action < 1), f'action out of range: {action}'
            assert not torch.isnan(action).any(), f'action nan: {action}'
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            actor_loss = -log_prob.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            if i % print_interval == 0:
                print('actor_loss:',i, actor_loss)
        return actor_loss.item()

    def actor_wd_loss(self, current_agent, pre_meta_agent, current_buffer,current_buffer_start_idx, meta_buffer, update_num):
        print_interval = update_num // 5

        for i in range(update_num):
            if pre_meta_agent is not None:
                meta_obs, meta_action, meta_reward, meta_success, meta_next_obs, meta_not_done_no_max = meta_buffer.sample(self.batch_size)
                meta_obs = torch.as_tensor(meta_obs, device=self.device).float()
                with torch.no_grad():
                    pre_meta_agent_dist = pre_meta_agent.actor(meta_obs)
                    meta_mu, meta_std = pre_meta_agent_dist.loc, pre_meta_agent_dist.scale
                    # print('meta shape:',meta_mu.shape,meta_std.shape)

                dist1 = self.actor(meta_obs)
                mu1, std1 = dist1.loc, dist1.scale
                # print('shape1:',mu1.shape,std1.shape)
                meta_loss = torch.mean(torch.square(mu1 - meta_mu).sum(-1) + torch.square(std1 - meta_std).sum(-1))
            else:
                meta_loss = 0

            current_obs, current_action, current_reward, current_success, current_next_obs, current_not_done_no_max = current_buffer.sample_last(current_buffer_start_idx, self.batch_size)
            current_obs = torch.as_tensor(current_obs, device=self.device).float()
            with torch.no_grad():
                current_agent_dist = current_agent.actor(current_obs)
                current_mu, current_std = current_agent_dist.loc, current_agent_dist.scale
                # print('current shape:',current_mu.shape,current_std.shape)

            dist2 = self.actor(current_obs)
            mu2, std2 = dist2.loc, dist2.scale
            # print('shape2:',mu2.shape,std2.shape)
            current_loss = torch.mean(torch.square(mu2 - current_mu).sum(-1) + torch.square(std2 - current_std).sum(-1))

            actor_loss = meta_loss + current_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            if i % print_interval == 0:
                print('actor_loss:',i, actor_loss)

        return actor_loss.item()



