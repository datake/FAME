import numpy as np
import torch
import os
from collections import defaultdict

class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, device, window=1):
        self.capacity = capacity
        self.device = device

        self.obs_shape = obs_shape
        self.action_shape = action_shape

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.obs_dtype = np.float32 if len(self.obs_shape) == 1 else np.uint8

        self.obses = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.next_obses = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.successes = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((self.capacity, 1), dtype=np.float32)
        self.window = window

        self.idx = 0
        self.last_save = 0
        self.full = False

    def reset(self):
        self.obses = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.next_obses = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        self.actions = np.empty((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.successes = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((self.capacity, 1), dtype=np.float32)
        self.idx = 0
        self.last_save = 0
        self.full = False

    def save_data(self,path):
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez_compressed(path + '/offline_data',
                            obses = self.obses,
                            next_obses = self.next_obses,
                            actions = self.actions,
                            rewards = self.rewards,
                            successes = self.successes,
                            not_dones = self.not_dones,
                            not_dones_no_max = self.not_dones_no_max,
                            others = [self.window,self.idx,self.last_save,self.full])
        print('data saved!',path)

    def load_data(self,path):
        print('data loading ...',path)

        try:
            data = np.load(path + '/offline_data.npz')
            assert len(data['obses']) <= self.capacity, [len(data['obses']),self.capacity]
            self.obses = data['obses']
            self.next_obses = data['next_obses']
            self.actions = data['actions']
            self.rewards = data['rewards']
            self.successes = data['successes']
            self.not_dones = data['not_dones']
            self.not_dones_no_max = data['not_dones_no_max']

            self.window = data['others'][0]
            self.idx = data['others'][1]
            self.last_save = data['others'][2]
            self.full = data['others'][3]

            # self.freqency_distribution()
            return True
        except BaseException as e:
            print(e)
            return False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, success, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.successes[self.idx], success)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)

        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        successes = self.successes[idxs]
        next_obses = self.next_obses[idxs]
        not_dones_no_max = self.not_dones_no_max[idxs]

        return obses, actions, rewards, successes, next_obses, not_dones_no_max

    def sample_last(self, start_idx, batch_size):
        idxs = np.random.randint(start_idx, self.capacity if self.full else self.idx, size=batch_size)

        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        successes = self.successes[idxs]
        next_obses = self.next_obses[idxs]
        not_dones_no_max = self.not_dones_no_max[idxs]

        return obses, actions, rewards, successes, next_obses, not_dones_no_max

    def as_torch(self,obses, actions, rewards, successes, next_obses, not_dones_no_max):
        return (torch.as_tensor(obses, device=self.device).float(),
                torch.as_tensor(actions, device=self.device).float(),
                torch.as_tensor(rewards, device=self.device).float(),
                torch.as_tensor(successes, device=self.device).float(),
                torch.as_tensor(next_obses, device=self.device).float(),
                torch.as_tensor(not_dones_no_max, device=self.device).float())

class Collector():
    def __init__(self,env,replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer
        self.obs = None
        self.return_list = []
        self.success_list = []

    def reset_stats(self):
        self.return_list = []
        self.success_list = []

    def initial_collect(self,steps):
        # collect for some random steps
        print('initial_collect ...')
        self.obs,_ = self.env.reset()
        for i in range(steps):
            actions = self.env.env.action_space.sample()
            next_obs, rewards, terminateds, truncateds, infos = self.env.no_count_step(actions) # do not count the random steps

            # allow infinite bootstrap
            done = float(terminateds) or float(truncateds) # this env is weird, done is always false: return (np.array(self._last_stable_obs, dtype=np.float64), reward, False, truncate, info,)
            done_no_max = 0. if truncateds else done
            # if done:
            #     print('done!')
            # if done_no_max:
            #     print('done_no_max!')
            # if truncateds:
            #     print('truncateds!')
            # assert np.all((actions >= -1.) & (actions <= 1.0)), actions
            self.replay_buffer.add(self.obs, actions, rewards, infos.get("success", False), next_obs, done, done_no_max)

            self.obs = next_obs
            if terminateds or truncateds:
                self.obs, _ = self.env.reset()
                self.return_list.append(infos['episode']['r'])
                self.success_list.append(self.env.env.successes[-1])
                assert len(self.success_list) == len(self.return_list), [len(self.success_list), len(self.return_list)]

    def initial_agent_collect(self,steps,agent_list,num_eval_runs):
        print('------ initial_collect ...')
        return_dict = defaultdict(list)
        success_dict = defaultdict(list)
        episode_count = 0
        agent_idx = 0
        self.obs,_ = self.env.reset()
        for i in range(steps): # take care the num_eval_runs * num_agents should be less than steps, right now is 200*10*2, is ok
            if agent_idx == len(agent_list):
                actions = self.env.env.action_space.sample()
            else:
                with torch.inference_mode():
                    actions = agent_list[agent_idx].act(self.obs)
            next_obs, rewards, terminateds, truncateds, infos = self.env.no_count_step(actions)
            # allow infinite bootstrap
            done = float(terminateds) or float(truncateds)
            done_no_max = 0. if truncateds else done
            # assert np.all((actions >= -1.) & (actions <= 1.0)), actions
            self.replay_buffer.add(self.obs, actions, rewards, infos.get("success", False), next_obs, done, done_no_max)

            self.obs = next_obs
            if terminateds or truncateds:
                self.obs, _ = self.env.reset()
                self.return_list.append(infos['episode']['r'])
                self.success_list.append(self.env.env.successes[-1])
                assert len(self.success_list) == len(self.return_list), [len(self.success_list), len(self.return_list)]

                return_dict[agent_idx].append(self.return_list[-1])
                success_dict[agent_idx].append(self.success_list[-1])
                episode_count += 1

                if episode_count == num_eval_runs:
                    episode_count = 0
                    if agent_idx < len(agent_list):
                        agent_idx += 1 # the last agent will be the random agent

        return return_dict, success_dict


    def run_one_step(self, current_step, agent):
        if current_step == 0:
            self.obs,_ = self.env.reset()
        with torch.inference_mode():
            actions = agent.act(self.obs, sample=True)
        # actions = self.env.env.action_space.sample()
        next_obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

        # allow infinite bootstrap
        done = float(terminateds) or float(truncateds)
        done_no_max = 0. if truncateds else done
        # assert np.all((actions >= -1.) & (actions <= 1.0)), actions
        self.replay_buffer.add(self.obs, actions, rewards, infos.get("success", False), next_obs, done, done_no_max)

        self.obs = next_obs
        if terminateds or truncateds:
            self.obs, _ = self.env.reset()
            if infos.get("episode", False) and len(self.env.env.successes) > 0:
                self.return_list.append(infos['episode']['r']) # no episode if self.make_task() change tasks
                self.success_list.append(self.env.env.successes[-1])
                assert len(self.success_list) == len(self.return_list), [len(self.success_list), len(self.return_list)]