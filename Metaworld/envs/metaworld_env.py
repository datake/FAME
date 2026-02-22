#####
# Get the metaworld envs
# Custom envs to train multiple metaworld envs in sequence
#
#
#####

import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE, ALL_V2_ENVIRONMENTS_GOAL_HIDDEN


from typing import Callable, Dict, List
import gymnasium as gym
from gymnasium.wrappers import DtypeObservation, TimeLimit, RecordEpisodeStatistics, RecordVideo, ClipAction
import numpy as np
from envs.metaworld_env_sequences import GOOD_RPO_SEQS, RPO20_SEQ, RPO10_SEQ, RPO10_SHORT


version = 2

EnvFn = Callable[[], gym.Env]

# these envs are learned by PPO within 2M steps each.
GOOD_ENVS = ['handle-press-side-v2', 'faucet-close-v2', 'plate-slide-v2', 'window-open-v2',
             'reach-wall-v2', 'button-press-v2', 'plate-slide-side-v2', 'handle-press-v2']
# window-open seems really hard in the sequential setup

# SMALL_SEQUENCE_ENVS = ['plate-slide-v2', 'handle-press-v2', 'button-press-v2', 'faucet-close-v2',
#                        'plate-slide-side-v2', 'handle-press-side-v2']

SMALL_SEQUENCE_ENVS = ['handle-press-v2', 'plate-slide-side-v2', 'button-press-v2', 'plate-slide-v2', 'handle-press-side-v2', 'faucet-close-v2']
class MetaWorldSingleEnvSequence:
    def __init__(self,  change_freq=1e7, base_task_name=None, env_sequence="metaworld_sequence_set1", goal_hidden=True, normalize_obs="straight", normalize_avg_coef=0.0001,
                 normalize_rewards=True, reset_obs_stats=False,
                 change_when_solved=False,
                 capture_video=False, seed=None, *args, **kwargs):
        '''
        This class is used to generate a stream of tasks in one of the metaworld envs
        Each task is generated with a new random seed. (I think you can go on forever?)
        base_task_name: name of metaworld env e.g. 'reach-v2'
        change_freq: number of steps until a task change
        env_sequence: String that specifies which env sequence to use
        normalize_obs: If True, keeps a moving average of observations and normalizes the observations
        normalize_avg_coef: To be used in the moving avg
        reset_obs_stats: If True, resets the observation normalization stats every time the task changes
        goal_hidden: If True, the agent does not observe the goal location
        change_when_solved: Change to the next task whenever the agent receives above 90% success 5 times in a row.
        '''

        self.base_task_name = base_task_name
        self.env_sequence = env_sequence

        if self.env_sequence is not None:
            if self.env_sequence[-2:].isnumeric(): # we expect something like "set15" or "set3"
                env_set_id = int(self.env_sequence[-2:])
            else:
                env_set_id = int(self.env_sequence[-1])
            self.env_list = RPO10_SEQ[env_set_id - 1]

        self.goal_str = '-goal-hidden' if goal_hidden else '-goal-observable'
        self.metaworld_envs = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN if goal_hidden else ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        if base_task_name is not None:
            self.base_task_class = self.metaworld_envs[base_task_name + self.goal_str]

        # consider making the goal hidden

        self.env = None
        self.base_seed = seed
        self.current_seed = seed
        self.normalize_obs = normalize_obs
        self.normalize_avg_coef = normalize_avg_coef
        self.normalize_rewards = normalize_rewards
        self.reset_obs_stats = reset_obs_stats
        self.change_when_solved = change_when_solved

        self.change_freq = change_freq
        self.timestep_counter = 0
        self.task_counter = 0
        self.obs_mean = None
        self.obs_var = None
        self.obs_count = 1e-4  # initialize it to a small value (as per the gym.NormalizeObservation wrapper)
        self.bias_correction = False

        self.eval_success_history = []
        self._change_task_next_step = False

        self.rng = np.random.RandomState(seed=self.current_seed)
        # the default magnitude of the observations are about ~1 or smaller I think

        self.make_task()

    def reset(self):
        obs, info = self.env.reset()

        if self.normalize_obs is not None:
            obs = self._normalize_obs(obs)
            # self._update_obs_statistics(obs)
        # obs = obs.astype('float32')
        return obs, info

    def make_task(self):
        # self.current_seed += 100
        self.task_counter += 1

        if self.env_sequence is not None:
            if self.env_sequence[-2:].isnumeric(): # we expect something like "set15" or "set3"
                env_set_id = int(self.env_sequence[-2:])
            else:
                env_set_id = int(self.env_sequence[-1])

            # self.base_task_name = RPO10_SHORT[env_set_id-1][(self.task_counter-1) % len(RPO10_SHORT[0])]
            self.base_task_name = RPO10_SEQ[env_set_id-1][(self.task_counter-1) % len(self.env_list)]
            # self.base_task_name = SEQUENCE_ENVS[(self.task_counter-1) % len(SMALL_SEQUENCE_ENVS)]  # for testing
            self.base_task_class = self.metaworld_envs[self.base_task_name + self.goal_str]

        temp_env = self.base_task_class(seed=self.current_seed)

        self.env = self._wrap_env(temp_env)

        self.env.action_space.seed(self.current_seed)
        self.env.observation_space.seed(self.current_seed)

        if self.obs_mean is None or self.reset_obs_stats:
            self.obs_mean = np.zeros(self.env.observation_space.shape)

            if self.bias_correction:
                self.obs_var = np.zeros(self.env.observation_space.shape)
            else:
                self.obs_var = np.ones(self.env.observation_space.shape)

        print(f'TASK {self.task_counter}  {self.current_seed} {self.base_task_name}')
        return

    def set_task(self, task_name): # for evaluation
        self.base_task_name = task_name
        self.base_task_class = self.metaworld_envs[self.base_task_name + self.goal_str]

        temp_env = self.base_task_class(seed=self.current_seed)

        self.env = self._wrap_env(temp_env)

        self.env.action_space.seed(self.current_seed)
        self.env.observation_space.seed(self.current_seed)

        if self.obs_mean is None or self.reset_obs_stats:
            self.obs_mean = np.zeros(self.env.observation_space.shape)

            if self.bias_correction:
                self.obs_var = np.zeros(self.env.observation_space.shape)
            else:
                self.obs_var = np.ones(self.env.observation_space.shape)
        # print(f'TASK {self.base_task_name}')


    def _wrap_env(self, env, eval_mode=False):
        # env = RecordVideo(env)  # do this later

        # env = TransformObservation(env, lambda obs: obs.astype('float32'), env.observation_space)
        env = DtypeObservation(env, dtype=np.float32)
        # env = ClipAction(env) # why need this? it is -1 to 1, this makes it -inf to inf and then clip it to -1 to 1, seems no need ...
        if not eval_mode and self.normalize_rewards:
            env = gym.wrappers.TransformReward(env, lambda r: r / 500)
        env = TimeLimit(env, max_episode_steps=200)
        env = RecordEpisodeStatistics(env)
        env = SuccessCounter(env)
        # don't use normalize observations wrapper here because we need to use the same one for the test env
        # in evaluate_agent()
        # don't use any wrapper that depends on maintaining statistics
        return env

    def step(self, action):
        self.timestep_counter += 1

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.normalize_obs is not None:
            # print('!!!')
            self._update_obs_statistics(obs)
            obs = self._normalize_obs(obs)
            # print(obs[:8])
            # np.set_printoptions(suppress=True)
        # obs = obs.astype('float32')

        # reward = reward / 1000 # tried rescaling. consider if I should be doing this. Reward scale does matter to optimization

        if self.change_when_solved:
            if self._change_task_next_step:
                self._change_task_next_step = False
                self.make_task()  # change tasks
                truncated = True

        # elif (self.timestep_counter+1) % self.change_freq == 0: # why +1?
        elif self.timestep_counter % self.change_freq == 0:
                self.make_task()
                truncated = True
                # return obs, reward, terminated, True, info   # truncate current episode, get reset called

        return obs, reward, terminated, truncated, info

    def no_count_step(self, action): # run a step without counting it
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.normalize_obs is not None:
            self._update_obs_statistics(obs)
            obs = self._normalize_obs(obs)

        return obs, reward, terminated, truncated, info


    def evaluate_agent(self, agent, num_eval_episodes=10):
        ''' Runs and evaluation of the agent
        It runs on the current seed i.e. the current task '''

        test_env = self._wrap_env(self.base_task_class(seed=self.current_seed), eval_mode=True)
        obs, _ = test_env.reset()

        if self.normalize_obs is not None:
            obs = self._normalize_obs(obs)

        eval_results = {}
        episodic_returns = []
        agent.eval()

        while len(episodic_returns) < num_eval_episodes:
            action = agent.act(obs)
            # action = test_env.env.action_space.sample()

            next_obs, _, terminated, truncated, info = test_env.step(action)

            if self.normalize_obs is not None:
                next_obs = self._normalize_obs(next_obs)

            if "episode" in info:
                episodic_returns.append(info['episode']['r'])

            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if "episode" not in info:
            #             continue
            #         # print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
            #         episodic_returns.append(info["episode"]["r"][0])
            obs = next_obs

            if terminated or truncated:
                obs, _ = test_env.reset()

        agent.train()
        eval_results['episodic_returns'] = episodic_returns
        eval_results['successes'] = test_env.pop_successes()

        if self.change_when_solved:
            self.eval_success_history.append(np.mean(eval_results['successes']))
            self._change_task_next_step = True
        return eval_results

    def _update_obs_statistics(self, obs):
        ''' Update mean and variance statistics '''

        if self.normalize_obs.lower() == 'ema':
            # print("Using EMA")
            # print(self.timestep_counter)
            # print('obs', obs[:4])
            # print('obs mean', self.obs_mean[:4])
            self.obs_mean = (1-self.normalize_avg_coef) * self.obs_mean + self.normalize_avg_coef * obs
            # print('obs mean 2', self.obs_mean[:4])
            self.obs_var = (1-self.normalize_avg_coef) * self.obs_var + self.normalize_avg_coef * (obs - self.obs_mean)**2
            # print("NORM", np.linalg.norm(self.obs_var))
            # print('obs var', self.obs_var)
            # print(self.normalize_avg_coef)
            # print(obs[0:5], self.obs_mean[0:5])#, (1-self.normalize_avg_coef) * self.obs_mean + self.normalize_avg_coef * obs )

        elif self.normalize_obs.lower() == 'straight':
            mean = self.obs_mean
            var = self.obs_var
            count = self.obs_count

            delta = obs - mean
            tot_count = count + 1

            new_mean = mean + delta / tot_count
            new_var = var * count / tot_count + delta ** 2 * count / tot_count ** 2
            new_count = tot_count

            self.obs_mean = new_mean
            self.obs_var = new_var
            self.obs_count = new_count

    def _normalize_obs(self, obs):
        if self.timestep_counter == 0:
            return obs  # don't normalize when there's no data

        if self.bias_correction:
            bias_correction = 1 - (1 - self.normalize_avg_coef) ** self.timestep_counter
            obs_mean = self.obs_mean / bias_correction
            obs_var = self.obs_var / bias_correction
        else:
            obs_mean = self.obs_mean
            obs_var = self.obs_var
        normalized_obs = (obs - obs_mean) / (np.sqrt(obs_var) + 1e-8)
        normalized_obs = np.clip(normalized_obs, -10, 10)
        return normalized_obs

    def _check_solved_task(self):
        if len(self.eval_success_history) >= 5:
            if np.min(self.eval_success_history[-5:]) >= 0.799:
                return True
        return False


class SuccessCounter(gym.Wrapper):
    """From Continual World's Codebase"""
    def __init__(self, env):
        super().__init__(env)
        self.successes = []
        self.current_success = False

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("success", False):
            self.current_success = True
        if terminated or truncated:
            self.successes.append(self.current_success)
        return obs, reward, terminated, truncated, info

    def pop_successes(self):
        res = self.successes
        self.successes = []
        return res

    def reset(self, **kwargs):
        self.current_success = False
        return self.env.reset(**kwargs)


if __name__ == '__main__':
    ...

    np.set_printoptions(suppress=True)
    env = MetaWorldSingleEnvSequence(1000,'reach-v2', seed=123,
                                     obs_drift_std=0.0, obs_noise_std=0.01, normalize_obs=False)
    obs, _ = env.reset()

    print(obs)
    # for i in range(100):
    #     env.step(action=[0,0,0,0])
    #     print(obs)
    #     # print(env.obs_drift)

    # env.evaluate_agent(None)
    ...
