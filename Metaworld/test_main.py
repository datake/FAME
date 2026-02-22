import argparse
import torch
import time
from agent.sac import SACAgent
from replay_buffer import ReplayBuffer, Collector
import numpy as np
from collections import defaultdict
import pandas as pd
import os
import random
import copy


import sys

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()

def set_seed_everywhere(seed_value):
    seed_value = int(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

class ConfigDictConverter:
    def __init__(self, config_dict):
        '''
        This class takes a config_dict which contains all the variables needed to do one run
        and converts it into the variables needed to run the RL experiments
        We assume that the config file has certain variables and is organized in the proper way
        For the env and agent parameters, we will pass config_dict on to them and assume that they handle it properly
        Note that we *cannot* use the same variable names for env and agent parameters. If the agent or env expect the
        same name, we will need to write it down different in the config file and then convert here

        Attributes:
        agent_dict:
        env_dict:
        '''
        # Improvement: possible to split agent and env parameters here. That is this class contains
        # two dicts for agent_parameters and env_parameters.
        # This would help with passing only the required arguments for envs that are already created (e.g gym)
        # Also, it could help with dealing with parameters that have the same name but are different for agent and env

        self.config_dict = config_dict.copy()

        # training shouldn't need these variables
        if 'num_repeats' in self.config_dict.keys():
            del self.config_dict['num_repeats']
        if 'num_runs_per_group' in self.config_dict.keys():
            del self.config_dict['num_runs_per_group']

        self.agent_dict = self.config_dict.copy()
        self.env_dict = self.config_dict.copy()

        # TODO remove maybe?
        self.repeat_idx = config_dict['repeat_idx']

        # environment
        # reinforcement learning
        import envs.metaworld_env
        self.env_class = envs.metaworld_env.MetaWorldSingleEnvSequence

        env_key_lst = ['env', 'base_task_name', 'seed', 'goal_hidden', 'normalize_obs', 'normalize_rewards',
                       'capture_video', 'save_name', 'change_freq', 'env_sequence',
                       'obs_drift_mean', 'obs_drift_std', 'obs_scale_drift', 'obs_noise_std', 'normalize_avg_coef',
                       'reset_obs_stats', 'change_when_solved']
        env = config_dict['env'].lower()

        if env[0:19] == 'metaworld_sequence_':  # e.g. 'metaworld_sequence_reach'
            if env[19:22] == 'set':  # e.g. "metaworld_sequence_set1"
                self.env_dict['env_sequence'] = env[19:]  # e.g. "set1"
            else:
                self.env_dict['base_task_name'] = f'{env[19:]}-v2'

        self.env_dict = {k: v for k, v in self.env_dict.items() if k in env_key_lst}
        self.env_dict['env_type'] = 'rl'

        print('env_dict keys', self.env_dict.keys())
        print('env_dict', self.env_dict)

        # Other params
        self.agent_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Adjust the seed based on repeat
        self.env_dict['seed'] += self.repeat_idx * 1


def str2none(value):
    if value.lower() in {"none", ""}:
        return None
    return value

def main():
    parser = argparse.ArgumentParser(description='Run RL experiments')
    # General experiment arguments
    parser.add_argument('--repeat_idx', type=int, default=0, help='Index of the repeat (for multiple runs)')
    parser.add_argument('--env', type=str, default='metaworld_sequence_set6', help='Environment to run')
    parser.add_argument('--change_freq', type=int, default=1e6,help='Frequency to change tasks in the environment')  # note this is overriden below per environment
    parser.add_argument("--normalize_obs", type=str2none, default=None, help="Normalize observations (pass 'none' to keep None)")
    parser.add_argument('--seed', type=int, default=0, help='Num steps to run')
    parser.add_argument('--save_path', type=str, default='results/', help='Path to the folder to be saved in')
    parser.add_argument('--save_freq', type=int, default=25000, help='Number steps between recording metrics')
    parser.add_argument('--save_model_freq', type=int, default=-1,help='Number of steps between saving the model. Set to -1 for never. ')
    parser.add_argument('--method', type=str, default='independent', help='Method to use for multitask learning') # 'independent', 'average', 'continue', 'buffer', 'buffer_wd'
    parser.add_argument('--store_traj_num', type=int, default=10, help='Number of trajectories to store in the buffer for each task, only for buffer method')
    parser.add_argument('--use_ttest', type=int, default=0, help='Whether to use t-test for agent selection (0: False, 1: True)')
    parser.add_argument('--gpu', type=str, default='0', help='Comma separated list of GPU IDs')

    # args = parser.parse_args(args=[])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # "0,1"

    config_obj = ConfigDictConverter(vars(args))
    env_parameters = config_obj.env_dict

    env = config_obj.env_class(**env_parameters)
    eval_env = copy.deepcopy(env)
    print('env_list:',env.env_list)
    # print(env_parameters)

    num_steps_per_run = len(env.env_list) * args.change_freq
    # print('num_steps_per_run:',num_steps_per_run, env.normalize_obs)

    num_eval_runs = 10
    set_seed_everywhere(args.seed)

    method = args.method
    log_path = 'log/' + args.env + '/'
    log_name = 'sac_' + args.env + '_' + str(args.seed) + '_' + method

    # sys.stdout = Logger(log_path + log_name + ".txt")
    # sys.stderr = sys.stdout

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.perf_counter()

    # print(env.env.action_space)
    # print(env.env.observation_space)
    # env.env.env.env.env.env.env.env.base_task_name
    # print('---')
    # print(env.env.env.env.env.env.env.action_space)

    random_steps = 10000
    replay_buffer_capacity = args.change_freq

    replay_buffer = ReplayBuffer(
        env.env.observation_space.shape,
        env.env.action_space.shape,
        int(replay_buffer_capacity) + random_steps,
        device)

    agent_list = []
    for i in range(len(env.env_list)):
        agent_list.append(SACAgent(obs_dim=env.env.observation_space.shape[0],
                         action_dim=env.env.action_space.shape[0],
                         action_range=[-1., 1.],
                         device=device, ))

    if method == 'buffer' or method == 'buffer_wd':
        meta_agent_list = []
        for i in range(len(env.env_list)):
            meta_agent_list.append(SACAgent(obs_dim=env.env.observation_space.shape[0],
                                  action_dim=env.env.action_space.shape[0],
                                  action_range=[-1., 1.],
                                  device=device, )) # only the last one will be used finally, save all just for monitoring the training process
        meta_buffer = ReplayBuffer(
            env.env.observation_space.shape,
            env.env.action_space.shape,
            int(replay_buffer_capacity) + random_steps,
            device) # no need this large buffer, just store a few trajectories for each task
        from_meta = False

    collector = Collector(env, replay_buffer)

    obs, _ = env.reset()  # match the gymnasium interface

    task_counter = env.task_counter
    agent = agent_list[task_counter-1] # task_counter starts from 1, agent_list from 0
    if method == 'buffer' or method == 'buffer_wd':
        meta_agent = meta_agent_list[task_counter-1]
    collector.initial_collect(random_steps)

    intermediate_stats = defaultdict(list)
    final_stats = defaultdict(list)

    i_step = 0
    count_success = -1

    while i_step <= num_steps_per_run:
        if task_counter != env.task_counter:
            # reset the replay buffer and agent
            task_counter = env.task_counter
            if method == 'buffer' or method == 'buffer_wd':
                # store the last n trajectories in the buffer
                assert replay_buffer.full == True, f"Replay buffer not full! Current idx: {replay_buffer.idx}"
                # print(replay_buffer.not_dones,len(replay_buffer.not_dones))
                # print(np.where(replay_buffer.not_dones == 0))
                done_indices = np.where(replay_buffer.not_dones == 0)[0][-args.store_traj_num-1:] # get the indices of the last n trajectories, -1 because the last traj may not be a full traj
                # print('done_indices:', done_indices)
                # print('meta_buffer idx:', meta_buffer.idx)
                count_success = 0
                for i in range(len(done_indices)-1): # check if the last n trajectories have success
                    if np.sum(replay_buffer.successes[done_indices[i]+1:done_indices[i+1]+1]) > 0:
                        count_success += 1
                done_ind = done_indices[0]+1 # from a new trajectory to add

                if method == 'buffer_wd': # update the meta agent first, then add the new trajectory
                    if task_counter - 1 - 1 - 1 >= 0:
                        # -1 because task_counter starts from 1, agent_list from 0,
                        # -1 because task_counter = env.task_counter makes it the next task,
                        # -1 because we want to use the previous task's agent
                        meta_agent.actor_wd_loss(agent,meta_agent_list[task_counter-1-1-1],replay_buffer,done_ind, meta_buffer, args.store_traj_num * task_counter * 100)
                    else:
                        meta_agent.actor_wd_loss(agent, None, replay_buffer, done_ind, meta_buffer, args.store_traj_num * task_counter * 100)

                while done_ind < replay_buffer.capacity:
                    meta_buffer.add(replay_buffer.obses[done_ind],
                                    replay_buffer.actions[done_ind],
                                    replay_buffer.rewards[done_ind],
                                    replay_buffer.successes[done_ind],
                                    replay_buffer.next_obses[done_ind],
                                    not replay_buffer.not_dones[done_ind],
                                    not replay_buffer.not_dones_no_max[done_ind])
                    done_ind += 1
                    # print(meta_buffer.idx)
                print('meta_buffer idx:', meta_buffer.idx, 'count_success:', count_success)
                # print(meta_buffer.successes[:meta_buffer.idx], len(meta_buffer.successes[:meta_buffer.idx]))
                # print('done_indices:',np.where(meta_buffer.not_dones[:meta_buffer.idx] == 0)[0])
                # print('success_indices:',np.where(meta_buffer.successes[:meta_buffer.idx] == 1)[0])
                # print('action:',np.max(meta_buffer.actions[:meta_buffer.idx]),np.min(meta_buffer.actions[:meta_buffer.idx]))
                # learn the meta agent
                # todo: how many steps to update
                if method == 'buffer': # update the meta agent affter adding the new trajectory
                    meta_agent.actor_nll(meta_buffer,args.store_traj_num * task_counter * 100)

                # test the meta agent
                eval_env.set_task(env.env_list[(task_counter-1-1)%len(env.env_list)]) # should be task_counter -1-1, last task not the current task

                eval_results = eval_env.evaluate_agent(meta_agent, num_eval_runs)
                eval_episode_returns = eval_results['episodic_returns']
                eval_successes = eval_results['successes']
                print(f"meta_agent: task {eval_env.base_task_name}, success {round(np.mean(eval_successes), 3)} +/- {round(np.std(eval_successes), 3)}, "
                      f"return {round(np.mean(eval_episode_returns), 3)} +/- {round(np.std(eval_episode_returns), 3)}")

                # eval_results = eval_env.evaluate_agent(agent, num_eval_runs)
                # eval_episode_returns = eval_results['episodic_returns']
                # eval_successes = eval_results['successes']
                # print(f"task {eval_env.base_task_name}, success {round(np.mean(eval_successes), 3)} +/- {round(np.std(eval_successes), 3)}, "
                #     f"return {round(np.mean(eval_episode_returns), 3)} +/- {round(np.std(eval_episode_returns), 3)}")

                agent.save('model', log_name + '_' + str(task_counter-1-1))
                meta_agent.save('model', log_name + '_' + str(task_counter-1-1) + '_meta')

            if i_step == num_steps_per_run:
                break

            replay_buffer.reset()

            if method == 'continue':
                agent_list[(task_counter-1)%len(env.env_list)].critic.load_state_dict(agent.critic.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].critic_target.load_state_dict(agent.critic_target.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].actor.load_state_dict(agent.actor.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].log_alpha = agent.log_alpha
                agent_list[(task_counter-1)%len(env.env_list)].critic_optimizer.load_state_dict(agent.critic_optimizer.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].actor_optimizer.load_state_dict(agent.actor_optimizer.state_dict())
                agent_list[(task_counter-1)%len(env.env_list)].log_alpha_optimizer.load_state_dict(agent.log_alpha_optimizer.state_dict())
            # reset the agent
            if method == 'buffer' or method == 'buffer_wd':
                agent = agent_list[(task_counter - 1) % len(env.env_list)]
                return_dict, success_dict = collector.initial_agent_collect(random_steps, [agent,meta_agent], num_eval_runs)
                
                if args.use_ttest:
                    from scipy import stats
                    # H0: meta_agent <= agent (mean return)
                    # H1: meta_agent > agent (mean return)
                    # We use a one-sided t-test. If p-value < 0.05, we reject H0 and conclude meta_agent is better.
                    t_stat, p_value = stats.ttest_ind(return_dict[1], return_dict[0], alternative='greater')
                    print(f'T-test: t_stat={t_stat:.3f}, p_value={p_value:.3f}')
                    if p_value < 0.05:
                        agent.actor.load_state_dict(meta_agent.actor.state_dict())
                        from_meta = True
                    else:
                        from_meta = False
                else:
                    if np.mean(return_dict[1]) > np.mean(return_dict[0]):
                        agent.actor.load_state_dict(meta_agent.actor.state_dict())
                        from_meta = True
                    else:
                        from_meta = False
                print('return_dict: {:.3f} {:.3f}, success_dict: {:.3f} {:.3f}, from_meta: {}'.format(np.mean(return_dict[0]),np.mean(return_dict[1]),np.mean(success_dict[0]),np.mean(success_dict[1]), from_meta))
                # copy the meta agent to the current agent
                meta_agent_list[(task_counter-1)%len(env.env_list)].actor.load_state_dict(meta_agent.actor.state_dict())
                meta_agent_list[(task_counter-1)%len(env.env_list)].actor_optimizer.load_state_dict(meta_agent.actor_optimizer.state_dict())
                meta_agent = meta_agent_list[task_counter-1]
            else:
                agent = agent_list[(task_counter-1)%len(env.env_list)]
                collector.initial_collect(random_steps) # random steps for the new task
        # interact with the environment and update the agent
        if method == 'independent' or method == 'continue' or method == 'buffer' or method == 'buffer_wd':
            agent.sac_update(replay_buffer,i_step)
        elif method == 'average': # alpha = 1/agent_num
            obs, action, reward, success, next_obs, not_done_no_max = replay_buffer.sample(agent.batch_size)
            obs, action, reward, success, next_obs, not_done_no_max = replay_buffer.as_torch(obs, action, reward, success, next_obs, not_done_no_max)
            if task_counter >= len(env.env_list):
                agent_num = len(env.env_list)
            else:
                agent_num = task_counter
            q_target = agent.compute_target_q(reward, next_obs, not_done_no_max)
            q_target /= agent_num
            for agent_idx in range(agent_num-1):
                q_target += agent_list[agent_idx].compute_target_q(reward, next_obs, not_done_no_max) / agent_num
            agent.update_with_target_q(obs, action, q_target, i_step)
        collector.run_one_step(i_step, agent)
        # evaluate the agent in the current task, to plot the learning curve
        if i_step % args.save_freq == 0:
            print('step:',i_step, 'time:', round((time.perf_counter() - start_time) / 60, 3), "SPS:", int(i_step / (time.perf_counter() - start_time)),
                  'task_counter', (task_counter,env.base_task_name), 'agent', (task_counter-1)%len(env.env_list), 'method', method, 'seed', args.seed)

            eval_results = env.evaluate_agent(agent, num_eval_runs)
            eval_episode_returns = eval_results['episodic_returns']
            eval_successes = eval_results['successes']

            intermediate_stats['mean_return'].append(np.mean(eval_episode_returns))
            intermediate_stats['mean_success'].append(np.mean(eval_successes))
            intermediate_stats['steps'].append(i_step)
            intermediate_stats['task'].append(env.base_task_name)
            intermediate_stats['seed'].append(args.seed)
            intermediate_stats['task_idx'].append(task_counter)
            intermediate_stats['method'].append(method)
            intermediate_stats['time'].append(round((time.perf_counter() - start_time) / 3600, 3)) # in hours
            if method == 'buffer' or method == 'buffer_wd':
                intermediate_stats['from_meta'].append(from_meta)
            if -1 in intermediate_stats['count_success']:
                intermediate_stats['count_success'][intermediate_stats['count_success'].index(-1)] = count_success
            else:
                intermediate_stats['count_success'].append(count_success)

            print(f"success {round(np.mean(eval_successes), 3)} +/- {round(np.std(eval_successes), 3)},",
                  f"eval return {round(np.mean(eval_episode_returns), 3)} +/- {round(np.std(eval_episode_returns), 3)}")
        i_step += 1

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    print('len:',len(intermediate_stats['count_success']), len(intermediate_stats['mean_return']), len(intermediate_stats['mean_success']))
    intermediate_stats['count_success'].extend([count_success] * (len(intermediate_stats['mean_return']) - len(intermediate_stats['count_success'])))
    intermediate_stats = pd.DataFrame(intermediate_stats)
    intermediate_stats.to_csv(log_path + "/" + log_name + ".csv", index=False)
    # evaluate all agents on all previous tasks
    print('---')
    if method == 'buffer' or method == 'buffer_wd':
        eval_agent_list = meta_agent_list
    else:
        eval_agent_list = agent_list
    for agent_idx, agent in enumerate(eval_agent_list):
        for i in range(len(env.env_list[:agent_idx+1])):
            env.set_task(env.env_list[i])
            eval_results = env.evaluate_agent(agent, num_eval_runs)
            eval_episode_returns = eval_results['episodic_returns']
            eval_successes = eval_results['successes']

            # print(f"Final task {env.env_list[i]} success {round(np.mean(eval_successes), 3)} +/- {round(np.std(eval_successes), 3)}")
            print(f"Final task {env.env_list[i]} success {round(np.mean(eval_successes), 3)} return {round(np.mean(eval_episode_returns), 3)}")

            final_stats['mean_return'].append(np.mean(eval_episode_returns))
            final_stats['mean_success'].append(np.mean(eval_successes))
            final_stats['task'].append(env.base_task_name)
            final_stats['task_idx'].append(i + 1)
            final_stats['seed'].append(args.seed)
            final_stats['method'].append(method)
            final_stats['agent_idx'].append(agent_idx+1)

    # evaluate the meta agent on all tasks
    # if method == 'buffer':
    #     agent = meta_agent
    #     for i in range(len(env.env_list)):
    #         env.set_task(env.env_list[i])
    #         eval_results = env.evaluate_agent(agent, num_eval_runs)
    #         eval_episode_returns = eval_results['episodic_returns']
    #         eval_successes = eval_results['successes']
    #
    #         print(f"Final task {env.env_list[i]} success {round(np.mean(eval_successes), 3)} +/- {round(np.std(eval_successes), 3)}")
    #
    #         final_stats['mean_return'].append(np.mean(eval_episode_returns))
    #         final_stats['mean_success'].append(np.mean(eval_successes))
    #         final_stats['task'].append(env.base_task_name)
    #         final_stats['task_idx'].append(i+1)
    #         final_stats['seed'].append(args.seed)
    #         final_stats['method'].append(method)
    #         final_stats['agent_idx'].append(-1) # meta agent

    final_stats = pd.DataFrame(final_stats)
    final_stats.to_csv(log_path + "/" + log_name + "_final.csv", index=False)

'''
python test_main.py --seed 0 --method independent --gpu 2 --env metaworld_sequence_set21
'''

if __name__ == "__main__":
    main()