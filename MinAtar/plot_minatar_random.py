import time

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import matplotlib.backends.backend_pdf
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from tqdm import tqdm
from CL_envs import *
import torch
from model import *
import pandas as pd
torch.set_printoptions(linewidth=200)
np.set_printoptions(linewidth=200)
pd.options.display.float_format = "{:.2f}".format
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)
matplotlib.use('TkAgg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

np.seterr(invalid='ignore')
plt.style.use('seaborn-v0_8-white')

benchmark_DQN = [ # 60000 steps, each result is very stable; use the first one under seed 1 is fine
    {'breakout': 12.7, 'space_invaders': 30.9, 'freeway': 3.04}, # seq:0
    {'breakout': 0.0, 'space_invaders': 0.0, 'freeway': 0.0}, # seq:1
    {'breakout': 0.0, 'space_invaders': 0.0, 'freeway': 0.0}, # seq:2
    {'breakout': 12.5, 'space_invaders': 30.8, 'freeway': 2.125}, # seq:3
    {'breakout': 0.0, 'space_invaders': 0.0, 'freeway': 0.0}, # seq:4
    {'breakout': 0.0, 'space_invaders': 0.0, 'freeway': 0.0}, # seq:5
    {'breakout': 0.0, 'space_invaders': 0.0, 'freeway': 0.0}, # seq:6
    {'breakout': 0.0, 'space_invaders': 0.0, 'freeway': 0.0}, # seq:7
    {'breakout': 0.0, 'space_invaders': 0.0, 'freeway': 0.0}, # seq:8
    {'breakout': 0.0, 'space_invaders': 0.0, 'freeway': 0.0}, # seq:9
]
env_name = "all"
seeds = [1, 2, 3]
seq_list = range(10) # [seq]
# seq_list = [4] # [seq]
# seq = 9
t_steps = 3500000
plot_steps = 3500000
gamma = 0.99
update = 50000
switch = 500000
t_seeds = len(seeds) * len(seq_list)

AveragePerformance = False # , save the excel;
"""
1. load the model at the time t and evaluate on the three enviroments 
2.together with CF_step_normalized to determined which time of the model to load, then save the excel to plot the overtime figure
3.default: 7 for final ave performance (reported in the table)

"""
CF_step_normalized = 7 # [1, 7]
CF_Evaluation = True # to ,
"""
1.evaluate the forgetting by looping over all the past environments (NOT JUST THREE)
2.can be costly, CF_step_normalized = 7
"""

PLOT = False

ForwardTransfer = False # evaluate on the return data
"""
1. Only evaluate on the return data (results)
2. self-normalization
"""
Evaluation_Step = 6000

Game_normalized_list = ["breakout", "space_invaders", "freeway"]

def AverageSeeds(seeds_returns):
    result_mean = np.zeros((len(seq_list), t_steps))
    for i in range(len(seq_list)):
        result_mean[i] =  np.mean(seeds_returns[i * len(seeds): (i + 1) * len(seeds)], axis=0)
    return result_mean


if ForwardTransfer:
    Returns = []


def moving_average(a, n=3):
    cumsum_vec = np.cumsum(np.insert(a, 0, 0))
    ma_vec = (cumsum_vec[n:] - cumsum_vec[:-n]) / n
    return np.concatenate((a[0:n-1]/n, ma_vec))


############## get the environments
## Note: CF also depends on the frequency of the past environments
Games = []

for seq_i in range(10):
    game_list = []
    for gameid_i in range(7):
        env = CL_envs_func_replacement(seq=seq_i, game_id=gameid_i, seed=seq_i)
        game_list.append(env)
    Games.append(game_list)

# print('Benchmark Evaluation envs:', Games[seq][CF_step_normalized - 1].game_name)

# normalization by the DQN with train from scratch

def GenerateGames_normalized(seqid, seed):
    Games_normalized = []
    # for seed_i in seeds:
    #     game_list = []
    for gameid_i in range(3):
        env = CL_envs_func_replacement(seq=seqid, game_id=gameid_i, seed=seed, evaluation=True)
        Games_normalized.append(env)
    # Games_normalized.append(game_list) # Games_normalized[seed][game_id]
    return Games_normalized

def GenerateGames_past(seqid, seed):
    Games_list = []
    for gameid_i in range(7):
        env = CL_envs_func_replacement(seq=seqid, game_id=gameid_i, seed=seed)
        Games_list.append(env)
    # Games_normalized.append(game_list) # Games_normalized[seed][game_id]
    return Games_list

def AveragePerformance_Evaluation(filename, seqid, seed, modeltype='DQN'):

    Games_normalized = GenerateGames_normalized(seqid, seed) # Games_normalized[seed][game_id]
    #### evaluation for the i-th step

    """
        modeltype: DQN, DQN_Finetune, PT-DQN, Ours, Multitask, LargeBuffer
        """
    # Episode_evaluation = 0 # need to be changed
    env_initial = Games_normalized[0]
    # print(env_initial.game_name)
    in_channels = env_initial.observation_space.shape[2]  # [10, 10, 7]
    num_actions = env_initial.action_space.n
    if modeltype in ['PT-DQN']:
        model = CNN_half(in_channels, num_actions)
    elif modeltype == 'Multitask':
        model = CNN_three_heads(in_channels, num_actions)
    else:
        model = CNN(in_channels, num_actions)
    CF_list = []


    # load the model


    if modeltype == 'Ours':
        model.load_state_dict(torch.load(f"./models/RandomEnvs/seq{seq}/" + filename + "_Meta" + str(CF_step_normalized - 1) + ".pt",
                                         map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(f"./models/RandomEnvs/seq{seq}/" + filename + "_Net" + str(CF_step_normalized - 1) + ".pt",
                                         map_location=torch.device('cpu')))
    for j in range(3):  # loop the evaluation game
        env = Games_normalized[j]
        print('Evaluation env:', env.game_name)
        task_id = envs_to_id[env.game_name]

        cs = env.reset()
        # evaluate the average return within 100 steps
        sumreward = []
        epi_return = 0
        epi_count = 0
        max_step = 0
        Max_step = 300
        for step_small in range(Evaluation_Step):
            cs = np.moveaxis(cs, 2, 0)
            cs = torch.tensor(cs, dtype=torch.float)
            with torch.no_grad():
                if modeltype == 'Multitask':
                    curr_Q_vals = model(cs.unsqueeze(0))[task_id]
                else:
                    curr_Q_vals = model(cs.unsqueeze(0))
            c_action = curr_Q_vals.max(1)[1].item()
            ns, rew, done, _ = env.step(c_action)
            epi_return += rew
            cs = ns
            max_step += 1
            if done or max_step >= Max_step:
                cs = env.reset()
                epi_count += 1
                sumreward.append(epi_return)
                epi_return = 0
                max_step = 0

        if len(sumreward) == 0:
            print("no episode")
        else:
            CF_list.append(np.mean(sumreward))

    return CF_list



###### evaluate the castrophic forgetting across the past environments
def CF_evaluation(filename, seq, seed, return_seq, modeltype='DQN'):
    """
    modeltype: DQN, DQN_Finetune, PT-DQN, Ours, Multitask, LargeBuffer
    """
    # Episode_evaluation = 0 # need to be changed

    Games_past = GenerateGames_past(seq, seed) # Games_past[game_id]

    env_initial = Games_past[0]
    print(env_initial.game_name)
    in_channels = env_initial.observation_space.shape[2]  # [10, 10, 7]
    num_actions = env_initial.action_space.n
    if modeltype in ['PT-DQN']:
        model = CNN_half(in_channels, num_actions)
    elif modeltype == 'Multitask':
        model = CNN_three_heads(in_channels, num_actions)
    else:
        model = CNN(in_channels, num_actions)
    CF_list = []

    # print(f'Evaluate CF the time {i}')
    # average_CF = [] # record the average return in the past environments
    # load the model

    if modeltype == 'Ours':
        model.load_state_dict(torch.load(f"./models/RandomEnvs/seq{seq}/" + filename + "_Meta" + str(CF_step_normalized-1) + ".pt", map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(f"./models/RandomEnvs/seq{seq}/" + filename + "_Net" + str(CF_step_normalized-1) + ".pt", map_location=torch.device('cpu')))
    for j in range(CF_step_normalized): # loop the evaluation game
        # print(j)
        env = Games_past[j]
        # print('Evaluation env:', env.game_name)
        task_id = envs_to_id[env.game_name]

        # calculate pk(kdelta)
        envstep_size = int(len(return_seq) / len(Games_past))
        return_kdelta = return_seq[(j+1) * envstep_size - 1]


        cs = env.reset()
        # evaluate the average return within 100 steps
        sumreward = []
        epi_return = 0
        epi_count = 0
        max_step = 0
        Max_step = 300
        for step_small in range(Evaluation_Step):
            cs = np.moveaxis(cs, 2, 0)
            cs = torch.tensor(cs, dtype=torch.float)
            with torch.no_grad():
                if modeltype == 'Multitask':
                    curr_Q_vals = model(cs.unsqueeze(0))[task_id]
                else:
                    curr_Q_vals = model(cs.unsqueeze(0))
            c_action = curr_Q_vals.max(1)[1].item()
            ns, rew, done, _ = env.step(c_action)
            epi_return += rew
            cs = ns
            max_step += 1
            if done or max_step >= Max_step:
                cs = env.reset()
                epi_count += 1
                sumreward.append(epi_return)
                epi_return = 0
                max_step = 0

        if len(sumreward) == 0:
            print("no episode")
        else:
            # average_CF.append(np.mean(sumreward))
            CF_list.append(return_kdelta - np.mean(sumreward)) # store the average performance by averaging average_CF among all past environments

    temp_df = pd.DataFrame({'CF': CF_list, 'Env': [env.game_name for env in Games_past]})
    temp_df = temp_df.groupby('Env')['CF'].mean().reset_index()
    average_CF = []
    for envname in Game_normalized_list:
        if len(temp_df[temp_df['Env'] == envname]['CF']) != 0:
            average_CF.append(float(temp_df[temp_df['Env'] == envname]['CF']))
        else:
            average_CF.append(0.0)
    return average_CF

envs_to_id = {"breakout":0, "space_invaders":1, "freeway":2}

smooth_coeff = 10000
z_star = 1.3

fig, ax = plt.subplots(figsize=(12,5))
ax.set_rasterized(True)



script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))

## DQN ###
best_lr_dqn = 1e-5
seeds_returns = np.zeros((t_seeds, t_steps))
reset = 1 # reset, 0: no reset, 1: reset with , 2: reset without setting target net
clearbuffer = 1


DQN_CF = []
DQN_CF_Normalized = []

for seq in seq_list:
    for s in tqdm(seeds):
        # if len(seeds) == 1:
        #     s = seq
        fname = "DQN"+"_env_name_"+env_name+"_gamma_"+str(gamma)+\
                "_steps_"+str(t_steps)+"_switch_"+str(switch)+"_batch_"+\
    str(64)+"_lr1_"+str(best_lr_dqn)+ '_seq_' + str(seq) + "_reset_" +str(reset)+ "_clearbuffer_" + str(clearbuffer) +"_seed_"+str(s)
        with open("results/RandomEnvs/" +fname +"_returns.pkl", "rb") as f:

            return_seq = moving_average(pickle.load(f), n=smooth_coeff)
            if len(seeds) == 1:
                seeds_returns[0] = return_seq
            else:
                if len(seq_list) == 1:
                    seeds_returns[s - 1] = return_seq
                else:
                    seeds_returns[len(seeds) * seq + s-1] = return_seq

        if AveragePerformance:
            DQN_CF_Normalized.append(AveragePerformance_Evaluation(fname,seq, s,modeltype='DQN'))
            print('Finish the DQN evaluation')
        if CF_Evaluation:
            DQN_CF.append(CF_evaluation(fname, seq, s, return_seq, modeltype='DQN'))
            print('Finish the DQN evaluation')

if ForwardTransfer:
    Returns.append(AverageSeeds(seeds_returns))




if AveragePerformance:
    DQN_CF_Normalized_mean = np.array(DQN_CF_Normalized).mean(axis=0) # averaege over seed
    DQN_CF_Normalized_std = np.array(DQN_CF_Normalized).std(axis=0) / np.sqrt(len(seq_list) * len(seeds)) # averaege over seed
    print('DQN Average Performance / Final CF:', DQN_CF_Normalized_mean)
    print('DQN Benchmark Performance:', DQN_CF_Normalized_mean)

if CF_Evaluation:
    DQN_CF = np.array(DQN_CF).mean(axis=0) # averaege over seed
    print('DQN CF:', DQN_CF)



rew_mean = np.mean(seeds_returns, axis=0)
rew_std = np.std(seeds_returns, axis=0)
ax.plot(rew_mean, label="DQN", lw=1.0, color="green", alpha=0.75)
ax.fill_between(range(t_steps), rew_mean+z_star*(rew_std/t_seeds**0.5), rew_mean-z_star*(rew_std/t_seeds**0.5), alpha=0.2, color="green")

## DQN_finetune ###
best_lr_dqn = 1e-5
seeds_returns = np.zeros((t_seeds, t_steps))
reset = 0 # reset, 0: no reset, 1: reset with , 2: reset without setting target net
clearbuffer = 1


DQNFT_CF = []
DQNFT_CF_Normalized = []

for seq in seq_list:
    for s in tqdm(seeds):
        # if len(seeds) == 1:
        #     s = seq
        fname = "DQN"+"_env_name_"+env_name+"_gamma_"+str(gamma)+\
                "_steps_"+str(t_steps)+"_switch_"+str(switch)+"_batch_"+\
    str(64)+"_lr1_"+str(best_lr_dqn)+ '_seq_' + str(seq) + "_reset_" +str(reset)+ "_clearbuffer_" + str(clearbuffer) +"_seed_"+str(s)
        with open("results/RandomEnvs/" +fname +"_returns.pkl", "rb") as f:
            return_seq = moving_average(pickle.load(f), n=smooth_coeff)
            if len(seeds) == 1:
                seeds_returns[0] = return_seq
            else:
                if len(seq_list) == 1:
                    seeds_returns[s - 1] = return_seq
                else:
                    seeds_returns[len(seeds) * seq + s - 1] = return_seq

        if AveragePerformance:
            DQNFT_CF_Normalized.append(AveragePerformance_Evaluation(fname, seq, s, modeltype='DQN_Finetune'))
            print('Finish the DQN_finetune evaluation')

        if CF_Evaluation:
            DQNFT_CF.append(CF_evaluation(fname, seq, s, return_seq, modeltype='DQN_Finetune'))
            print('Finish the DQN_Finetune evaluation')

if ForwardTransfer:
    Returns.append(AverageSeeds(seeds_returns))

if AveragePerformance:
    DQNFT_CF_Normalized_mean = np.array(DQNFT_CF_Normalized).mean(axis=0) # averaege over seed
    DQNFT_CF_Normalized_std = np.array(DQNFT_CF_Normalized).std(axis=0) / np.sqrt(len(seq_list) * len(seeds)) # averaege over seed
    print('DQN Finetune Average Performance / Final CF:', DQNFT_CF_Normalized_mean)

if CF_Evaluation:
    DQNFT_CF = np.array(DQNFT_CF).mean(axis=0) # averaege over seed
    print('DQN Finetune CF:', DQNFT_CF)

rew_mean = np.mean(seeds_returns, axis=0)
rew_std = np.std(seeds_returns, axis=0)
ax.plot(rew_mean, label="DQN-Finetune", lw=1.0, color="pink", alpha=0.75)
ax.fill_between(range(t_steps), rew_mean+z_star*(rew_std/t_seeds**0.5), rew_mean-z_star*(rew_std/t_seeds**0.5), alpha=0.2, color="pink")



# ### Our Method ###
best_dec = 0.75
best_lr1 = 1e-3
best_lr2 = 1e-5
size_fast2meta = 12000 # 12000
detection_step = 5000
finetunefast = 1 # 0: meta vs random, 1: meta vs fast
CNNhalf = 0
clearbuffer = 1
epoch_meta = 100
reset = 1
Qnormalization = 0
policy = 1
policyloss = 0
p_explore=0.0
warmstep= 50000
epoch_meta2fast = 0
lambda_reg = 1.0
seeds_returns = np.zeros((t_seeds, t_steps))

Our_CF = []
Our_CF_Normalized = []

for seq in seq_list:
    for s in tqdm(seeds):
        fname = ("FAME" +  "_steps_" + str(t_steps) + "_switch_" + str(
            switch) + "_update_" + str(update) +  "_lr1_" + str(best_lr1) + "_lr2_" + str(best_lr2) + "_size_fast2meta_" + str(
            size_fast2meta) + "_detection_step_" + str(detection_step) + '_seq_' + str(seq) + '_CNNhalf_' + str(CNNhalf)
                  + "_epoch_meta_" + str(epoch_meta) + "_warmstep_" + str(warmstep) + '_epoch_meta2fast_' + str(epoch_meta2fast) + "_lambda_reg_" + str(lambda_reg)
                 +"_seed_" + str(s))

        with open("results/RandomEnvs/"+fname+"_returns.pkl", "rb") as f:
            return_seq = moving_average(pickle.load(f), n=smooth_coeff)
            if len(seeds) == 1:
                seeds_returns[0] = return_seq
            else:
                if len(seq_list) == 1:
                    seeds_returns[s - 1] = return_seq
                else:
                    seeds_returns[len(seeds) * seq + s - 1] = return_seq

        if AveragePerformance:
            Our_CF_Normalized.append(AveragePerformance_Evaluation(fname, seq, s, modeltype='Ours'))
            print('Finish the CF evaluation')

        if CF_Evaluation:
            Our_CF.append(CF_evaluation(fname, seq, s, return_seq, modeltype='Ours'))
            print('Finish the CF evaluation')

if ForwardTransfer:
    Returns.append(AverageSeeds(seeds_returns))

if AveragePerformance:
    Our_CF_Normalized_mean = np.array(Our_CF_Normalized).mean(axis=0) # averaege over seed
    Our_CF_Normalized_std = np.array(Our_CF_Normalized).std(axis=0) / np.sqrt(len(seq_list) * len(seeds)) # averaege over seed
    print('Ours Average Performance / Final CF:', Our_CF_Normalized_mean)

if CF_Evaluation:
    Our_CF_mean = np.array(Our_CF).mean(axis=0) # averaege over seed
    Our_CF_se = np.array(Our_CF).std(axis=0) / np.sqrt(len(seq_list) * len(seeds)) # averaege over seed
    print('Ours CF:', Our_CF)

rew_mean = np.mean(seeds_returns, axis=0)
rew_std = np.std(seeds_returns, axis=0)
ax.plot(rew_mean, label="Our", lw=1.0, color="red", alpha=0.75)
ax.fill_between(range(t_steps), rew_mean+z_star*(rew_std/t_seeds**0.5), rew_mean-z_star*(rew_std/t_seeds**0.5), alpha=0.2, color="red")

# ### PT-DQN ###
best_dec = 0.75
best_lr1 = 1e-8
best_lr2 = 1e-4
CNNhalf = 1
boundary = 0
seeds_returns = np.zeros((t_seeds, t_steps))
clearbuffer = 1
reset = 1

PTDQN_CF = []
PTDQN_CF_Normalized = []

for seq in seq_list:
    for s in tqdm(seeds):
        fname = ("PT_DQN_0.5x" + "_env_name_" + env_name + "_gamma_" + str(gamma) + \
                "_steps_" + str(t_steps) + "_switch_" + str(switch) + "_update_" + str(update) + "_decay_" + str(best_dec) + \
                "_lr1_" + str(best_lr1) + "_lr2_" + str(best_lr2) + "_batch_" + str(64) + '_seq_' + str(seq) + "_CNNhalf_" + str(CNNhalf)
                 + "_boundary" + str(boundary) + '_reset_'+ str(reset) + "_clearbuffer_" + str(clearbuffer) + "_seed_" + str(s))
        with open("results/RandomEnvs/" +fname+"_returns.pkl", "rb") as f:
            return_seq = moving_average(pickle.load(f), n=smooth_coeff)
            if len(seeds) == 1:
                seeds_returns[0] = return_seq
            else:
                if len(seq_list) == 1:
                    seeds_returns[s - 1] = return_seq
                else:
                    seeds_returns[len(seeds) * seq + s - 1] = return_seq

        if AveragePerformance:
            PTDQN_CF_Normalized.append(AveragePerformance_Evaluation(fname, seq, s, modeltype='PT-DQN'))
            print('Finish the PT-DQN evaluation')

        if CF_Evaluation:
            PTDQN_CF.append(CF_evaluation(fname, seq, s, return_seq,modeltype='PT-DQN'))
            print('Finish the PT-DQN evaluation')

if ForwardTransfer:
    Returns.append(AverageSeeds(seeds_returns))

if AveragePerformance:
    PTDQN_CF_Normalized_mean = np.array(PTDQN_CF_Normalized).mean(axis=0) # averaege over seed
    PTDQN_CF_Normalized_std = np.array(PTDQN_CF_Normalized).std(axis=0) / np.sqrt(len(seq_list) * len(seeds)) # averaege over seed
    print('PT-DQN Average Performance / Final CF:', PTDQN_CF_Normalized_mean)

if CF_Evaluation:
    PTDQN_CF = np.array(PTDQN_CF).mean(axis=0) # averaege over seed
    print('PT-DQN CF:', PTDQN_CF)

rew_mean = np.mean(seeds_returns, axis=0)
rew_std = np.std(seeds_returns, axis=0)
ax.plot(rew_mean, label="PT-DQN-half", lw=1.0, color="blue", alpha=0.75)
ax.fill_between(range(t_steps), rew_mean+z_star*(rew_std/t_seeds**0.5), rew_mean-z_star*(rew_std/t_seeds**0.5), alpha=0.2, color="blue")
#
#
### DQN multi task ###
best_lr_dqn_mt = 1e-5
seeds_returns = np.zeros((t_seeds, t_steps))
clearbuffer = 1
reset = 1

MultiTask_CF = []
MultiTask_CF_Normalized = []

for seq in seq_list:
    for s in tqdm(seeds):
        fname = "DQN_multi_task" + "_env_name_" + env_name + "_gamma_" + str(gamma) + \
                "_steps_" + str(t_steps) + "_switch_" + str(switch) + "_batch_" + str(64) + \
                "_lr1_" + str(best_lr_dqn_mt) +  '_seq_' + str(seq) + "_reset_" +str(reset) +  "_clearbuffer_" + str(clearbuffer) + "_seed_" + str(s)
        with open("results/RandomEnvs/"+fname+"_returns.pkl", "rb") as f:
            return_seq = moving_average(pickle.load(f), n=smooth_coeff)
            if len(seeds) == 1:
                seeds_returns[0] = return_seq
            else:
                if len(seq_list) == 1:
                    seeds_returns[s - 1] = return_seq
                else:
                    seeds_returns[len(seeds) * seq + s - 1] = return_seq

        if AveragePerformance:
            MultiTask_CF_Normalized.append(AveragePerformance_Evaluation(fname, seq, s, modeltype='Multitask'))
            print('Finish the DQN multi-task evaluation')

        if CF_Evaluation:
            MultiTask_CF.append(CF_evaluation(fname, seq, s, return_seq, modeltype='Multitask'))
            print('Finish the DQN multi-task evaluation')

if ForwardTransfer:
    Returns.append(AverageSeeds(seeds_returns))

if AveragePerformance:
    MultiTask_CF_Normalized_mean = np.array(MultiTask_CF_Normalized).mean(axis=0) # averaege over seed
    MultiTask_CF_Normalized_std = np.array(MultiTask_CF_Normalized).std(axis=0) / np.sqrt(len(seq_list) * len(seeds)) # averaege over seed
    print('Multi-task Average Performance / Final CF:', MultiTask_CF_Normalized_mean)

if CF_Evaluation:
    MultiTask_CF = np.array(MultiTask_CF).mean(axis=0) # averaege over seed
    print('Multi-task CF:', MultiTask_CF)

rew_mean = np.mean(seeds_returns, axis=0)
rew_std = np.std(seeds_returns, axis=0)
ax.plot(rew_mean, label="DQN (multi-task)", lw=1.0, color="brown", alpha=0.75)
ax.fill_between(range(t_steps), rew_mean+z_star*(rew_std/t_seeds**0.5), rew_mean-z_star*(rew_std/t_seeds**0.5), alpha=0.2, color="brown")

### DQN large buffer ###
best_lrs_dqn_large = 1e-4
seeds_returns = np.zeros((t_seeds, t_steps))
clearbuffer = 1
LargeBuffer_CF = []
LargeBuffer_CF_Normalized = []
reset = 1

for seq in seq_list:
    for s in tqdm(seeds):
        fname = "DQN_large_buffer" + "_env_name_" + env_name + "_gamma_" + str(gamma) + \
                "_steps_" + str(t_steps) + "_switch_" + str(switch) + "_batch_" + str(64) + \
                "_lr1_" + str(best_lrs_dqn_large) + '_seq_' + str(seq) + "_reset_" +str(reset)+  "_clearbuffer_" + str(clearbuffer) + "_seed_" + str(s)

        with open("results/RandomEnvs/"+fname+"_returns.pkl", "rb") as f:
            return_seq = moving_average(pickle.load(f), n=smooth_coeff)
            if len(seeds) == 1:
                seeds_returns[0] = return_seq
            else:
                if len(seq_list) == 1:
                    seeds_returns[s - 1] = return_seq
                else:
                    seeds_returns[len(seeds) * seq + s - 1] = return_seq

        if AveragePerformance:
            LargeBuffer_CF_Normalized.append(AveragePerformance_Evaluation(fname, seq, s, modeltype='LargeBuffer'))
            print('Finish the DQN large buffer evaluation')

        if CF_Evaluation:
            LargeBuffer_CF.append(CF_evaluation(fname, seq, s, return_seq, modeltype='LargeBuffer'))
            print('Finish the DQN large buffer evaluation')

if ForwardTransfer:
    Returns.append(AverageSeeds(seeds_returns))

if AveragePerformance:
    LargeBuffer_CF_Normalized_mean = np.array(LargeBuffer_CF_Normalized).mean(axis=0) # averaege over seed
    LargeBuffer_CF_Normalized_std = np.array(LargeBuffer_CF_Normalized).std(axis=0) / np.sqrt(len(seq_list) * len(seeds)) # averaege over seed
    print('DQN large buffer Average Performance / Final CF:', LargeBuffer_CF_Normalized_mean)

if CF_Evaluation:
    LargeBuffer_CF = np.array(LargeBuffer_CF).mean(axis=0) # averaege over seed
    print('DQN large buffer CF:', LargeBuffer_CF)


rew_mean = np.mean(seeds_returns, axis=0)
rew_std = np.std(seeds_returns, axis=0)
ax.plot(rew_mean, label="DQN (large buffer)", lw=1.0, color="black", alpha=0.75)
ax.fill_between(range(t_steps), rew_mean+z_star*(rew_std/t_seeds**0.5), rew_mean-z_star*(rew_std/t_seeds**0.5), alpha=0.2, color="black")

### Random ###

seeds_returns = np.zeros((t_seeds, t_steps))
clearbuffer = 1
reset = 1

for seq in seq_list:
    for s in seeds:
        fname = ("Random_"+str(t_steps)+"_switch_"+str(switch)+"_env_name_"+str(env_name)+'_seq_' + str(seq) + "_reset_" +str(reset)
                 + "_clearbuffer_" + str(clearbuffer) + "_seed_"+str(s))
        with open("results/RandomEnvs/"+fname+".pkl", "rb") as f:
            return_seq = moving_average(pickle.load(f), n=smooth_coeff)
            if len(seeds) == 1:
                seeds_returns[0] = return_seq
            else:
                if len(seq_list) == 1:
                    seeds_returns[s - 1] = return_seq
                else:
                    seeds_returns[len(seeds) * seq + s - 1] = return_seq

print(np.mean(seeds_returns, axis=0).sum())
rew_mean = np.mean(seeds_returns, axis=0)
rew_std = np.std(seeds_returns, axis=0)
ax.plot(rew_mean[:plot_steps], label="Random", lw=1.0, color="orange", alpha=0.75)
ax.fill_between(range(plot_steps), rew_mean[:plot_steps]+z_star*(rew_std[:plot_steps]/t_seeds**0.5), rew_mean[:plot_steps]-z_star*(rew_std[:plot_steps]/t_seeds**0.5), alpha=0.2, color="orange")


########### for evaluation of catastrophic forgetting

if ForwardTransfer:
    # result_benchmark = [] # [10, 7]
    # for seq_i in range(len(seq_list)):
    #     benchmark = [benchmark_DQN[0][env.game_name] for env in Games[seq_i]]
    #     result_benchmark.append(benchmark)
    # result_benchmark = np.array(result_benchmark)

    # select the max return among all method in each enviroment
    result_benchmark = []
    for j in range(7):
        target_return = []
        for i, method_transfer in enumerate(Returns):
            target_return.append(method_transfer[:,j*switch:(j+1)*switch].max(axis=1))
        target_return = np.array(target_return).T.max(axis=1) # [10, 6] -> [10,1]
        result_benchmark.append(target_return)
    result_benchmark = np.array(result_benchmark).T # [10, 7]

    # df_transfer = pd.DataFrame()
    row_names = ["DQN-Reset", 'DQN-Finetune', "Ours","PT-DQN-0.5x", "DQN-MultiHead", "DQN-LargeBuffer",  ]
    # row_names = [ "Ours" ]
    transfer_list_mean = []
    transfer_list_se = []
    for i, method_transfer in enumerate(Returns):
        diff = method_transfer - Returns[0] # Reset
        ave_temp = []
        for j in range(7):
            numerator = diff[:,j*switch:(j+1)*switch].mean(axis=1)/result_benchmark[:,j]
            denominator = Returns[0][:,j*switch:(j+1)*switch].mean(axis=1)/result_benchmark[:,j]
            ave_temp.append(numerator/ (1 - denominator))
        transfer_list_mean.append(np.array(ave_temp).mean()) # across seqs and seeds [10, 7] -> 1
        transfer_list_se.append(np.array(ave_temp).std() / np.sqrt(len(seq_list)*7)) # across seqs and seeds [10, 7] -> 1
    df_transfer_mean = pd.DataFrame(transfer_list_mean, index=row_names, columns=['Forward Transfer (mean)'])
    df_transfer_se = pd.DataFrame(transfer_list_se, index=row_names, columns=['Forward Transfer (se)' ])
    print(df_transfer_mean)
    print('The SE!')
    print(df_transfer_se)


if AveragePerformance:
    print('mean: ', [round(i, 2) for i in Our_CF_Normalized_mean])
    # print(np.mean(Our_CF_Normalized_mean))
    print('std: ', [round(i, 2) for i in Our_CF_Normalized_std])


    # CF_list = np.vstack((DQN_CF_Normalized_mean, DQNFT_CF_Normalized_mean, MultiTask_CF_Normalized_mean, LargeBuffer_CF_Normalized_mean, PTDQN_CF_Normalized_mean, Our_CF_Normalized_mean))
    # row_names = ["DQN-Reset", 'DQN-Finetune',   "DQN-MultiHead", "DQN-LargeBuffer", "PT-DQN-0.5x", "Ours"]
    # col_names = Game_normalized_list
    # col_names_normalized = [env+"_normalized" for env in col_names]
    # df = pd.DataFrame(CF_list, index=row_names, columns=col_names)
    # df['Average'] = df.mean(axis=1)
    
    # for i, col in enumerate(col_names):
    #     df[col_names_normalized[i]] = df[col] / benchmark_DQN[0][col] # we fix the seq=0, which can be ajusted for each seed in the future
    # df['Average_normalized (%)'] = df[col_names_normalized].mean(axis=1) * 100.0
    # print(benchmark_DQN[0])
    # print(df)
    
    # CF_list_std = np.vstack((DQN_CF_Normalized_std, DQNFT_CF_Normalized_std, MultiTask_CF_Normalized_std, LargeBuffer_CF_Normalized_std, PTDQN_CF_Normalized_std, Our_CF_Normalized_std))
    # df_std = pd.DataFrame(CF_list_std, index=row_names, columns=col_names)
    # print(df_std)

    #################### to plot the average performance over time, need to save and run plot_averageperformance_overtime

    # df.to_excel(f"MinAtar_Ave_T{CF_step_normalized}_mean.xlsx")
    # df_std.to_excel(f"MinAtar_Ave_T{CF_step_normalized}_std.xlsx")

if CF_Evaluation:
    print('mean: ', [round(i, 2) for i in Our_CF_mean])
    # print(np.mean(Our_CF_Normalized_mean))
    print('std: ', [round(i, 2) for i in Our_CF_se])

    # CF_list = np.vstack((DQN_CF_mean, DQNFT_CF_mean, MultiTask_CF_mean, LargeBuffer_CF_mean, PTDQN_CF_mean, Our_CF_mean))
    # row_names = ["DQN-Reset", 'DQN-Finetune',   "DQN-MultiHead", "DQN-LargeBuffer", "PT-DQN-0.5x", "Ours"]
    # col_names = Game_normalized_list
    # col_names_normalized = [env + "_normalized" for env in col_names]
    # df = pd.DataFrame(CF_list, index=row_names, columns=col_names)
    # df['Average'] = df.mean(axis=1)
    
    # for i, col in enumerate(col_names):
    #     df[col_names_normalized[i]] = df[col] / benchmark_DQN[0][col] # we fix the seq=0, which can be ajusted for each seed in the future
    # df['Average_normalized (%)'] = df[col_names_normalized].mean(axis=1) * 100.0
    
    # print(df)

print([env.game_name for env in Games[seq]])
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))




if PLOT:
    for v_cord in np.arange(switch - 1, plot_steps, switch):
        plt.axvline(x=v_cord, color='k', ls=':', alpha=0.8, lw=0.6)
    custom_lines = [
        Line2D([0], [0], color='g', lw=2),
        Line2D([0], [0], color='pink', lw=2),
        Line2D([0], [0], color='brown', lw=2),
        Line2D([0], [0], color='black', lw=2),
        Line2D([0], [0], color='orange', lw=2),
        Line2D([0], [0], color='b', lw=2),
        Line2D([0], [0], color='r', lw=2),
    ]
    fig.legend(custom_lines,
               ["DQN-Reset", 'DQN-Finetune', "DQN-MultiHead", "DQN-LargeBuffer", "Random", "PT-DQN-0.5x", "Ours"],
               ncol=4,
               fontsize=14, loc='upper center', bbox_to_anchor=(0.52, 0.92, 0.0, 0.0), frameon=False)
    # fig.legend(custom_lines, ["Ours","PT-DQN-0.5x", "DQN-multi-head", "DQN-large buffer", "Random"], ncol=3, fontsize=14,loc="lower center", bbox_to_anchor=(0.35, 0.21, 0.4, 0.0), frameon=False)
    # fig.legend(custom_lines, ["DQN", f"Ours (finetunefast) {finetunefast}", "PT-DQN-0.5x (ours)"], ncol=3, fontsize=14,loc="lower center", bbox_to_anchor=(0.35, 0.21, 0.4, 0.0), frameon=False)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(5, 1))
    ax.xaxis.major.formatter._useMathText = True
    ax.set_xlabel("Steps", fontsize=20)
    ax.set_ylabel("Episodic Return", fontsize=20)
    ax.set_title(f"MinAtar (Sequence {seq})", fontsize=24)
    ax.tick_params(labelsize=18)
    fig.tight_layout()
    # pdf.savefig(fig, bbox_inches = 'tight', dpi=300)
    plt.show()
# pdf.close()