import numpy as np
import random
import pickle
import copy

from CL_envs import *
from argparse import ArgumentParser
from tqdm import tqdm
import os, time

parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help="seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=3500000, help="number of episodes")
parser.add_argument('--switch', type=int, default=500000, help="switch env steps")
parser.add_argument('--save', action="store_true")
parser.add_argument("--gpu", type=int, default=0, help="Random seed and device selector")
###### no need to save model
parser.add_argument('--seq', type=int, default=0, help="selected sequence in the environment list")
parser.add_argument('--reset', type=int, default=1, help="reset every environment")

args = parser.parse_args()

np.random.seed(args.seed)
random.seed(args.seed)

returns_array = np.zeros(args.t_steps)


Games = []
gameid = 0
env = CL_envs_func_new(seq=args.seq, game_id=gameid, seed=args.seed)
Games.append(env.game_name)
avg_return = 0
epi_return = 0
done = False
_ = env.reset()

for step in tqdm(range(args.t_steps)):
    
    if step %args.switch == 0 and step > 0:
        gameid += 1
        env = CL_envs_func_new(seq=args.seq, game_id=gameid, seed=args.seed)
        Games.append(env.game_name)
        _ = env.reset()
        epi_return = 0

    curr_action = env.action_space.sample()
    _, rew, done, _ = env.step(curr_action)
    epi_return += rew

    if done:
        cs = env.reset()
        avg_return = 0.99 * avg_return + 0.01 * epi_return
        epi_return = 0

    returns_array[step] = copy.copy(avg_return)

if args.save:
    os.makedirs("results", exist_ok=True)
    filename = ("Random_"+str(args.t_steps)+"_switch_"+str(args.switch)+"_env_name_"+str(args.env_name) +
                "_seq_" + str(args.seq) +"_reset_"+str(args.reset) + "_seed_"+str(args.seed)+".pkl")
    with open("results/"+filename, "wb") as f:
        pickle.dump(returns_array, f)

print('Games: ', Games, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))