import numpy as np
import pickle
import itertools
import torch
import torch.optim as optim
import copy

from model import *
from replay import *
from CL_envs import *
from tqdm import tqdm
from argparse import ArgumentParser
from configparser import ConfigParser
import os, time

parser = ArgumentParser(description="Parameters for the code - DQN")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=3500000, help="total number of steps")
parser.add_argument('--switch', type=int, default=500000, help="switch env steps")
parser.add_argument('--lr1', type=float, default=1e-5, help="learning rate for DQN")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--save', action="store_true")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save-model', action="store_true")
parser.add_argument("--gpu", type=int, default=0, help="Random seed and device selector")
parser.add_argument('--seq', type=int, default=0, help="selected sequence in the environment list")
parser.add_argument('--reset', type=int, default=1, help="default=1: reset every environment")


args = parser.parse_args()
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config[str(args.env_name)]
gamma = float(misc_param['gamma'])
epsilon = float(misc_param['epsilon'])

def train_Net():
	states, actions, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		next_pred = Target_net(next_states)
		next_pred = next_pred.max(1)[0]
	pred = Net(states)
	pred = pred.gather(1, actions)
	targets = rewards + (1 - done) * gamma * next_pred.reshape(-1, 1)
	loss = criterion(pred, targets)
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item()

def get_action(c_obs):
	c_obs = np.moveaxis(c_obs, 2, 0)
	c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
	with torch.no_grad():
		curr_Q_vals = Net(c_obs.unsqueeze(0))
	if np.random.random() <= epsilon:
		action = env.action_space.sample()
	else:
		action = curr_Q_vals.max(1)[1].item()
	return curr_Q_vals[0][action], action

if torch.cuda.is_available():
	device = torch.device(f"cuda:{args.gpu}")
	torch.cuda.set_device(device)
else:
	device = torch.device("cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

filename = "DQN_large_buffer"+"_env_name_"+args.env_name+"_gamma_"+misc_param['gamma']+\
		"_steps_"+str(args.t_steps)+"_switch_"+str(args.switch)+"_batch_"+\
		str(args.batch_size)+"_lr1_"+str(args.lr1) + "_seq_" + str(args.seq) + "_reset_"+str(args.reset) + "_seed_"+str(args.seed)

Games = []
gameid = 0
env = CL_envs_func_replacement(seq=args.seq, game_id=gameid, seed=args.seed)
Games.append(env.game_name)
in_channels = env.observation_space.shape[2]
num_actions = env.action_space.n

Net = CNN(in_channels, num_actions).to(device)
opt = optim.Adam(Net.parameters(), lr=args.lr1)
criterion = torch.nn.MSELoss()

Target_net = CNN(in_channels, num_actions).to(device)
Target_net.load_state_dict(Net.state_dict())

exp_replay = expReplay_Large(batch_size=args.batch_size, device=device)

returns_array = np.zeros(args.t_steps)

avg_return = 0
epi_return = 0
done = False
cs = env.reset()

for step in tqdm(range(args.t_steps)):
	
	if step % args.switch == 0 and step > 0:
		gameid += 1
		env = CL_envs_func_replacement(seq=args.seq, game_id=gameid, seed=args.seed)
		Games.append(env.game_name)

		cs = env.reset()
		epi_return = 0

		if args.reset == 1:  # reset instead of finetuning
			avg_return = 0
			Net = CNN(in_channels, num_actions).to(device)
			opt = optim.Adam(Net.parameters(), lr=args.lr1)
			print('Reset the fast learner')


	_, c_action = get_action(cs)
	ns, rew, done, _ = env.step(c_action)
	epi_return += rew
	exp_replay.store(cs, c_action, ns, rew, done)


	if step % 1000 == 0 and step > 0: # before updaing the leaner, guarantee the target net is correct not from the last environment
		Target_net.load_state_dict(Net.state_dict())

	if exp_replay.size() >= args.batch_size:
		loss = train_Net()
	
	cs = ns



	if done:
		cs = env.reset()
		avg_return = 0.99 * avg_return + 0.01 * epi_return
		epi_return = 0

	returns_array[step] = copy.copy(avg_return)
	if (step + 1) % args.switch == 0:

		exp_replay.delete()
		print('Clear the buffer, the current memory size is :', exp_replay.size())


		if args.save_model:
			torch.save(Net.state_dict(), "models/" + filename + "_Net" + str(gameid) + ".pt")


if args.save:
	os.makedirs("results", exist_ok=True)
	with open("results/"+filename+"_returns.pkl", "wb") as f:
		pickle.dump(returns_array, f)

print('Games: ', Games, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

