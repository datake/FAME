import numpy as np
import pickle
import itertools
import torch
import torch.optim as optim
import copy

from model import *
from replay import *
from CL_envs import *

from argparse import ArgumentParser
from configparser import ConfigParser
from tqdm import tqdm
import os, time

parser = ArgumentParser(description="Parameters for the code - ARTD on gym envs")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=3500000, help="total number of steps")
parser.add_argument('--switch', type=int, default=500000, help="switch env steps")
parser.add_argument('--lr1', type=float, default=1e-8, help="learning rate for weights")
parser.add_argument('--lr2', type=float, default=1e-4, help="learning rate for transient values")
parser.add_argument('--update', type=int, default=50000, help="PM update frequency")
parser.add_argument('--decay', type=float, default=0, help="decay transient weights after transfer")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--save', action="store_true")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save-model', action="store_true")
parser.add_argument("--gpu", type=int, default=0, help="Random seed and device selector")
parser.add_argument('--seq', type=int, default=0, help="selected sequence in the environment list")
parser.add_argument('--CNNhalf', type=int, default=1, help="half CNN, 1: half CNN(default), 0: full CNN")
parser.add_argument('--boundary', type=int, default=0, help="1: known boundary, 0: unknown boundary")
parser.add_argument('--reset', type=int, default=1, help="default=1, reset every environment")

args = parser.parse_args()
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config[str(args.env_name)]
gamma = float(misc_param['gamma'])
epsilon = float(misc_param['epsilon'])


####### T update: decompsing Q value into P+T, and only update T_Net #######
def train_T_Net(): # fast/transient network for the current task with the parameters w
	states, actions, next_states, rewards, done = exp_replay.sample() # only on the current buffer
	with torch.no_grad(): # no update for targets and P_Net
		T_next_pred = Target_net(next_states)
		P_next_pred = P_Net(next_states)
		P_pred = P_Net(states)
		P_pred = P_pred.gather(1, actions)
	T_pred = T_Net(states)
	T_pred = T_pred.gather(1, actions)
	##### the loss function is on TD erro of the whole PT, but we only update T_Net #####
	"""
	Essentially, it is a classical TD learning, but value function is decomposed into P+T. and here we only update T.
	"""
	targets = rewards + (1 - done) * gamma * ((P_next_pred + T_next_pred).max(1)[0]).reshape(-1, 1)
	loss = T_criterion(T_pred+P_pred, targets)
	T_opt.zero_grad()
	loss.backward()
	T_opt.step()
	return loss.item()

def train_P_Net(): # update the permanent network with samples from the PM buffer
	loss_u = 0
	u_steps = (exp_replay_PM.size()//args.batch_size) - 1
	for p_update in range(u_steps):
		curr_batch = list(itertools.islice(exp_replay_PM.memory, p_update*args.batch_size, (p_update+1)*args.batch_size))
		states, actions, old_p_vals = map(torch.stack, zip(*curr_batch))
		states = states.to(device)
		actions = actions.to(device)
		old_p_vals = old_p_vals.to(device)
		with torch.no_grad():
			T_pred = T_Net(states).gather(1, actions)
		P_pred = P_Net(states).gather(1, actions)
		loss = P_criterion(P_pred, T_pred+old_p_vals) 
		P_opt.zero_grad()
		loss.backward()
		P_opt.step()
		loss_u += loss.item()
	return loss_u/u_steps

def get_action(c_obs): # take action by using P+T network -> [P network Q value, action]
	c_obs = np.moveaxis(c_obs, 2, 0)
	c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
	with torch.no_grad():
		curr_T_vals = T_Net(c_obs.unsqueeze(0))
		curr_P_vals = P_Net(c_obs.unsqueeze(0))
		curr_Q_vals = curr_T_vals + curr_P_vals
	if np.random.random() <= epsilon:
		action = env.action_space.sample()
	else:
		action = curr_Q_vals.max(1)[1].item()
	return curr_P_vals[0][action], action

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

filename = ("PT_DQN_0.5x"+"_env_name_"+str(args.env_name)+"_gamma_"+misc_param['gamma']+\
		"_steps_"+str(args.t_steps)+"_switch_"+str(args.switch)+"_update_"+str(args.update)+"_decay_"+str(args.decay)+\
		"_lr1_"+str(args.lr1)+"_lr2_"+str(args.lr2)+"_batch_"+str(args.batch_size)+ "_seq_" + str(args.seq) + "_CNNhalf_"+str(args.CNNhalf)+
			"_boundary"+ str(args.boundary) + '_reset_'+ str(args.reset) +"_seed_"+str(args.seed))


######## initialization
Games = []
gameid = 0
env = CL_envs_func_replacement(seq=args.seq, game_id=gameid, seed=args.seed)
Games.append(env.game_name)

in_channels = env.observation_space.shape[2] # [10, 10, 7]
num_actions = env.action_space.n

# CNN_half: half CNN channels and last feature map dimension
if args.CNNhalf == 1:
	T_Net = CNN_half(in_channels, num_actions).to(device)
else:
	T_Net = CNN(in_channels, num_actions).to(device)
T_opt = optim.Adam(T_Net.parameters(), lr=args.lr2)
T_criterion = torch.nn.MSELoss()

if args.CNNhalf == 1:
	P_Net = CNN_half(in_channels, num_actions).to(device)
else:
	P_Net = CNN(in_channels, num_actions).to(device)
P_opt = optim.SGD(P_Net.parameters(), lr=args.lr1)
P_criterion = torch.nn.MSELoss()

# for the transient network, we use the same structure as the target network
if args.CNNhalf == 1:
	Target_net = CNN_half(in_channels, num_actions).to(device)
else:
	Target_net = CNN(in_channels, num_actions).to(device)

Target_net.load_state_dict(T_Net.state_dict())

exp_replay = expReplay(batch_size=args.batch_size, device=device)
if args.boundary == 1: # known boundary: update the P net every args.switch
	exp_replay_PM = expReplay_PM(max_size=args.switch, batch_size=args.batch_size, device=device)
else: # unknown boundary: every args.update, update the P net
	exp_replay_PM = expReplay_PM(max_size=args.update, batch_size=args.batch_size, device=device)

returns_array = np.zeros(args.t_steps)

avg_return = 0
epi_return = 0
done = False
cs = env.reset()

for step in tqdm(range(args.t_steps)):
	
	if step %args.switch == 0 and step > 0:

		gameid += 1
		env = CL_envs_func_replacement(seq=args.seq, game_id=gameid, seed=args.seed)
		Games.append(env.game_name)

		cs = env.reset()
		epi_return = 0

		if args.reset == 1:
			if args.CNNhalf == 1:
				T_Net = CNN_half(in_channels, num_actions).to(device)
			else:
				T_Net = CNN(in_channels, num_actions).to(device)
			T_opt = optim.Adam(T_Net.parameters(), lr=args.lr2)

			avg_return = 0

			print('Reset the fast learner')

	val_p, c_action = get_action(cs)
	ns, rew, done, _ = env.step(c_action)
	epi_return += rew
	exp_replay.store(cs, c_action, ns, rew, done)
	exp_replay_PM.store(cs, c_action, val_p)


	if step % 1000 == 0 and step > 0: # before updaing the leaner, guarantee the target net is correct not from the last environment
		Target_net.load_state_dict(T_Net.state_dict())

	if exp_replay.size() >= args.batch_size:
		loss = train_T_Net()

	cs = ns



	if args.boundary == 1:
		if (step + 1) % args.switch == 0:
			p_loss = train_P_Net()
			# reset T_Net and buffer
			T_Net.__init__(in_channels, num_actions)
			T_Net = T_Net.to(device)
			exp_replay_PM.delete()

	else:
		if (step+1)%args.update == 0: # update the parmanent network 50000
			p_loss = train_P_Net()
			for params in T_Net.parameters():
				params.data *= args.decay



	if done:
		cs = env.reset()
		avg_return = 0.99 * avg_return + 0.01 * epi_return
		epi_return = 0

	returns_array[step] = copy.copy(avg_return)

	if (step + 1) % args.switch == 0:

		exp_replay.delete()
		print('Clear the buffer, the current memory size is :', exp_replay.size())

		if args.save_model:
			torch.save(P_Net.state_dict(), "models/" + filename + "_Net"+ str(gameid) + ".pt")


if args.save:
	os.makedirs("results", exist_ok=True)
	with open("results/"+filename+"_returns.pkl", "wb") as f:
		pickle.dump(returns_array, f)

print('Games: ', Games, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))