import time
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
import os
from torch.optim.lr_scheduler import ExponentialLR
from scipy import stats

parser = ArgumentParser(description="Parameters for the FAME in MinAtari")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=3500000, help="total number of steps")
parser.add_argument('--switch', type=int, default=500000, help="switch env steps")
parser.add_argument('--lr1', type=float, default=1e-3, help="learning rate for meta learner")
parser.add_argument('--lr2', type=float, default=1e-5, help="learning rate for fast learner")
parser.add_argument('--update', type=int, default=50000, help="PM update frequency")
parser.add_argument('--decay', type=float, default=0.75, help="decay transient weights after transfer")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--save', action="store_true")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save-model', action="store_true")
parser.add_argument("--gpu", type=int, default=0, help="Random seed and device selector")

parser.add_argument('--seq', type=int, default=0, help="selected sequence in the environment list")
parser.add_argument('--size_fast2meta', type=int, default=12000, help="size of fast2meta buffer")
parser.add_argument('--size_meta', type=int, default=100000, help="size of meta buffer")
parser.add_argument('--detection_step', type=int, default=1200, help="detection step: number of expisodes in detection, 300 steps average episode!")
parser.add_argument('--epoch_meta', type=int, default=200,help="epoch to train meta learner")
parser.add_argument('--reset', type=int, default=1,help="reset the network every time")

### forward transfer by behavior cloning
parser.add_argument('--warmstep', type=int, default=50000, help="the number of steps to do warm-up")
parser.add_argument('--lambda_reg', type=float, default=1.0, help="hyperparameter for the regularization behavior cloning term, default method")

### one-vs-one hypothesis test
parser.add_argument('--use_ttest', type=int, default=0, help="one-vs-one hypothesis test:, 1: on, 0: off, default off")


args = parser.parse_args()
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config[str(args.env_name)]
gamma = float(misc_param['gamma'])
epsilon = float(misc_param['epsilon'])


#### check the cuda status
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("device_count:", torch.cuda.device_count())
torch.cuda.init()
device = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(device)
print("device_count:", torch.cuda.device_count())

print(args)
num_envs = int(args.t_steps / args.switch) # number of environments
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def train_faster(reg=None):  # fast/transient network for the current task with the parameters w
    states, actions, next_states, rewards, done = exp_replay_fast.sample()  # only on the current buffer
    with torch.no_grad():
        fast_next_pred = Target_net(next_states)
    # TD update
    targets = rewards + (1 - done) * gamma * (fast_next_pred.max(1)[0]).reshape(-1, 1)
    fast_pred = Fast_Learner(states).gather(1, actions)
    loss = Fast_criterion(fast_pred, targets)

    if reg is not None and args.lambda_reg > 0:
        # behavior cloning regularization term
        with torch.no_grad():
            soft_target = F.softmax(reg(states), dim=-1)
        logit_input = F.log_softmax(Fast_Learner(states), dim=-1)
        loss_reg = F.kl_div(logit_input, soft_target, reduction='batchmean')  # [bs, actions] -> [bs, 1]
        loss = loss + args.lambda_reg * loss_reg

    Fast_opt.zero_grad()
    loss.backward()
    Fast_opt.step()
    return loss.item()


def train_meta():  # update the meta learner only on meta buffer with old data and on fast buffer with new data
    # copy the k-1 meta learner
    Meta_Learner_old = copy.deepcopy(Meta_Learner).to(device)

    u_steps = (exp_replay_meta.size() // args.batch_size) - 1 # e.g., (args.size_fast2meta // 64) - 1
    for epoch in range(args.epoch_meta):
        for i, p_update in enumerate(range(u_steps)):

            ######## step 1: update the meta learner via old data from meta buffer
            states_meta, actions_meta = exp_replay_meta.sample() # sample helps reduce the correlation

            states_meta = states_meta.to(device)
            actions_meta = actions_meta.to(device)
 
            logits = Meta_Learner(states_meta)
            log_probs = F.log_softmax(logits, dim=-1)
            loss1 = Meta_criterion2(log_probs, actions_meta.view(-1)) # [bs, actions] -> [bs, 1] selection the argmax action => MLE
          
            if i % gameid == 0: #### this is because the objective put equal weight to each enviroment and so only update the new data every gameid time
                ######### step 2: update the meta learner via new data from fast2meta buffer
                states_fast, actions_fast = exp_replay_fast2meta.sample()

                # always only Meta learner, but with new data
                logits = Meta_Learner(states_fast)
                log_probs = F.log_softmax(logits, dim=-1)
                loss2 = Meta_criterion2(log_probs, actions_fast.view(-1))  # [bs, actions] -> [bs, 1]
              
                loss = loss1 + loss2
            else:
                loss = loss1

            ######### Step 3: update the meta learner
            Meta_opt.zero_grad()
            loss.backward()
            Meta_opt.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch: {epoch+1}/{args.epoch_meta}, Meta Loss: {loss.item():.2e}, current lr: {Meta_opt.param_groups[0]['lr']:.2e}", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        if (epoch + 1) % 2 == 0:
            Meta_scheduler.step()  # decay learning rate




def get_action_detection(c_obs, testQ):  # take action by greedy action on a given Q function
    c_obs = np.moveaxis(c_obs, 2, 0)
    c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
    with torch.no_grad():
        curr_Q_vals = testQ(c_obs.unsqueeze(0))
    action = curr_Q_vals.max(1)[1].item()
    return action, curr_Q_vals[0][action]

def get_action(c_obs, LEARNER):  # take action by the fast learner with the environment, but update via a regularization
    c_obs = np.moveaxis(c_obs, 2, 0)
    c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
    with torch.no_grad():
        curr_Q_vals = LEARNER(c_obs.unsqueeze(0))
    if np.random.random() <= epsilon:
        action = env.action_space.sample()
    else:
        action = curr_Q_vals.max(1)[1].item()
    return action

def get_action_exploration(c_obs, LEARNER):  # when p_explore > 0, use the expert learner to guide the exploration
    c_obs = np.moveaxis(c_obs, 2, 0)
    c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
    with torch.no_grad():
        curr_Q_vals = LEARNER(c_obs.unsqueeze(0))
    if np.random.random() <= epsilon:
        action = env.action_space.sample()
    else:
        action = curr_Q_vals.max(1)[1].item()
    return action


if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



filename = ("FAME" + "_steps_" + str(args.t_steps) + "_switch_" + str(args.switch) + "_update_" + str(
    args.update) + "_lr1_" + str(args.lr1) + "_lr2_" + str(args.lr2)  + "_size_fast2meta_" + str(args.size_fast2meta) + "_detection_step_" + str(args.detection_step)
            + "_seq_" + str(args.seq) + '_epoch_meta_' + str(args.epoch_meta)
            + "_warmstep_" + str(args.warmstep) + "_lambda_reg_" + str(args.lambda_reg)
            + "_seed_" + str(args.seed))



######## initialization: randomly select an env
Games = []
gameid = 0
env = CL_envs_func_replacement(seq=args.seq, game_id=gameid, seed=args.seed)
Games.append(env.game_name)
in_channels = env.observation_space.shape[2]  # [10, 10, 7]
num_actions = env.action_space.n


Fast_Learner = CNN(in_channels, num_actions).to(device)
Fast_opt = optim.Adam(Fast_Learner.parameters(), lr=args.lr2)
Fast_criterion = torch.nn.MSELoss()


Random_Learner = CNN(in_channels, num_actions).to(device)
Meta_Learner = CNN(in_channels, num_actions).to(device)
Meta_opt = optim.Adam(Meta_Learner.parameters(), lr=args.lr1)
Meta_scheduler = ExponentialLR(Meta_opt, gamma=0.95)
Meta_criterion = torch.nn.MSELoss()
Meta_criterion2 = torch.nn.NLLLoss()
Meta2fast_criterion = torch.nn.MSELoss()

# for the fast learner network, we use the same structure as the target network
Target_net = CNN(in_channels, num_actions).to(device)
Target_net.load_state_dict(Fast_Learner.state_dict())


exp_replay_fast = expReplay(batch_size=args.batch_size, device=device) # used for learning fast learner
# fast2meta: contribute to the data in meta learner; learn the classifier
exp_replay_fast2meta = expReplay_Meta(max_size=args.size_fast2meta, batch_size=args.batch_size, device=device)
exp_replay_meta = expReplay_Meta(max_size=args.size_meta, batch_size=args.batch_size, device=device)


returns_array = np.zeros(args.t_steps)

avg_return = 0
epi_return = 0
done = False
cs = env.reset()
print(f'##################### Environment {gameid+1}/{num_envs}: {env.game_name}#####################')

# define interval to store data to fast2meta and meta buffer
interval = [(i*args.switch-args.size_fast2meta-1, i*args.switch-1) for i in range(1, int(args.t_steps / args.switch)+1)]
def in_intervals(x):
    return any(start <= x <= end for start, end in interval)

Reg_Learner = None
pbar = tqdm(total=args.t_steps)

Flag_Reg = []
MAX_STEP = 300
step = 0

Q_normalize = []

META_WARMUP = 0

while step < args.t_steps:
    """
    avg_return: changes after each episode by rewards then smooth
    returns_array: be constant within each episode and change by ave_return 
    """
    if step % args.switch == 0 and step > 0: # when it comes to a new env

        if args.reset == 1:
            avg_return = 0

        META_WARMUP = 0 # INTIALIZATION: NO METAWARMUP

        ### Step 1: Start a New Environment

        gameid += 1
        old_envname = env.game_name
        env = CL_envs_func_replacement(seq=args.seq, game_id=gameid, seed=args.seed)
        print(f'##################### Environment {gameid+1}/{num_envs}: {env.game_name}#####################')
        Games.append(env.game_name)
        cs = env.reset()
        cs_initial = cs

        ### Step 2: Detection to Determine the Regularization: set the number of episodes does not work as the random policy is often non-stopping

        print('##################### Step 1: Detection via Policy Evaluation !')
        # use flag to avoid unnecessary detection to avoid sample sacrifice

        FLAG_ENV2 = True if step != args.switch else False # in the second environment, if reset=0, no need to evaluate, otherwise we still to detect one time
      
        if not FLAG_ENV2:
            print('No Detection for meta, only compare fast and reset as in the 2nd environment!')

        epi_return = 0


        max_step = 0

        Num_detection_meta = args.detection_step * FLAG_ENV2 # when k=2, meta learner is always inferior to fast learner, no need to evaluate meta rewards to imporve sample efficiency
        Num_detection_fast = args.detection_step

        ##### evaluation on the fast policy
        epi_return_fast = 0
        avereward_fast = []

        for step_small in range(Num_detection_fast):
            c_action, _ = get_action_detection(cs, Fast_Learner)  # interact by fast learner
            ns, rew, done, _ = env.step(c_action)
            exp_replay_fast.store(cs, c_action, ns, rew, done) # store the experience for the fast learner
            epi_return_fast += rew # for plotting the learning curves
            cs = ns
            step += 1
            max_step += 1
            if done or max_step > MAX_STEP:
                cs = env.reset()
                avg_return = 0.99 * avg_return + 0.01 * epi_return_fast
                avereward_fast.append(epi_return_fast)
                epi_return_fast = 0
                max_step = 0
            returns_array[step] = copy.copy(avg_return)
            pbar.update(1)
        if Num_detection_fast > 0:
            if len(avereward_fast) == 0:
                print(f'Evaluation on Fast Learner, Number of Episodes: {len(avereward_fast)}', 'Even one episode is not finished yet....')
            else:
                print(f'Evaluation on Fast Learner, Average Reward: {np.mean(avereward_fast)}, Number of Episodes: {len(avereward_fast)}, all: ', avereward_fast)
        else:
            print(f"No Evaluation on Fast Learner")

        epi_return_meta = 0
        avereward_meta = []
        max_step = 0

        if Num_detection_meta > 0:
            cs = env.reset()

        ##### evaluation on the meta policy
        for step_small in range(Num_detection_meta):
            c_action, _ = get_action_detection(cs, Meta_Learner) # interact by meta learner
            ns, rew, done, _ = env.step(c_action)
            exp_replay_fast.store(cs, c_action, ns, rew, done) # store the experience for the fast learner
            epi_return_meta += rew
            cs = ns
            step += 1
            max_step += 1
            if done or max_step > MAX_STEP:
                cs = env.reset()
                avg_return = 0.99 * avg_return + 0.01 * epi_return_meta
                avereward_meta.append(epi_return_meta)
                epi_return_meta = 0
                max_step = 0
            returns_array[step] = copy.copy(avg_return)
            pbar.update(1)

        if Num_detection_meta > 0:
            if len(avereward_meta) == 0:
                print(f'Evaluation on Meta Learner, Number of Episodes: {len(avereward_meta)}', 'Even one episode is not finished yet....')
            else:
                print(f'Evaluation on Meta Learner, Average Reward: {np.mean(avereward_meta)}, Number of Episodes: {len(avereward_meta)}, all: ', avereward_meta)
        else:
            print(f"No Evaluation on Meta Learner")

        # if one eposide is not finished, the initialization is very poor, just reset
        Avereward_meta = -1000 if len(avereward_meta) == 0 else np.mean(avereward_meta)
        Avereward_fast = -1000 if len(avereward_fast) == 0 else np.mean(avereward_fast)

        _, value_rand = get_action_detection(cs_initial, Random_Learner)
        # _, value_fast = get_action_detection(cs_initial, Fast_Learner)
        Avereward_rand = float(value_rand.cpu().numpy())
        print('Reward meta', round(Avereward_meta, 2),'Reward_fast', round(Avereward_fast, 2), 'Reward_random', round(Avereward_rand, 2))

        def Hypothesis_test(avereward_list1, avereward_list2, Avereward1, Avereward2):
            # if one eposide is not finished, or the number of episodes is too small, just compare the average reward without t-test, otherwise do the t-test to check if the improvement is significant
            if len(avereward_list1) < 2 or len(avereward_list2) < 2:
                return Avereward1 > Avereward2
            else:
                t_statistic, p_value = stats.ttest_ind(avereward_list1, avereward_list2, alternative='greater', equal_var=False)
                return p_value < 0.05
        
        Meta_Fast = Hypothesis_test(avereward_meta, avereward_fast, Avereward_meta, Avereward_fast) if args.use_ttest == 1 else (Avereward_meta > Avereward_fast)
        Fast_Meta = Hypothesis_test(avereward_fast, avereward_meta, Avereward_fast, Avereward_meta) if args.use_ttest == 1 else (Avereward_fast > Avereward_meta)
        ######## Hypothesis test via empirical ranking
        if Meta_Fast and Avereward_meta > Avereward_rand:
            # meta initialization is better than the others
            print('##################### Step 2: Use Meta Initialization and Start Training !')
            META_WARMUP = 1 # step >= switch * 2,
            Flag_Reg.append('Meta')

        elif Fast_Meta and Avereward_fast > Avereward_rand:
            Flag_Reg.append('Fast')
            print('##################### Step 2: Use Fast Initialization / FineTune Fast Learner!')
        else: # random initialization is better than the others
            if args.reset == 1:
                Fast_Learner = CNN(in_channels, num_actions).to(device)
                Fast_opt = optim.Adam(Fast_Learner.parameters(), lr=args.lr2)
                Flag_Reg.append('Random')
                print('##################### Step 2: Use Random Initialization')
            else:
                Flag_Reg.append('Fast')
                print('##################### Step 2: Use Fast Initialization / FineTune Fast Learner!')


        if Num_detection_meta + Num_detection_fast > 0:
            # always do that when it occurs a new environment, otherwise it will remain in the old environment
            Target_net.load_state_dict(Fast_Learner.state_dict())

            cs = env.reset() # reinitialize the environment

    # guided exploration
    if args.p_explore > 0 and META_WARMUP == 1 and (step % args.switch < args.warmstep):
        if step % args.switch == args.warmstep - 1:
            print(f'Finished the guided exploration at the step {step}')
        c_action = get_action_exploration(cs, Fast_Learner, Meta_Learner, args.p_explore)
    else:
        c_action = get_action(cs, Fast_Learner)
    ns, rew, done, _ = env.step(c_action)
    epi_return += rew
    # always store the experience for the buffer for the fast leaner
    exp_replay_fast.store(cs, c_action, ns, rew, done)

    # at the end of each environment (<args.size_fast2meta), store the data into fast2meta buffer for knowledge integration
    if in_intervals(step):
        exp_replay_fast2meta.store(cs, c_action)

    if step % 1000 == 0 and step > 0:  # before updaing the leaner, guarantee the target net is correct not from the last environment
        Target_net.load_state_dict(Fast_Learner.state_dict())

    if exp_replay_fast.size() >= args.batch_size:

        if META_WARMUP == 1 and args.lambda_reg > 0 and (step % args.switch < args.warmstep):
            if step % args.switch == args.warmstep - 1:
                print(f'Use the behavior cloning as an regularization at the step {step}')
            loss = train_faster(reg=Meta_Learner)
        else:
            loss = train_faster(reg=None)

    cs = ns

    if done:
        cs = env.reset()
        avg_return = 0.99 * avg_return + 0.01 * epi_return
        epi_return = 0

    returns_array[step] = copy.copy(avg_return)

    ############ the last step in the current env: knowledge distillation
    if (step+1) % args.switch == 0:
        if step+1 == args.switch:  # the first time
            # Meta_Learner.load_state_dict(Fast_Learner.state_dict())

            print('First time: No need to update Meta learner')
        else:
            print('##################### Step 3: Updating Meta Learner!')
            print('Old Meta data set: ', exp_replay_meta.size(), 'fast data set: ', exp_replay_fast2meta.size())

            # should reset the optimizer with the initial lr as args.lr1
            Meta_opt = optim.Adam(Meta_Learner.parameters(), lr=args.lr1)
            Meta_scheduler = ExponentialLR(Meta_opt, gamma=0.95)
            train_meta()  # Fast_learner + copied old k-1 Meta_learner -> new k meta learner

        exp_replay_fast2meta.copy_to(exp_replay_meta)
        exp_replay_fast2meta.delete()  # clear the memory
        print('##################### Step 4: Fast2Meta Copy to Meta buffer: New Meta data set: ', exp_replay_meta.size(), 'fast data set: ', exp_replay_fast2meta.size())


        exp_replay_fast.delete()

        if args.save_model:
            os.makedirs("models", exist_ok=True)
            torch.save(Meta_Learner.state_dict(), "models/" + filename + "_Meta" + str(gameid) + ".pt")




    ###### use while loop instead of for
    step += 1
    pbar.update(1) # for progress bar

pbar.close()



if args.save:
    os.makedirs("results", exist_ok=True)
    with open("results/" + filename + "_returns.pkl", "wb") as f:
        pickle.dump(returns_array, f)


print('Regularization: ', Flag_Reg, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print('Games: ', Games, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
print(args)