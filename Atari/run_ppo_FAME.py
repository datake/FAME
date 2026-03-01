# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from typing import Literal, Tuple, Optional
import pathlib
from tqdm import tqdm
from replay import *
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import copy
from task_utils import *
from scipy import stats


from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from models import (
    CnnSimpleAgent,
    DinoSimpleAgent,
    CnnCompoNetAgent,
    ProgressiveNetAgent,
    PackNetAgent,
    FAMEAgent,
)


@dataclass
class Args:
    # Model type
    model_type: Literal[
        "cnn-simple",
        "cnn-simple-ft",
        "dino-simple",
        "cnn-componet",
        "prog-net",
        "packnet",
        "FAME",
    ] = "FAME"
    # model_type: Literal[
    #     "cnn-simple",
    #     "cnn-simple-ft",
    #     "dino-simple",
    #     "cnn-componet",
    #     "prog-net",
    #     "packnet",
    #
    # ]
    #
    """The name of the model to use as agent."""
    dino_size: Literal["s", "b", "l", "g"] = "s"
    """Size of the dino model (only needed when using dino)"""

    save_dir: str = 'agents'
    """Directory where the trained model will be saved. If not provided, the model won't be saved"""
    prev_units: Tuple[pathlib.Path, ...] = () # finetune and componet
    """Paths to the previous models. Only used when employing a CompoNet or cnn-simple-ft (finetune) agent"""
    mode: int = 0
    """Playing mode for the Atari game. The default mode is used if not provided"""
    componet_finetune_encoder: bool = False
    """Whether to train the CompoNet's encoder from scratch of finetune it from the encoder of the previous task"""
    total_task_num: Optional[int] = 10
    """Total number of tasks, required when using PackNet"""
    prevs_to_noise: Optional[int] = 0
    """Number of previous policies to set to randomly selected distributions, only valid when model_type is `cnn-componet`"""

    # Experiment arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ppo-atari"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "ALE/SpaceInvaders-v5"
    """the id of the environment"""
    total_timesteps: int = int(3e3) # int(1e6)
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    lr_fast: float = 2.5e-4
    lr_meta: float = 2.5e-4
    size_fast2meta: int = 20000 # 12000
    size_meta: int = 200000 # 100000
    detection_step: int = 1200
    epoch_meta: int = 200 #
    warmstep: int=50000 #1% # 50000
    lambda_reg: float = 1.0
    buffer_path: Tuple[pathlib.Path, ...] = ()
    use_ttest: int = 0 # 1: use t-test; 0: use empirical ranking



def make_env(env_id, idx, capture_video, run_name, mode=None, dino=False):
    def thunk():
        if mode is None:
            env = gym.make(env_id)
        else:
            env = gym.make(env_id, mode=mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)

        if not dino:
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
        else:
            env = gym.wrappers.ResizeObservation(env, (224, 224))
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk





def train_meta():  # update the meta learner only on meta buffer with old data and on fast buffer with new data
    # copy the k-1 meta learner

    # Meta_Learner_old = copy.deepcopy(Meta_agent).to(device)

    u_steps = (exp_replay_meta.size() // BATCHSIZE) - 1 # e.g., (args.size_fast2meta // 64) - 1
    for epoch in range(args.epoch_meta):
        for i, p_update in enumerate(range(u_steps)):

            ######## step 1: update the meta learner via old data from meta buffer
            states_meta, actions_meta = exp_replay_meta.sample() # sample helps reduce the correlation

            states_meta = states_meta.to(device)
            actions_meta = actions_meta.to(device)
            logits = Meta_agent.forward(states_meta)
            log_probs = F.log_softmax(logits, dim=-1)
            loss1 = Meta_criterion2(log_probs, actions_meta.view(-1).long()) # [bs, actions] -> [bs, 1] selection the argmax action => MLE

            if i % (mode+1) == 0: #### this is because the objective put equal weight to each enviroment and so only update the new data every gameid time
                ######### step 2: update the meta learner via new data from fast2meta buffer
                states_fast, actions_fast = exp_replay_fast2meta.sample()
                logits = Meta_agent.forward(states_fast)
                log_probs = F.log_softmax(logits, dim=-1)
                loss2 = Meta_criterion2(log_probs, actions_fast.view(-1).long())  # [bs, actions] -> [bs, 1]
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


def train_agent(reg, start_time, writer, next_obs, next_done):
    # for rollout steps and save data into rollout buffer
    global global_step
    for step in range(0, args.num_steps):
        global_step += args.num_envs
        # print('global_step', global_step)
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: take the action logic, PPO use softmax instead of epsilon-greedy
        with torch.no_grad():
            if (
                    args.track
                    and args.model_type == "cnn-componet"
                    and global_step % 100 == 0
            ):
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs / 255.0,
                    log_writter=writer,
                    global_step=global_step,
                    prevs_to_noise=args.prevs_to_noise,
                )
            elif args.model_type == "cnn-componet":
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs / 255.0, prevs_to_noise=args.prevs_to_noise
                )
            else:
                # action, logprob entroy, critic value
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs / 255.0
                )

            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.step(
            action.cpu().numpy()
        )
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        ####### collect the last fast2meta_step steps's state and action into fast2meta/meta buffer
        if args.model_type == "FAME":
            if global_step > fast2meta_step:
                rollout_obs = obs.reshape(obs.shape[0]*obs.shape[1], *obs.shape[2:])  # (128, 8, 4, 84, 84) -> (1024, 4, 84, 84)
                rollout_action = actions.reshape(actions.shape[0]*actions.shape[1], *actions.shape[2:])  # (128, 8, 1) -> (1024, 1)
                exp_replay_fast2meta.store(rollout_obs, rollout_action) # [1024, 4, 84, 84], [1024, 1]

        ####### printing the log info
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    # print(
                    #     f"global_step={global_step}, episodic_return={info['episode']['r']}", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info["elapsed_time"]))
                    # )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )

    # bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs / 255.0).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        ### General Advantage Estimation
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = (rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t])
            advantages[t] = lastgaelam = (delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        returns = advantages + values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):  # default: 4
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            if args.model_type == "cnn-componet":
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds] / 255.0,
                    b_actions.long()[mb_inds],
                    prevs_to_noise=args.prevs_to_noise,
                )
            else:
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds] / 255.0, b_actions.long()[mb_inds]
                )
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [
                    ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                ]

            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                )

            # Policy loss: clipping loss and then max
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - args.clip_coef, 1 + args.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            ##### add behavioral cloning loss for FAME: KL divergence between reg's actor distribution and agent's actor distribution
            if reg is not None and args.lambda_reg > 0:
                with torch.no_grad():
                    logits_reg = reg.actor(reg.network(b_obs[mb_inds] / 255.0))
                    soft_target = F.softmax(logits_reg, dim=-1)
                logit_input = F.log_softmax(agent.actor(agent.network(b_obs[mb_inds] / 255.0)), dim=-1)
                loss_reg = F.kl_div(logit_input, soft_target, reduction='batchmean')  # [bs, actions] -> [bs, 1]
                v_loss = v_loss + args.lambda_reg * loss_reg

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            if args.model_type == "packnet":
                if global_step >= packnet_retrain_start:
                    agent.start_retraining()  # can be called multiple times, only the first counts
                agent.before_update()
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar(
        "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
    )
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    # print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar(
        "charts/SPS", int(global_step / (time.time() - start_time)), global_step
    )

if __name__ == "__main__":
    args = tyro.cli(Args)
    # freeway: 7 modes; SpaceInvaders: 10 modes
    args.batch_size = int(args.num_envs * args.num_steps) # 8*128=1024
    args.minibatch_size = int(args.batch_size // args.num_minibatches) # 1024 / 4 = 256
    args.num_iterations = args.total_timesteps // args.batch_size # 1e6 / 1024 = 976


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup:
    dino = "dino" in args.model_type # 'dino-simple' => True
    # initialize 8 envs

    if args.model_type == "FAME":
        BATCHSIZE = 64
        # exp_replay_fast = expReplay(batch_size=BATCHSIZE, device=device)  # no need for fast learner buffer
        # fast2meta: contribute to the data in meta learner; learn the classifier
        game = args.env_id.split("/")[-1].split('-')[0]  # e.g., SpaceInvaders, Freeway
        exp_replay_fast2meta = expReplay_Meta(max_size=args.size_fast2meta, batch_size=BATCHSIZE, device=device)
        exp_replay_meta = expReplay_Meta(max_size=args.size_meta, batch_size=BATCHSIZE, device=device)




    mode_list = TASKS[args.env_id]

    env_default = gym.vector.SyncVectorEnv(
            [
                make_env(
                    args.env_id, i, args.capture_video, f"{args.env_id.replace('/', '-')}0__{args.model_type}__{args.exp_name}__{args.seed}", mode=0, dino=dino  # args.env_id: Freeway-v5
                )
                for i in range(args.num_envs)
            ]
    )
    ACTION_SPACE = env_default.single_action_space

    if args.model_type == "FAME":
        """"only define one single fast agent and meta agent"""
        agent = FAMEAgent(env_default, fast=True).to(device)
        Meta_agent = FAMEAgent(env_default, fast=False).to(device)
        random_agent = FAMEAgent(env_default, fast=True).to(device)
        num_envs = args.num_envs
        optimizer = optim.Adam(agent.parameters(), lr=args.lr_fast, eps=1e-5)
        Fast_criterion = torch.nn.MSELoss()
        Meta_opt = optim.Adam(Meta_agent.parameters(), lr=args.lr_meta, eps=1e-5)
        Meta_scheduler = ExponentialLR(Meta_opt, gamma=0.95)
        Meta_criterion = torch.nn.MSELoss()
        Meta_criterion2 = torch.nn.NLLLoss()
        Meta2fast_criterion = torch.nn.MSELoss()
        warmstep = int(args.warmstep / args.batch_size)  # 50000 / 1024 = 48.8 (~5% of the total steps)
        fast2meta_step = int(args.size_fast2meta / args.batch_size)  # 12000 / 1024 = 11.7 (~1% of the total steps)

        MAX_STEP = 100

    for mode in mode_list:
        m = f"_{mode}" if args.mode is not None else ""
        run_name = f"{args.env_id.replace('/', '-')}{m}__{args.model_type}__{args.exp_name}__{args.seed}"  # ALE-Freeway-v5_0__packnet__run_ppo__1
        print("*** Run's name:", run_name)  # e.g., *** Run's name: ALE-Freeway-v5_0__packnet__run_ppo__1
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    args.env_id, i, args.capture_video, run_name, mode=mode, dino=dino  # args.env_id: Freeway-v5
                )
                for i in range(args.num_envs)
            ],
        )
        assert isinstance(
            envs.single_action_space, gym.spaces.Discrete
        ), "only discrete action space is supported"

        print(f"******************************************************** Model: {args.model_type}, mode: {mode} ********************************************************")

        ########################## different method ########################
        # baseline: train from scratch
        if args.model_type == "cnn-simple":
            agent = CnnSimpleAgent(envs).to(device)
        # Finetune: load the previous model, only load encoder
        elif args.model_type == "cnn-simple-ft":
            if len(args.prev_units) > 0:
                agent = CnnSimpleAgent.load(  # dirname: args.prev_units[0]
                    args.prev_units[0], envs, load_critic=False, reset_actor=True
                ).to(device)
            else:
                agent = CnnSimpleAgent(envs).to(device)
        # Dinov2 Meta transformer to encode
        elif args.model_type == "dino-simple":
            agent = DinoSimpleAgent(
                envs, dino_size=args.dino_size, frame_stack=4, device=device
            ).to(device)
        # CompoNet: load the previous modules
        elif args.model_type == "cnn-componet":
            agent = CnnCompoNetAgent(
                envs,
                prevs_paths=args.prev_units,
                finetune_encoder=args.componet_finetune_encoder,
                map_location=device,
            ).to(device)
        # ProgressiveNet: load the previous modules
        elif args.model_type == "prog-net":
            agent = ProgressiveNetAgent(
                envs, prevs_paths=args.prev_units, map_location=device
            ).to(device)
        # Packnet
        elif args.model_type == "packnet":
            # retraining in 20% of the total timesteps
            packnet_retrain_start = args.total_timesteps - int(args.total_timesteps * 0.2)
            if args.total_task_num is None:
                print("CLI argument `total_task_num` is required when using PackNet.")
                quit(1)
            if len(args.prev_units) == 0:
                agent = PackNetAgent(
                    envs,
                    task_id=(mode + 1),
                    is_first_task=True,
                    total_task_num=args.total_task_num,
                ).to(device)
            else:
                agent = PackNetAgent.load(
                    args.prev_units[0],
                    task_id=mode + 1,
                    restart_actor_critic=True,
                    freeze_bias=True,
                ).to(device)



        # load the meta and fast2meta buffer
        if args.model_type == "FAME":

            epi_return = 0
            Flag_Reg = []
            META_WARMUP = 0

            # start fast2meta collection from: args.size_fast2meta / args.batch_size = 12000 / 1024 = 11.7 (~1%)
        else:
            optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: rollout buffer
        obs = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape
            # e.g, (128, 8, 4, 84, 84) for Freeway-v5
        ).to(device)
        actions = torch.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape  # e.g, (128, 8, 1) for Freeway-v5
        ).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(
            device)  # action probability in each step given each state
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)  # next_obs: (8, 4, 84, 84) for Freeway-v5
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        for iteration in tqdm(range(1, args.num_iterations + 1)): # 1e6/batchsize=976
            """
            each interation includes:
                1. a rollout out across args.num_steps (128) steps
                2. GAE (Generalized Advantage Estimation) to compute the advantages
                3. optimization of the policy and value networks across args.update_epochs epochs
            """
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow


            ###### Step 1 (FAME): Adaptive Meta Warm Up ######
            if iteration == 1 and mode > 0 and args.model_type == "FAME":


                ### evaluate the step: sacrifice data in the detection step
                step = 0

                avg_return = 0
                META_WARMUP = 0  # INTIALIZATION: NO METAWARMUP

                # next_obs has already been reset
                currnt_obs = next_obs.clone()  # save the initial state
                currnt_obs_initial = next_obs.clone()  # save the initial state

                ### Step 2: Detection to Determine the Regularization: set the number of epsilod does not work as the random policy is often non-stopping
                print('##################### Step 1: Detection via Policy Evaluation !')
                FLAG_ENV2 = True if mode != 1 else False
                if not FLAG_ENV2:
                    print('No Detection for meta, only compare fast and reset as in the 2nd environment!')
                epi_return = 0
                max_step = 0
                Num_detection_meta = args.detection_step * FLAG_ENV2  # when k=2, meta learner is always inferior to fast learner, no need to evaluate meta rewards to imporve sample efficiency
                Num_detection_fast = args.detection_step

                ##### evaluation on the fast policy
                epi_return_fast = 0
                avereward_fast = []

                for step_small in range(Num_detection_fast):
                    action, logprob, _, value = agent.get_action_and_value(currnt_obs / 255.0)
                    next_obs, reward, terminations, truncations, infos = envs.step(
                        action.cpu().numpy()
                    )
                    next_obs = torch.Tensor(next_obs).to(device)
                    epi_return_fast += reward  # for plotting the learning curves
                    currnt_obs = next_obs
                    done = np.logical_or(terminations, truncations)
                    step += 1
                    max_step += 1
                    if done.all() or max_step > MAX_STEP:
                        currnt_obs, _ = envs.reset(seed=args.seed)  # next_obs: (8, 4, 84, 84) for Freeway-v5
                        currnt_obs = torch.Tensor(currnt_obs).to(device)
                        avg_return = 0.99 * avg_return + 0.01 * epi_return_fast
                        avereward_fast.append(epi_return_fast)
                        epi_return_fast = 0
                        max_step = 0
                    # returns_array[step] = copy.copy(avg_return)
                    # pbar.update(1)
                if Num_detection_fast > 0:
                    if len(avereward_fast) == 0:
                        print(f'Evaluation on Fast Learner, Number of Episodes: {len(avereward_fast)}', 'Even one episode is not finished yet....')
                    else:
                        print(f'Evaluation on Fast Learner, Average Reward: {np.mean(avereward_fast)}, Number of Episodes: {len(avereward_fast)}, all: ',[arr.mean() for arr in avereward_fast ])
                else:
                    print(f"No Evaluation on Fast Learner")

                epi_return_meta = 0
                avereward_meta = []
                max_step = 0

                if Num_detection_meta > 0:
                    currnt_obs, _ = envs.reset(seed=args.seed)  # next_obs: (8, 4, 84, 84) for Freeway-v5
                    currnt_obs = torch.Tensor(currnt_obs).to(device)


                action, logprob, _, value = agent.get_action_and_value(next_obs / 255.0)
                next_obs, reward, terminations, truncations, infos = envs.step(
                    action.cpu().numpy()
                )
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = np.logical_or(terminations, truncations)
                reward = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                ##### evaluation on the meta policy
                for step_small in range(Num_detection_meta):
                    action, logprob, _, value = Meta_agent.get_action_and_value(currnt_obs / 255.0)
                    next_obs, reward, terminations, truncations, infos = envs.step(
                        action.cpu().numpy()
                    )
                    next_obs = torch.Tensor(next_obs).to(device)
                    epi_return_meta += reward  # for plotting the learning curves
                    currnt_obs = next_obs
                    done = np.logical_or(terminations, truncations)
                    step += 1
                    max_step += 1
                    if done.all() or max_step > MAX_STEP:
                        currnt_obs, _ = envs.reset(seed=args.seed)  # next_obs: (8, 4, 84, 84) for Freeway-v5
                        currnt_obs = torch.Tensor(currnt_obs).to(device)
                        avg_return = 0.99 * avg_return + 0.01 * epi_return_meta
                        avereward_meta.append(epi_return_meta)
                        epi_return_meta = 0
                        max_step = 0

                    # returns_array[step] = copy.copy(avg_return)
                    # pbar.update(1)

                global_step += step

                if Num_detection_meta > 0:
                    if len(avereward_meta) == 0:
                        print(f'Evaluation on Meta Learner, Number of Episodes: {len(avereward_meta)}','Even one episode is not finished yet....')
                    else:
                        print(f'Evaluation on Meta Learner, Average Reward: {np.mean(avereward_meta)}, Number of Episodes: {len(avereward_meta)}, all: ',avereward_meta)
                else:
                    print(f"No Evaluation on Meta Learner")

                # if one eposide is not finished, the initialization is very poor, just reset
                Avereward_meta = -1000 if len(avereward_meta) == 0 else np.mean(avereward_meta)
                Avereward_fast = -1000 if len(avereward_fast) == 0 else np.mean(avereward_fast)

                _, _, _, value_rand = random_agent.get_action_and_value(currnt_obs / 255.0)
                # _, value_fast = get_action_detection(cs_initial, Fast_Learner)
                Avereward_rand = float(value_rand.detach().cpu().numpy().mean())
                print('Reward meta', round(Avereward_meta, 2), 'Reward_fast', round(Avereward_fast, 2), 'Reward_random',round(Avereward_rand, 2))

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
                # if Avereward_meta > Avereward_fast and Avereward_meta > Avereward_rand:
                if Meta_Fast and Avereward_meta > Avereward_rand:
                    # meta initialization is better than the others
                    print('##################### Step 2: Use Meta Initialization and Start Training !')

                    META_WARMUP = 1  # step >= switch * 2,
                    Flag_Reg.append('Meta')
                # elif Avereward_fast >= Avereward_meta and Avereward_fast > Avereward_rand:
                elif Fast_Meta and Avereward_fast > Avereward_rand:
                    Flag_Reg.append('Fast')
                    print('##################### Step 2: Use Fast Initialization / FineTune Fast Learner!')
                else: # random initialization is better than the others
                    agent = FAMEAgent(envs, fast=True).to(device)
                    optimizer = optim.Adam(agent.parameters(), lr=args.lr_fast, eps=1e-5)
                    Flag_Reg.append('Random')
                    print('##################### Step 2: Use Random Initialization')


            ######## to guarantee the total number of steps is args.total_timesteps
            # if args.model_type == "FAME" and mode > 0:
            #     if step > args.batch_size:
            #         step = step - args.batch_size #
            #         print(f'Detection takes step {step}, continue')
            #         if iteration + 1 < args.num_iterations: # if not, we need to update the meta policy
            #             continue # sacrifice the detection step to update the policy



            ###### Step 2: train agent (fast learner with behavior cloning for FAME) ######

            if args.model_type == "FAME":
                if META_WARMUP == 1 and args.lambda_reg > 0 and (iteration < warmstep):
                    if iteration == warmstep - 1:
                        print(f'Use the behavior cloning as an regularization at the step {iteration}')
                    train_agent(reg=Meta_agent, start_time=start_time, writer=writer, next_obs=next_obs, next_done=next_done)
                else:
                    train_agent(reg=None,  start_time=start_time, writer=writer, next_obs=next_obs, next_done=next_done)
            else:
                train_agent(reg=None, start_time=start_time, writer=writer, next_obs=next_obs, next_done=next_done)

            ###### Step 3 (FAME): the last step in the current env: knowledge distillation
            if iteration == args.num_iterations and args.model_type == "FAME":
                if mode == 0:
                    print('First time: No need to update Meta learner')
                else:
                    print('##################### Step 3: Updating Meta Learner!')
                    print('Old Meta data set: ', exp_replay_meta.size(), 'fast data set: ', exp_replay_fast2meta.size())

                    # should reset the optimizer with the initial lr as args.lr1
                    Meta_opt = optim.Adam(Meta_agent.parameters(), lr=args.lr_meta, eps=1e-5)
                    Meta_scheduler = ExponentialLR(Meta_opt, gamma=0.95)
                    train_meta()  # Fast_learner + copied old k-1 Meta_learner -> new k meta learner

                exp_replay_fast2meta.copy_to(exp_replay_meta)
                exp_replay_fast2meta.delete()  # clear the memory
                print('##################### Step 4: Fast2Meta Copy to Meta buffer: New Meta data set: ', exp_replay_meta.size(), 'fast data set: ', exp_replay_fast2meta.size())

                # exp_replay_fast.delete()




        envs.close()
        writer.close()


        if args.save_dir is not None:
            print(f"Saving trained agent in `{args.save_dir}` with name `{run_name}`")
            print(f"{args.save_dir}/{run_name}")
            agent.save(dirname=f"{args.save_dir}/{run_name}") # e.g, ALE-Freeway-v5_{mode}__packnet__run_ppo__{seed}
            if args.model_type == "FAME":
                Meta_agent.save(dirname=f"{args.save_dir}/{run_name}")
