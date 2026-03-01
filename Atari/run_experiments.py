import os.path
import subprocess
import argparse
import random
import time
from torch.optim.lr_scheduler import ExponentialLR
from task_utils import TASKS
from replay import *



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("--algorithm", type=str, choices=["componet", "finetune", "from-scratch", "prog-net", "packnet", 'fame'], default="fame")
    parser.add_argument("--env", type=str,choices=["ALE/SpaceInvaders-v5", "ALE/Freeway-v5"], default="ALE/Freeway-v5")
    parser.add_argument("--seed", type=int, required=False, default=1, help='[1,10]')
    # parser.add_argument("--iter", type=int, required=False, default=1e6)

    parser.add_argument("--start-mode", type=int, required=False, default=0)
    # parser.add_argument("--first-mode", type=int, required=True)
    # parser.add_argument("--last-mode", type=int, required=True)
    # fmt: on
    return parser.parse_args()


args = parse_args()

modes = TASKS[args.env] # list of task ids, e.g., [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] for SpaceInvaders
start_mode = args.start_mode # task id of the first task to run

if args.algorithm == "finetune":
    model_type = "cnn-simple-ft"
elif args.algorithm == "componet":
    model_type = "cnn-componet"
elif args.algorithm == "from-scratch":
    model_type = "cnn-simple"
elif args.algorithm == "prog-net":
    model_type = "prog-net"
elif args.algorithm == "packnet":
    model_type = "packnet"
elif args.algorithm == "fame":
    model_type = "FAME"

seed = random.randint(0, 1e6) if args.seed is None else args.seed

# ALE-SpaceInvaders-v5_{task_id}__cnn-componet__run_ppo__{seed}
run_name = (
    lambda task_id: f"{args.env.replace('/', '-')}_{task_id}__{model_type}__run_ppo__{seed}"
)
# timesteps = int(args.iter)
# timesteps = int(1e6)
timesteps = int(1e4)

first_idx = modes.index(start_mode) # eg., 0
for i, task_id in enumerate(modes[first_idx:]):
    # params = f"--track --model-type={model_type} --env-id={args.env} --seed={seed}"
    params = f"--model-type={model_type} --env-id={args.env} --seed={seed}"
    params += f" --mode={task_id} --save-dir=agents --total-timesteps={timesteps}"

    # algorithm specific CLI arguments
    if args.algorithm == "componet":
        params += " --componet-finetune-encoder"
    if args.algorithm == "packnet":
        params += f" --total-task-num={len(modes)}"

    ###### very important: leveraging the previous models in agengs/***
    if first_idx > 0 or i > 0:
        # multiple previous modules
        if args.algorithm in ["componet", "prog-net"]:
            params += " --prev-units"
            for i in modes[: modes.index(task_id)]: # all previous tasks！！！！！
                params += f" agents/{run_name(i)}"
        # single previous module: the previous agent model
        elif args.algorithm in ["finetune", "packnet"]:
            params += f" --prev-units agents/{run_name(task_id-1)}"

        # Fame:
        if args.algorithm == "fame":
            game = args.env.split("/")[-1].split('-')[-1]  # e.g., SpaceInvaders, Freeway
            params += f" --buffer_path data_FAME/{game}_buffer_" # + meta/fast/fast2meta.pkl



    # Launch experiment
    print(f"\nRunning task {task_id} ({i + 1}/{len(modes) - first_idx})", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    cmd = f"python3 run_ppo.py {params}"
    print(cmd)
    """
    packnet: python3 run_ppo.py --model-type=packnet --env-id=ALE/Freeway-v5 --seed=303280 --mode=0 --save-dir=agents --total-timesteps=1000000 --total-task-num=8
    componet: python3 run_ppo.py  --model-type=cnn-componet --env-id=ALE/Freeway-v5 --seed=1 --mode=0 --save-dir=agents --total-timesteps=1000000 --componet-finetune-encoder
    fame: python3 run_ppo.py  --model-type=FAME --env-id=ALE/Freeway-v5 --seed=1 --mode=0 --save-dir=agents --total-timesteps=1000000
    """
    #
    res = subprocess.run(cmd.split(" "))
    if res.returncode != 0:
        print(f"*** Process returned code {res.returncode}. Stopping on error.")
        quit(1)
