from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
from glob import glob
import numpy as np
from tqdm import tqdm

method = 'FAME' # FAME, packnet
# method = 'packnet' # FAME, packnet
Game = 'Freeway'
# Game = 'SpaceInvaders'
N_modes = 8 if Game == 'Freeway' else 10
N_seed = 3

LOOP_seed = True
Average = True

FAME = True
FAME_path = '_FAME' if FAME else ''

if LOOP_seed:
    """load the existing data, add one column of returns for the FAME method or change the existing column for the packnet method"""
    for seed in range(1, N_seed+1):
        for mode in range(N_modes):
            event_path = f'runs/ALE-{Game}-v5_{mode}__{method}__run_ppo{FAME_path}__{seed}'
            print(event_path)
            event_files = glob(os.path.join(event_path, "**", "events.out.tfevents.*"), recursive=True)[0]
            ea = event_accumulator.EventAccumulator(event_path)
            ea.Reload()

            scalars = ea.Tags()["scalars"]
            all_data = []
            for tag in scalars:
                for event in ea.Scalars(tag):
                    all_data.append((event.step, tag, event.value))
            df = pd.DataFrame(all_data, columns=["step", "tag", "value"])
            df_return = df[df['tag']=='charts/episodic_return']
            x = df_return['step'].values
            y = df_return['value'].values
            # data_original = pd.read_csv(f'data/envs/{Game}/task_{mode}.csv')
            data_original = pd.read_csv(f'data_FAME/envs/{Game}/task_{mode}.csv')
            column_name = f'model_type: {method} (Mode {mode}) - charts/episodic_return'
            x_shared = data_original['global_step'].values
            y0 = np.interp(x_shared, x, y)
            data_original[column_name] = y0
            data_original.to_csv(f"data_FAME/envs/{Game}/task_{mode}{FAME_path}_seed{seed}.csv", index=False)
            # print("Saved csv")

if Average:
    # average across seed
    for mode in tqdm(range(N_modes)):
        # data0 = pd.read_csv(f'data_FAME/envs/{Game}/task_{mode}{FAME_path}_seed1.csv')
        data0 = pd.read_csv(f'data_FAME/envs/{Game}/task_{mode}.csv')
        column_name = f'model_type: {method} (Mode {mode}) - charts/episodic_return'
        l_return = []
        for seed in range(1, N_seed + 1):
            data = pd.read_csv(f'data_FAME/envs/{Game}/task_{mode}{FAME_path}_seed{seed}.csv')[column_name].values
            l_return.append(data)
        l_return = np.array(l_return).mean(0) # average over seeds
        data0[column_name] = l_return
        data0.to_csv(f"data_FAME/envs/{Game}/task_{mode}{FAME_path}.csv", index=False)
