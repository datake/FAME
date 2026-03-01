import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
matplotlib.use('TkAgg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

## Step 1: change CF_step_normalized in plot_minatar_random.py to save excel file across each time

## step 2: plot the figure with mean and SE

T = np.arange(2, 8)  # T = 1 to 7
# methods = ["DQN-Reset", 'DQN-Finetune',   "DQN-MultiHead", "DQN-LargeBuffer", "PT-DQN-0.5x", "FAME(Ours)"]
methods = ["DQN-Reset", 'Finetune',   "MultiHead", "LargeBuffer", "PT-DQN", "FAME"]
colors = ['g', 'pink',  'brown', 'black', 'blue', 'red']
Games = ["breakout", "space_invaders", "freeway"]
Games_title = ["Breakout", "Spaceinvaders", "Freeway"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for subplot_idx, game in enumerate(Games):
    ax = axes[subplot_idx]
    ax.set_title(Games_title[subplot_idx], fontsize=24)
    ax.set_xlabel("Task (K)", fontsize=20)
    ax.set_ylabel("Average Performance", fontsize=24)
    ax.set_xticks(T)
    # ax.set_ylim(0.3, 1.0)
    ax.grid(True, linestyle='--', alpha=0.3)
    all_means = []
    all_stds = []
    for t in range(1, 7):
        ############# important: if there is no meta learner in step 1, the performance would be worse
        df_mean = pd.read_excel(f"MinAtar_Ave_T{t + 1}_mean.xlsx")
        df_std = pd.read_excel(f"MinAtar_Ave_T{t + 1}_std.xlsx")
        all_means.append(df_mean[game].values)
        print(f"t: {t+1}, {len(all_means)} ")
        all_stds.append(df_std[game].values)

    all_means = np.array(all_means).T  # shape: [num_methods, 7] ???
    all_stds = np.array(all_stds).T

    for i, method in enumerate(methods):
        mean = all_means[i]
        std = all_stds[i]
        # ax.plot(T, mean, label=method, color=colors[i], linewidth=2)
        # ax.plot(T, mean, label=method, color=colors[i], linewidth=2)
        ax.errorbar(
            T, mean, yerr=std,
            label=method,
            color=colors[i],
            fmt='-o',
            capsize=4,
            linewidth=2, markersize=5
        )
        # ax.fill_between(T, mean - std, mean + std, color=colors[i], alpha=0.2)

ax.legend(loc='lower center', frameon=False, fontsize=16, bbox_to_anchor=(-0.7, -0.35, 0.0, 0.0), ncol=6)
plt.tight_layout()

plt.show()