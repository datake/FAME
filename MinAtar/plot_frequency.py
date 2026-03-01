import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ast
import seaborn as sns
import pandas as pd
matplotlib.use('TkAgg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
sns.set_style("whitegrid")

new = {'Random':0, 'Meta':0, 'Fast':0} # evaluate the frequency of using meta intialization
old = {'Random':0, 'Meta':0, 'Fast':0} # evaluate the frequency of using meta intialization

for seq in range(10):
    for seed in range(3):
        with open(f"Reg1.0_CF{seq}_seed{seed+1}.txt", "r") as f:
            for line in f:
                if line.startswith("Regularization:"):
                    reg_line = line.strip().split("Regularization:")[1].strip()
                    regularization = ast.literal_eval(reg_line[:reg_line.find('2025')])
                elif line.startswith("Games:"):
                    games_line = line.strip().split("Games:")[1].strip()
                    games = ast.literal_eval(games_line[:games_line.find('2025')])

        # evaluate the frequency!
        game_set = set([games[0]])
        for i in range(1, len(games)):
            if games[i] in game_set and (games[i-1] != games[i]):
                old[regularization[i-1]] += 1
            elif games[i] not in game_set:
                print(games[i])
                new[regularization[i-1]] += 1
                game_set.add(games[i])

        # print("Regularization:", regularization)
        # print("Games:", games)
        # print(1)
print(new)
print(old)

def compute_percent(d):
    total = sum(d.values())
    return {k: v * 100 / total for k, v in d.items()}

new = compute_percent(new)
old = compute_percent(old)
df = pd.DataFrame([
    {'Environment': 'New Environment', 'Warm-Up': k, 'Percentage': v} for k, v in new.items()
] + [
    {'Environment': 'Old Environment', 'Warm-Up': k, 'Percentage': v} for k, v in old.items()
])

sns.set(style="whitegrid", context='talk', font_scale=1.0)

custom_palette = ['#1f77b4', '#2ca02c', '#ff7f0e']

plt.figure(figsize=(7, 5))
ax = sns.barplot(
    data=df,
    hue='Warm-Up',
    y='Percentage',
    x='Environment',
    # palette='Set2'
    palette=custom_palette,
)

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=20)

ax.set_ylabel('Percentage (%)', fontsize=24)
ax.set_xlabel('', fontsize=24)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=22)
ax.set_ylim(0, 110)
ax.set_title('Warm-Up Selection Ratio (%)', fontsize=28)
ax.legend(title='', loc='upper center', frameon=False, fontsize=24)
plt.tight_layout()
plt.show()





