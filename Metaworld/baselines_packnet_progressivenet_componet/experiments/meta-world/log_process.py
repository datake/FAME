import os
import pandas as pd
from collections import defaultdict

path = '/Users/hongmingzhang/wakaka/2025/ContinualRL/FAME_componet/experiments/meta-world/log/'

# path_1 = '/Users/hongmingzhang/wakaka/2025/ContinualRL/FAME_componet/experiments/meta-world/log/sac_metaworld_sequence_set6_4_componet.csv'
# path_2 = '/Users/hongmingzhang/wakaka/2025/ContinualRL/FAME_componet/experiments/meta-world/log/sac_metaworld_sequence_set6_0_simple.csv'
# data_1 = pd.read_csv(path_1)
# data_2 = pd.read_csv(path_2)
# data_2.head(40)
# combined_data = pd.concat([data_2.head(40), data_1], ignore_index=True)
# combined_data['method'] = 'componet'
# combined_data.to_csv(path_2, index=False)

steps = [975000, 1975000, 2975000, 3975000, 4975000, 5975000, 6975000, 7975000, 8975000, 9975000]
for dirpath, dirnames, filenames in os.walk(path):
    for filename in filenames:
        # if 'sac_metaworld_sequence_set6_0_componet.csv' in filename:
        if 'packnet' in filename and 'final' not in filename: # packnet, prognet, componet
            print(os.path.join(dirpath, filename))
            data_1 = pd.read_csv(os.path.join(dirpath, filename))
            ############### only for componet #####################################
            if 1 not in data_1['task_idx'].values:
                path_1_simple = filename.split('componet.csv')[0] + 'simple.csv'
                data_1_simple = pd.read_csv(os.path.join(dirpath, path_1_simple))
                data_1 = pd.concat([data_1_simple.head(40), data_1], ignore_index=True)
                data_1['method'] = 'componet'
                data_1.to_csv(os.path.join(dirpath, filename), index=False)
            #######################################################################
            intermediate_stats = defaultdict(list)

            for last_ind in range(len(steps)):
                for first_ind in range(last_ind + 1):
                    intermediate_stats['mean_return'].append(data_1.loc[data_1['steps'] == steps[first_ind], 'mean_return'].item())
                    intermediate_stats['mean_success'].append(data_1.loc[data_1['steps'] == steps[first_ind], 'mean_success'].item())
                    intermediate_stats['task'].append(data_1.loc[data_1['steps'] == steps[first_ind], 'task'].item())
                    intermediate_stats['task_idx'].append(data_1.loc[data_1['steps'] == steps[first_ind], 'task_idx'].item())
                    intermediate_stats['seed'].append(data_1.loc[data_1['steps'] == steps[first_ind], 'seed'].item())
                    intermediate_stats['method'].append(data_1.loc[data_1['steps'] == steps[first_ind], 'method'].item())
                    intermediate_stats['agent_idx'].append(last_ind+1)

            intermediate_stats = pd.DataFrame(intermediate_stats)
            intermediate_stats.to_csv(os.path.join(dirpath, filename.split('.csv')[0] + '_final.csv'), index=False)


