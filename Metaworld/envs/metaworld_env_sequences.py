###
# This files just contains the sequence of envs to be used
#
#
### o easy, x hard, - medium

GOOD_RPO_ENVS = [
    "door-close-v2", # o
    "sweep-into-v2", # -
    "coffee-button-v2", # o
    "window-open-v2", # o
    "reach-wall-v2", # o
    "drawer-close-v2", # o
    "button-press-v2", # o
    "plate-slide-back-side-v2", # o
    "reach-v2", # -
    "plate-slide-side-v2", # o
    "coffee-push-v2", # -
    "plate-slide-back-v2", # o
    "soccer-v2", # x
    "window-close-v2", # o
    "handle-pull-side-v2", # o
    "hand-insert-v2", # -
    "door-lock-v2", # o
    "push-v2", # -
    "peg-unplug-side-v2" # -
]

## Removing the following tasks since they seem too difficult
# 'lever_pull', 'peg-insert-side', 'button-press-topdown', 'button-press-wall',
# 'stick-push', 'coffee-pull', 'push-wall'
## Removing more
# "button-press-topdown-wall", "plate-slide-v2", "handle-press-side", "handle-press"
# currently 20 left


# randomly generated below
# we keep these fixed so the agents have a fair comparison

RPO10_SHORT = [['door-close-v2','coffee-button-v2']]

RPO10_SEQ = [['handle-pull-side-v2', 'peg-unplug-side-v2', 'coffee-push-v2', 'soccer-v2', 'drawer-close-v2', 'reach-wall-v2', 'plate-slide-back-v2', 'window-open-v2', 'plate-slide-side-v2', 'plate-slide-back-side-v2'],
             ['window-close-v2', 'window-open-v2', 'hand-insert-v2', 'door-lock-v2', 'reach-v2', 'button-press-v2', 'sweep-into-v2', 'coffee-button-v2', 'door-close-v2', 'push-v2'],
             ['window-close-v2', 'reach-wall-v2', 'sweep-into-v2', 'reach-v2', 'soccer-v2', 'coffee-push-v2', 'plate-slide-side-v2', 'drawer-close-v2', 'hand-insert-v2', 'door-close-v2'],
             ['plate-slide-back-v2', 'reach-wall-v2', 'door-lock-v2', 'peg-unplug-side-v2', 'push-v2', 'button-press-v2', 'plate-slide-back-side-v2', 'coffee-push-v2', 'coffee-button-v2', 'handle-pull-side-v2'],
             ['push-v2', 'coffee-button-v2', 'sweep-into-v2', 'door-close-v2', 'drawer-close-v2', 'soccer-v2', 'peg-unplug-side-v2', 'hand-insert-v2', 'door-lock-v2', 'reach-v2'],
             ['button-press-v2', 'plate-slide-back-side-v2', 'window-close-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2', 'plate-slide-back-v2', 'coffee-button-v2', 'window-open-v2', 'handle-pull-side-v2', 'door-close-v2'],
             ['push-v2', 'button-press-v2', 'plate-slide-back-v2', 'drawer-close-v2', 'soccer-v2', 'plate-slide-side-v2', 'reach-wall-v2', 'coffee-push-v2', 'window-close-v2', 'door-lock-v2'],
             ['plate-slide-side-v2', 'hand-insert-v2', 'handle-pull-side-v2', 'plate-slide-back-side-v2', 'window-open-v2', 'sweep-into-v2', 'reach-wall-v2', 'reach-v2', 'soccer-v2', 'peg-unplug-side-v2'],
             ['hand-insert-v2', 'reach-v2', 'window-close-v2', 'drawer-close-v2', 'window-open-v2', 'coffee-button-v2', 'plate-slide-back-v2', 'coffee-push-v2', 'push-v2', 'plate-slide-back-side-v2'],
             ['sweep-into-v2', 'peg-unplug-side-v2', 'window-close-v2', 'door-lock-v2', 'hand-insert-v2', 'handle-pull-side-v2', 'window-open-v2', 'door-close-v2', 'button-press-v2', 'reach-wall-v2'],
             ['reach-v2', 'door-lock-v2', 'sweep-into-v2', 'push-v2', 'button-press-v2', 'coffee-push-v2', 'handle-pull-side-v2', 'plate-slide-side-v2', 'door-close-v2', 'drawer-close-v2'],
             ['plate-slide-back-side-v2', 'soccer-v2', 'sweep-into-v2', 'handle-pull-side-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2', 'door-lock-v2', 'reach-v2', 'plate-slide-back-v2', 'coffee-button-v2'],
             ['reach-wall-v2', 'plate-slide-back-v2', 'drawer-close-v2', 'hand-insert-v2', 'coffee-push-v2', 'coffee-button-v2', 'window-close-v2', 'plate-slide-back-side-v2', 'door-close-v2', 'button-press-v2'],
             ['soccer-v2', 'drawer-close-v2', 'push-v2', 'sweep-into-v2', 'window-open-v2', 'reach-wall-v2', 'door-lock-v2', 'window-close-v2', 'reach-v2', 'hand-insert-v2'],
             ['plate-slide-back-v2', 'plate-slide-side-v2', 'door-close-v2', 'push-v2', 'peg-unplug-side-v2', 'plate-slide-back-side-v2', 'coffee-push-v2', 'coffee-button-v2', 'button-press-v2', 'soccer-v2'],
             ['hand-insert-v2', 'coffee-button-v2', 'soccer-v2', 'window-open-v2', 'push-v2', 'reach-v2', 'drawer-close-v2', 'handle-pull-side-v2', 'door-lock-v2', 'plate-slide-back-side-v2'],
             ['coffee-push-v2', 'door-close-v2', 'handle-pull-side-v2', 'window-close-v2', 'plate-slide-back-v2', 'reach-wall-v2', 'sweep-into-v2', 'window-open-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2'],
             ['coffee-push-v2', 'button-press-v2', 'reach-v2', 'peg-unplug-side-v2', 'reach-wall-v2', 'door-close-v2', 'window-open-v2', 'handle-pull-side-v2', 'plate-slide-back-side-v2', 'soccer-v2'],
             ['sweep-into-v2', 'plate-slide-side-v2', 'button-press-v2', 'drawer-close-v2', 'push-v2', 'coffee-button-v2', 'door-lock-v2', 'hand-insert-v2', 'plate-slide-back-v2', 'window-close-v2'],
             ['reach-v2', 'button-press-v2', 'plate-slide-side-v2', 'door-close-v2', 'plate-slide-back-side-v2', 'plate-slide-back-v2', 'coffee-button-v2', 'sweep-into-v2', 'reach-wall-v2', 'drawer-close-v2'],
             ['button-press-v2', 'plate-slide-back-side-v2', 'window-close-v2'],]


RPO10_SEQ_OLD = [['plate-slide-back-v2', 'button-press-v2', 'handle-pull-side-v2', 'peg-unplug-side-v2', 'window-open-v2', 'handle-pull-v2', 'plate-slide-back-side-v2', 'coffee-push-v2', 'push-v2', 'reach-v2'],
             ['door-close-v2', 'hand-insert-v2', 'window-close-v2', 'sweep-into-v2', 'drawer-close-v2', 'plate-slide-side-v2', 'soccer-v2', 'reach-wall-v2', 'door-lock-v2', 'coffee-button-v2'],
             ['sweep-into-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2', 'window-open-v2', 'door-lock-v2', 'reach-v2', 'reach-wall-v2', 'plate-slide-back-v2', 'door-close-v2', 'drawer-close-v2'],
             ['handle-pull-v2', 'soccer-v2', 'handle-pull-side-v2', 'button-press-v2', 'push-v2', 'window-close-v2', 'coffee-button-v2', 'hand-insert-v2', 'plate-slide-back-side-v2', 'coffee-push-v2'],
             ['window-open-v2', 'push-v2', 'reach-v2', 'plate-slide-back-v2', 'handle-pull-side-v2', 'soccer-v2', 'sweep-into-v2', 'coffee-button-v2', 'window-close-v2', 'coffee-push-v2'],
             ['drawer-close-v2', 'peg-unplug-side-v2', 'plate-slide-side-v2', 'door-lock-v2', 'plate-slide-back-side-v2', 'reach-wall-v2', 'handle-pull-v2', 'button-press-v2', 'hand-insert-v2', 'door-close-v2'],
             ['door-lock-v2', 'coffee-button-v2', 'hand-insert-v2', 'window-close-v2', 'button-press-v2', 'plate-slide-back-side-v2', 'plate-slide-back-v2', 'window-open-v2', 'reach-wall-v2', 'soccer-v2'],
             ['drawer-close-v2', 'handle-pull-v2', 'push-v2', 'plate-slide-side-v2', 'coffee-push-v2', 'handle-pull-side-v2', 'sweep-into-v2', 'reach-v2', 'peg-unplug-side-v2', 'door-close-v2'],
             ['hand-insert-v2', 'door-lock-v2', 'window-close-v2', 'reach-wall-v2', 'coffee-push-v2', 'drawer-close-v2', 'plate-slide-back-v2', 'handle-pull-v2', 'button-press-v2', 'sweep-into-v2'],
             ['reach-v2', 'push-v2', 'plate-slide-back-side-v2', 'plate-slide-side-v2', 'door-close-v2', 'soccer-v2', 'peg-unplug-side-v2', 'window-open-v2', 'handle-pull-side-v2', 'coffee-button-v2']]


RPO20_SEQ = [
['door-lock-v2', 'handle-press-v2', 'handle-press-side-v2', 'button-press-v2', 'door-close-v2', 'hand-insert-v2', 'reach-v2', 'plate-slide-v2', 'handle-pull-side-v2', 'plate-slide-back-side-v2', 'plate-slide-back-v2', 'soccer-v2', 'sweep-into-v2', 'reach-wall-v2', 'window-open-v2', 'coffee-button-v2', 'coffee-push-v2', 'peg-unplug-side-v2', 'window-close-v2', 'plate-slide-side-v2'],
['plate-slide-side-v2', 'plate-slide-back-v2', 'handle-press-side-v2', 'peg-unplug-side-v2', 'handle-pull-v2', 'reach-wall-v2', 'plate-slide-back-side-v2', 'button-press-v2', 'soccer-v2', 'hand-insert-v2', 'door-lock-v2', 'push-v2', 'window-close-v2', 'button-press-topdown-wall-v2', 'drawer-close-v2', 'sweep-into-v2', 'reach-v2', 'coffee-button-v2', 'coffee-push-v2', 'door-close-v2'],
['hand-insert-v2', 'peg-unplug-side-v2', 'handle-pull-side-v2', 'handle-press-v2', 'button-press-v2', 'coffee-push-v2', 'plate-slide-back-v2', 'handle-pull-v2', 'button-press-topdown-wall-v2', 'push-v2', 'plate-slide-v2', 'door-close-v2', 'reach-v2', 'window-open-v2', 'coffee-button-v2', 'window-close-v2', 'drawer-close-v2', 'soccer-v2', 'plate-slide-side-v2', 'plate-slide-back-side-v2'],
['handle-pull-side-v2', 'button-press-v2', 'window-open-v2', 'door-close-v2', 'reach-wall-v2', 'push-v2', 'hand-insert-v2', 'drawer-close-v2', 'handle-press-side-v2', 'handle-press-v2', 'door-lock-v2', 'plate-slide-back-side-v2', 'window-close-v2', 'sweep-into-v2', 'button-press-topdown-wall-v2', 'coffee-button-v2', 'soccer-v2', 'handle-pull-v2', 'plate-slide-back-v2', 'plate-slide-v2'],
['soccer-v2', 'coffee-button-v2', 'handle-press-v2', 'handle-pull-side-v2', 'plate-slide-back-v2', 'door-lock-v2', 'drawer-close-v2', 'reach-v2', 'peg-unplug-side-v2', 'plate-slide-v2', 'reach-wall-v2', 'handle-pull-v2', 'push-v2', 'plate-slide-side-v2', 'coffee-push-v2', 'button-press-topdown-wall-v2', 'hand-insert-v2', 'sweep-into-v2', 'window-open-v2', 'handle-press-side-v2'],
['door-close-v2', 'reach-wall-v2', 'coffee-push-v2', 'sweep-into-v2', 'door-lock-v2', 'plate-slide-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2', 'handle-press-side-v2', 'handle-pull-v2', 'window-close-v2', 'push-v2', 'plate-slide-back-side-v2', 'reach-v2', 'handle-press-v2', 'window-open-v2', 'button-press-topdown-wall-v2', 'handle-pull-side-v2', 'drawer-close-v2', 'button-press-v2'],
['window-open-v2', 'window-close-v2', 'handle-pull-v2', 'push-v2', 'plate-slide-back-v2', 'button-press-v2', 'reach-v2', 'plate-slide-v2', 'coffee-button-v2', 'handle-pull-side-v2', 'hand-insert-v2', 'reach-wall-v2', 'drawer-close-v2', 'plate-slide-back-side-v2', 'sweep-into-v2', 'button-press-topdown-wall-v2', 'door-close-v2', 'plate-slide-side-v2', 'door-lock-v2', 'coffee-push-v2'],
['door-close-v2', 'soccer-v2', 'drawer-close-v2', 'handle-pull-side-v2', 'plate-slide-back-v2', 'hand-insert-v2', 'coffee-push-v2', 'reach-wall-v2', 'peg-unplug-side-v2', 'button-press-topdown-wall-v2', 'plate-slide-side-v2', 'reach-v2', 'window-close-v2', 'sweep-into-v2', 'button-press-v2', 'coffee-button-v2', 'handle-press-side-v2', 'handle-pull-v2', 'window-open-v2', 'handle-press-v2'],
['handle-pull-side-v2', 'push-v2', 'plate-slide-back-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2', 'reach-wall-v2', 'sweep-into-v2', 'door-lock-v2', 'plate-slide-v2', 'window-open-v2', 'handle-press-v2', 'hand-insert-v2', 'handle-press-side-v2', 'handle-pull-v2', 'soccer-v2', 'drawer-close-v2', 'reach-v2', 'button-press-v2', 'window-close-v2', 'plate-slide-back-side-v2'],
['peg-unplug-side-v2', 'handle-press-side-v2', 'reach-wall-v2', 'door-close-v2', 'button-press-topdown-wall-v2', 'reach-v2', 'handle-pull-v2', 'drawer-close-v2', 'plate-slide-side-v2', 'coffee-button-v2', 'window-close-v2', 'handle-press-v2', 'door-lock-v2', 'coffee-push-v2', 'window-open-v2', 'plate-slide-back-side-v2', 'button-press-v2', 'push-v2', 'plate-slide-v2', 'soccer-v2']
]





GOOD_RPO_SEQS = [
    ['sweep-into-v2', 'door-close-v2', 'drawer-close-v2', 'button-press-v2', 'window-close-v2', 'hand-insert-v2', 'soccer-v2', 'handle-pull-side-v2'],
    ['peg-unplug-side-v2', 'soccer-v2', 'plate-slide-side-v2', 'plate-slide-back-side-v2', 'reach-wall-v2', 'reach-v2', 'drawer-close-v2', 'coffee-button-v2'],
    ['window-close-v2', 'window-open-v2', 'button-press-topdown-wall-v2', 'handle-press-v2', 'coffee-button-v2', 'handle-press-side-v2', 'push-v2', 'plate-slide-back-v2'],
    ['plate-slide-v2', 'door-close-v2', 'sweep-into-v2', 'door-lock-v2', 'coffee-push-v2', 'handle-press-v2', 'hand-insert-v2', 'reach-wall-v2'],
    ['coffee-button-v2', 'soccer-v2', 'plate-slide-v2', 'handle-pull-v2', 'window-open-v2', 'handle-press-v2', 'handle-pull-side-v2', 'drawer-close-v2'],
    ['handle-pull-side-v2', 'handle-press-v2', 'window-open-v2', 'handle-pull-v2', 'window-close-v2', 'coffee-button-v2', 'plate-slide-v2', 'coffee-push-v2'],
    ['button-press-v2', 'button-press-topdown-wall-v2', 'handle-press-side-v2', 'reach-wall-v2', 'hand-insert-v2', 'plate-slide-side-v2', 'peg-unplug-side-v2', 'push-v2'],
    ['peg-unplug-side-v2', 'reach-wall-v2', 'plate-slide-side-v2', 'door-close-v2', 'door-lock-v2', 'handle-press-side-v2', 'reach-v2', 'plate-slide-back-side-v2'],
    ['coffee-push-v2', 'handle-pull-v2', 'sweep-into-v2', 'plate-slide-back-v2', 'plate-slide-back-side-v2', 'reach-v2', 'plate-slide-v2', 'window-open-v2'],
    ['button-press-topdown-wall-v2', 'push-v2', 'drawer-close-v2', 'door-lock-v2', 'plate-slide-back-v2', 'door-close-v2', 'sweep-into-v2', 'button-press-v2']
    ]

## OLD LIST 1st attempt
# GOOD_RPO_SEQS = [['plate-slide-back-v2', 'button-press-v2', 'door-close-v2', 'handle-press-v2', 'button-press-topdown-v2', 'plate-slide-v2', 'lever-pull-v2', 'peg-insert-side-v2'],
#                  ['plate-slide-side-v2', 'lever-pull-v2', 'window-close-v2', 'drawer-close-v2', 'soccer-v2', 'sweep-into-v2', 'button-press-wall-v2', 'handle-press-v2'],
#                  ['button-press-wall-v2', 'handle-press-side-v2', 'coffee-push-v2', 'peg-unplug-side-v2', 'soccer-v2', 'button-press-v2', 'button-press-topdown-wall-v2', 'platex`-slide-v2'],
#                  ['reach-v2', 'push-wall-v2', 'coffee-button-v2', 'plate-slide-back-side-v2', 'stick-push-v2', 'plate-slide-v2', 'coffee-pull-v2', 'handle-press-v2'],
#                  ['plate-slide-back-side-v2', 'push-v2', 'peg-insert-side-v2', 'reach-wall-v2', 'sweep-into-v2', 'button-press-wall-v2', 'handle-pull-side-v2', 'coffee-button-v2'],
#                  ['stick-push-v2', 'reach-wall-v2', 'peg-insert-side-v2', 'coffee-pull-v2', 'handle-pull-v2', 'coffee-push-v2', 'door-lock-v2', 'plate-slide-back-side-v2'],
#                  ['window-open-v2', 'peg-unplug-side-v2', 'hand-insert-v2', 'door-lock-v2', 'button-press-v2', 'handle-pull-v2', 'reach-v2', 'handle-press-side-v2'],
#                  ['coffee-button-v2', 'plate-slide-side-v2', 'door-close-v2', 'drawer-close-v2', 'window-close-v2', 'plate-slide-back-v2', 'button-press-topdown-v2', 'button-press-topdown-wall-v2'],
#                  ['reach-v2', 'push-wall-v2', 'drawer-close-v2', 'hand-insert-v2', 'window-open-v2', 'handle-press-side-v2', 'button-press-topdown-v2', 'door-close-v2'],
#                  ['coffee-push-v2', 'window-open-v2', 'sweep-into-v2', 'handle-pull-side-v2', 'reach-wall-v2', 'plate-slide-side-v2', 'push-v2', 'plate-slide-back-v2']]


if __name__ == "__main__":
    # for i, x in enumerate(RPO20_SEQ[9]):
    #     print(i+1, x)
    # quit()

    import random
    import numpy as np
    NUM_TASKS_PER_SEQ = 10
    NUM_SEQ = 20
    total_tasks = NUM_TASKS_PER_SEQ * NUM_SEQ

    #### Let's use stratified sampling, we make sure all tasks are represented as evenly as possible

    ### Method based on choosing tasks first, then adding the unused ones to the next set
    all_seq_tasks = []


    candidate_set = set(GOOD_RPO_ENVS)

    # we assume that the initial candidate set is larger than then number of tasks to sample
    for i_seq in range(NUM_SEQ):
        one_set_tasks = set([])
        num_tasks_to_sample = NUM_TASKS_PER_SEQ

        if len(candidate_set) < NUM_TASKS_PER_SEQ:
            one_set_tasks = one_set_tasks.union(candidate_set)
            num_tasks_to_sample -= len(candidate_set)
            candidate_set = set(GOOD_RPO_ENVS)

        # sample from the candidates ensuring there's no repeats
        sampled_set_tasks = random.sample(list(candidate_set.difference(one_set_tasks)), num_tasks_to_sample)
        one_set_tasks = one_set_tasks.union(sampled_set_tasks)

        # remove the sampled tasks from candidates
        candidate_set = candidate_set.difference(sampled_set_tasks)

        # add to master list of sequences
        all_seq_tasks.append(one_set_tasks)


    # make tasks into lists and shuffle them
    all_seq_tasks = [list(task_set) for task_set in all_seq_tasks]
    for task_lst in all_seq_tasks:
        random.shuffle(task_lst)

    # print them
    print(all_seq_tasks)






    ### Method based on rejection-sampling all the tasks and then regrouping them
    # seq_list = GOOD_RPO_ENVS * (int(total_tasks / len(GOOD_RPO_ENVS)) + 1)
    # seq_list = seq_list[:total_tasks]
    #
    #
    #
    # # printl(seq_list)z
    # import collections
    #
    # def check_duplicate(lst):
    #     counter = collections.Counter(lst)
    #     if counter.most_common(1)[0][1] > 1:  # check count of most common item
    #         return True
    #     return False
    #
    # same_task_twice = True
    #
    # # do rejection sampling until you find an ordering where tasks aren't duplicated in the same sequence
    # i_try = 0
    # while same_task_twice:
    #     same_task_twice = False
    #     random.shuffle(seq_list)
    #     if np.any([check_duplicate(seq_list[i*NUM_TASKS_PER_SEQ:(i+1)*NUM_TASKS_PER_SEQ]) for i in range(NUM_SEQ)]):
    #         same_task_twice = True
    #     i_try += 1
    #     if i_try % 100 == 0:
    #         print(i_try)
    #
    # # print(seq_list)
    #
    # seq_list = [ seq_list[i*NUM_TASKS_PER_SEQ:(i+1)*NUM_TASKS_PER_SEQ] for i in range(NUM_SEQ)]
    # for lst in seq_list:
    #     print(lst)
    #
    # ######
    # # for i in range(10):
    # #     rand_list = random.sample(GOOD_RPO_ENVS, 8)
    # #     seq_list.append(rand_list)
    # # print(seq_list)
    # # len(GOOD_RPO_ENVS)
    #
    # # printl(seq_list)z
    # import collections
    #
    # def check_duplicate(lst):
    #     counter = collections.Counter(lst)
    #     if counter.most_common(1)[0][1] > 1:  # check count of most common item
    #         return True
    #     return False
    #
    # same_task_twice = True
    #
    # # do rejection sampling until you find an ordering where tasks aren't duplicated in the same sequence
    # i_try = 0
    # while same_task_twice:
    #     same_task_twice = False
    #     random.shuffle(seq_list)
    #     if np.any([check_duplicate(seq_list[i*8:(i+1)*8]) for i in range(10)]):
    #         same_task_twice = True
    #     i_try += 1
    #     if i_try % 10000 == 0:
    #         print(i_try)
    #
    # # print(seq_list)
    #
    # seq_list = [ seq_list[i*8:(i+1)*8] for i in range(10)]
    # for lst in seq_list:
    #     print(lst)

    ######
    # for i in range(10):
    #     rand_list = random.sample(GOOD_RPO_ENVS, 8)
    #     seq_list.append(rand_list)
    # print(seq_list)
    #


    ## Checking distribution of tasks
    import collections
    seq_list = RPO10_SEQ

    for lst in seq_list:
        print(len(lst))

    flat_list = [
        x
        for xs in seq_list
        for x in xs
    ]
    # for env in GOOD_RPO_ENVS:
    #     print(env in flat_list, env)

    counter = collections.Counter(flat_list)
    print(counter)