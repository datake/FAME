import numpy as np

single_tasks = [
    "hammer-v2",
    "push-wall-v2",
    "faucet-close-v2",
    "push-back-v2",
    "stick-pull-v2",
    "handle-press-side-v2",
    "push-v2",
    "shelf-place-v2",
    "window-close-v2",
    "peg-unplug-side-v2",
]

tasks = single_tasks + single_tasks

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
             ['button-press-v2', 'plate-slide-back-side-v2', 'window-close-v2'],
             ['hammer-v2','push-wall-v2','faucet-close-v2','push-back-v2','stick-pull-v2','handle-press-side-v2','push-v2','shelf-place-v2','window-close-v2','peg-unplug-side-v2'],]


def get_task_name(task_id):
    return tasks[task_id]


def get_task(task_id, task_sequence, render=False):
    # from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
    # name = tasks[task_id] + "-goal-observable"
    # env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[name]

    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_HIDDEN
    # name = tasks[task_id] + "-goal-hidden"
    name = RPO10_SEQ[task_sequence-1][task_id] + "-goal-hidden"
    # print(f"Loading environment: {name}")
    # name = tasks[task_id] + "-goal-hidden"
    env_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[name]

    env = env_cls(seed=np.random.randint(0, 1024))

    if render:
        env.render_mode = "human"
    env._freeze_rand_vec = False

    return env


if __name__ == "__main__":
    env = get_task(0, render=True)

    for _ in range(200):
        obs, _ = env.reset()  # reset environment
        a = env.action_space.sample()  # sample an action

        # step the environment with the sampled random action
        obs, reward, terminated, truncated, info = env.step(a)

        if terminated:
            break
