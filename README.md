# FAME
Official Implementation of Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning (ICLR 2026))

# MiniAar: we adapte the implementation from https://github.com/NishanthVAnand/prediction-and-control-in-continual-reinforcement-learning/tree/main  (NeurIPS 2021)


## baseline methods:
```
python PT_DQN_half.py --lr1=1e-8 --lr2=1e-4 --decay=0.75 --seed=0 --save --save-model --seq 3 --gpu 1 
python DQN.py --lr1=1e-5 --seed=0 --save --save-model --seq 3 --reset 0 --gpu 0 
python DQN_multi_task.py --lr1=1e-5 --seed=0 --save --save-model --seq 3 --gpu 0
python DQN_large_buffer.py --lr1=1e-4 --seed=0 --save --save-model --seq 3 --gpu 0 
python random_act.py --save --seed=$s --seq 3
```

## Our FAME approach

### seq: the number of sequece of tasks
### detection_step: policy evaluation step
### size_fast2meta: weight estimation step
### warmstep: the number of warmstep 
### lambda_reg: regularization hyperparameter

```
python CF.py --lr1=1e-3 --lr2=1e-5 --size_fast2meta 12000 --detection_step 1200 --seed=1 --save --save-model --seq 1 --gpu 0 --policy 1 --warmstep 50000  --lambda_reg 1.0 
```

# MetaWorld Experiments

## Installation

To set up the environment, please run the following command:

```bash
pip install -r Metaworld/requirements.txt
```

**Note:** The requirements file includes `mujoco` and `metaworld`. Please ensure you have the necessary system dependencies for MuJoCo installed.


## 1. Running FAME & Standard Baselines

To run the main experiments using FAME and standard baselines (Reset, Average, Finetune), use the following command:

```bash
python test_main.py --seed 0 --method buffer --gpu 0 --store_traj_num 10 --use_ttest 1 --env metaworld_sequence_set18
```

### Arguments Reference

| Argument | Value | Description |
| :--- | :--- | :--- |
| `--method` | `buffer` | **FAME-KL** (Our method) |
| | `buffer_wd` | **FAME-MD** (Our method variant) |
| | `independent` | **Reset** (Baseline: Train from scratch) |
| | `average` | **Average** (Baseline: Parameter averaging) |
| | `continue` | **Finetune** (Baseline: Continual learning without regularization) |
| `--env` | `metaworld_sequence_set6` | Sequence of 6 tasks |
| | `metaworld_sequence_set12` | Sequence of 12 tasks |
| | `metaworld_sequence_set18` | Sequence of 18 tasks |
| | `metaworld_sequence_set22` | Sequence of CW10 |

---

## 2. Running Advanced Baselines (PackNet, ProgressiveNet, CompoNet)

These baselines are located in a separate directory.

### Prerequisites
**Important:** You must first run the `simple` algorithm (SAC) to generate the initial model for the first task, which is required by other baselines.

### Execution Steps

1. Navigate to the experiment directory:
   ```bash
   cd Metaworld/baselines_packnet_progressivenet_componet/experiments/meta-world
   ```

2. Run the experiment:
   ```bash
   python run_experiments.py --algorithm simple --seed 0 --start-mode 0 --task-sequence 6
   ```

### Arguments Reference

| Argument | Value | Description |
| :--- | :--- | :--- |
| `--algorithm` | `simple` | **SAC** (Standard Soft Actor-Critic) |
| | `packnet` | **PackNet** |
| | `prognet` | **ProgressiveNet** |
| | `componet` | **CompoNet** |

