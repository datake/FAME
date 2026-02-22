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

# MetaWorld: the code is based on "Parseval Regularization for Continual Reinforcement Learning"


## How to Run
```
python test_main.py --seed 0 --method buffer --gpu 0 --store_traj_num 10 --use_ttest 1 --env metaworld_sequence_set18
```

## Parameters Meaning

### --method
- buffer: FAME-KL
- buffer_wd: FAME-MD
- independent: Reset
- average: Average
- continue: Finetune

### --env
- metaworld_sequence_set6 / metaworld_sequence_set12 / metaworld_sequence_set18

### Baselines: PackNet, ProgressiveNet and CompoNet

cd to the baselines' code directory (Metaworld/baselines_packnet_progressivenet_componet/experiments/meta-world) and run the following command to run the baselines. Note that you need to run the code with --algorithm simple first to get the first task's model, which will be used for the other baselines.

```
python run_experiments.py --algorithm simple --seed 0 --start-mode 0 --task-sequence 6
```

## Parameters Meaning
### --algorithm

- simple: SAC
- packnet: PackNet
- prognet: ProgressiveNet
- componet: CompoNet

