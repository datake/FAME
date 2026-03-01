#!/bin/bash


seq=0 # seq number: 0-9
seeds=(1 2 3)
for s in ${seeds[*]};
do

 python DQN.py --lr1=1e-5 --seed=$s --save --save-model --seq $seq --reset 1 --gpu 0 &
 python DQN.py --lr1=1e-5 --seed=$s --save --save-model --seq $seq --reset 0 --gpu 0 &
 python DQN_multi_task.py --lr1=1e-5 --seed=$s --save --save-model --seq $seq --gpu 0 &
 python DQN_large_buffer.py --lr1=1e-4 --seed=$s --save --save-model --seq $seq --gpu 0 &
 python PT_DQN_half.py --lr1=1e-8 --lr2=1e-4 --decay=0.75 --seed=$s --save --save-model --seq $seq --gpu 0 &
 python FAME.py --lr1=1e-3 --lr2=1e-5 --size_fast2meta 12000 --detection_step 600 --seed=$s --save --save-model --seq $seq --warmstep 50000  --lambda_reg 1.0 --gpu 0 &

 wait
done

echo "Running job on $(hostname) at $(date)"

