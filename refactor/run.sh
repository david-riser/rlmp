#!/bin/bash

source activate pytorch_p36
python train.py --n_step=1 --env=MsPacmanNoFrameskip-v0 --lr=0.0001 --eps_decay=500000 --eps_end=0.05 --beta_frames=100000 --batch_size=32 --save_buffer=space_buffer.pkl --update_interval=1000 --eval_frequency=5000 --n_batches_per_epoch=500 --n_epochs=4000 --n_eval_games=10
#python train.py --n_step=1 --env=SpaceInvadersNoFrameskip-v0 --lr=0.0001 --eps_decay=0 --eps_end=0.05 --beta_frames=100000 --batch_size=32 --expert_buffer=space_buffer.pkl --expert_batch_size=32 --pretrain_steps=10000 --update_interval=1000 --eval_frequency=500 --n_batches_per_epoch=500 --n_epochs=1000 --n_eval_games=10 --use_bandit
#python train.py --n_step=1 --env=SpaceInvadersNoFrameskip-v0 --lr=0.0001 --eps_decay=0 --eps_end=0.05 --beta_frames=100000 --batch_size=32 --expert_buffer=space_buffer.pkl --expert_batch_size=32 --pretrain_steps=10000 --update_interval=1000 --eval_frequency=500 --n_batches_per_epoch=500 --n_epochs=1000 --n_eval_games=10 --use_bandit --bandit_eps=0.4 --bandit_alpha=0.1
sudo shutdown now -h 
