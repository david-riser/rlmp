#!/bin/bash

source activate pytorch_p36
#python train.py --n_step=1 --env=PongNoFrameskip-v0 --lr=0.0001 --eps_decay=0 --eps_end=0.05 --beta_frames=100000 --batch_size=32 --save_buffer=pong_buffer.pkl --update_interval=1000 --eval_frequency=500 --n_batches_per_epoch=500 --n_epochs=2000 --n_eval_games=4
#python train.py --n_step=1 --env=PongNoFrameskip-v0 --lr=0.0001 --eps_decay=0 --eps_end=0.05 --beta_frames=100000 --batch_size=32 --expert_buffer=pong_buffer.pkl --expert_batch_size=32 --pretrain_steps=0 --update_interval=1000 --eval_frequency=500 --n_b
python train.py --n_step=1 --env=PongNoFrameskip-v0 --lr=0.0001 --eps_decay=0 --eps_end=0.05 --beta_frames=100000 --batch_size=32 --expert_buffer=pong_buffer.pkl --expert_batch_size=32 --pretrain_steps=0 --update_interval=1000 --eval_frequency=500 --n_batches_per_epoch=500 --n_epochs=10 --n_eval_games=4 --use_bandit --bandit_eps=0.2 --bandit_alpha=0.1
#python train.py --n_step=1 --env=PongNoFrameskip-v0 --lr=0.0001 --eps_decay=0 --eps_end=0.05 --beta_frames=100000 --batch_size=32 --expert_buffer=pong_buffer.pkl --expert_batch_size=32 --pretrain_steps=0 --update_interval=1000 --eval_frequency=500 --n_batches_per_epoch=500 --n_epochs=1000 --n_eval_games=4 --use_bandit --bandit_eps=0.4 --bandit_alpha=0.1
#sudo shutdown now -h 
