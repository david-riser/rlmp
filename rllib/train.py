import argparse
import glob
import gym
import os
import ray
import ray.rllib.agents.dqn.apex as apex
import time
import tensorflow as tf
import wandb


from ray.rllib.env.atari_wrappers import (
    MonitorEnv, NoopResetEnv, MaxAndSkipEnv, 
    EpisodicLifeEnv, FireResetEnv, WarpFrame, 
    FrameStack
    )
from ray.tune import registry
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.visionnet import VisionNetwork


from tensorflow.keras.layers import (
    Activation, Dense, Flatten, Conv2D,
    MaxPooling2D, Input
)
from tensorflow.keras import Model

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="MsPacmanNoFrameskip-v4")
parser.add_argument("--iterations", default=30, type=int)
parser.add_argument("--video_log_freq", default=10, type=int)
parser.add_argument("--num_gpus", default=2, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--num_envs_per_worker", default=8, type=int)
parser.add_argument("--n_step", default=4, type=int)
parser.add_argument("--lr", default=0.00016, type=float)
parser.add_argument("--eps_final", default=0.015, type=float)
parser.add_argument("--eps_timesteps", default=200000, type=int)
parser.add_argument("--timesteps_per_iteration", default=38000, type=int)
parser.add_argument("--target_network_update_freq", default=48000, type=int)
parser.add_argument("--rollout_fragment_length", default=64, type=int)
parser.add_argument("--batch_size", default=64, type=int)
# parser.add_argument("--buffer_size", default=1000000, type=int)
parser.add_argument("--buffer_size", default=125000, type=int)


def custom_wrap_deepmind(env, framestack=4, noop_max=30):
    env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=noop_max)
    if "NoFrameskip" in env.spec.id:
        env = MaxAndSkipEnv(env, skip=framestack)    
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = FrameStack(env, framestack)
    return env


def build_env():
    env = gym.make("MsPacmanNoFrameskip-v4")
    return custom_wrap_deepmind(env)
    


def build_training_config(args):
    config = apex.APEX_DEFAULT_CONFIG.copy()
    config['framework'] = 'tf'
    config['double_q'] = 'true'
    config['dueling'] = 'true'
    config['num_atoms'] = 1
    config['noisy'] = 'false'
    config['n_step'] = args.n_step
    config['lr'] = args.lr
    config['adam_epsilon'] = 0.00015
    config['hiddens'] = [512]
    config['buffer_size'] = args.buffer_size
    config['exploration_config'] = {'final_epsilon':args.eps_final, 'epsilon_timesteps':args.eps_timesteps}
    config['prioritized_replay_alpha'] = 0.5
    config['final_prioritized_replay_beta'] = 1.0
    config['prioritized_replay_beta_annealing_timesteps'] = 2000000
    config['num_gpus'] = args.num_gpus
    config['num_workers'] = args.num_workers
    config['num_envs_per_worker'] = args.num_envs_per_worker
    config['rollout_fragment_length'] = args.rollout_fragment_length
    config['train_batch_size'] = args.batch_size
    config['target_network_update_freq'] = args.target_network_update_freq
    config['timesteps_per_iteration'] = args.timesteps_per_iteration
    config['monitor'] = 'true'
    config['model']['custom_model'] = 'pacnet'
    config['preprocessor_pref'] = ''
    return config


def build_log_config(args, config):
    config_loggables = [
        'double_q', 'dueling', 'num_atoms',
        'noisy', 'n_step', 'lr', 'adam_epsilon',
        'buffer_size', 'num_gpus', 'num_workers',
        'num_envs_per_worker', 'rollout_fragment_length',
        'train_batch_size', 'target_network_update_freq',
        'timesteps_per_iteration', 'prioritized_replay_alpha',
        'final_prioritized_replay_beta', 'prioritized_replay_beta_annealing_timesteps',
    ]
    log_config = { var:config[var] for var in config_loggables }
    log_config['final_episilon'] = config['exploration_config']['final_epsilon']
    log_config['episilon_timesteps'] = config['exploration_config']['epsilon_timesteps']
    return log_config
    

def setup_wandb(args, log_config):
    wandb.init(
        project='pacman', notes='rllib',
        tags=['apex', 'rllib', 'dev run'], config=log_config
    )
    
    
class PacNet(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(PacNet, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)

        print("Initializing PacNet")
        print(obs_space.shape, action_space.n)
        
        # Hard coded input size for now, there is no need
        # to make it complicated at this point. 
        self.input_layer = Input((210,160,12))

        """
        self.conv1 = Conv2D(
            filters=16, kernel_size=(8,8), strides=4,
            activation='relu'
        )(self.input_layer)
        self.conv2 = Conv2D(
            filters=32, kernel_size=(4,4), strides=2,
            activation='relu'
        )(self.conv1)
        """

        # The convolutional encoder part.  I am breaking from the
        # traditional architecture but sticking with the convention
        # of not using max-pooling just strides to down-sample.
        self.conv = Conv2D(filters=16, kernel_size=(3,3), strides=1,
                           activation='relu')(self.input_layer)
        self.conv = Conv2D(filters=16, kernel_size=(3,3), strides=(2,2),
                           activation='relu')(self.conv)
        self.conv = Conv2D(filters=32, kernel_size=(3,3), strides=1,
                           activation='relu')(self.conv)
        self.conv = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2),
                           activation='relu')(self.conv)
        self.conv = Conv2D(filters=64, kernel_size=(3,3), strides=1,
                           activation='relu')(self.conv)
        self.conv = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2),
                           activation='relu')(self.conv)
        
        # The output expects dims 256 to be compatible with
        # rllib.  I think it has to do with the LSTM option. 
        self.flat_layer = Flatten()(self.conv)
        self.flat_layer = Dense(256)(self.flat_layer)
        self.flat_layer = Activation('relu')(self.flat_layer)
        
        self.output_layer = Dense(action_space.n)(self.flat_layer)
        self.model = Model(self.input_layer, [self.flat_layer, self.output_layer])
        self.register_variables(self.model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.model(
            tf.cast(input_dict["obs"], tf.float32))
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    """
    def  __init__(self, obs_space, action_space, num_outputs,
                  model_config, name):
        model_config['conv_filters'] = [
            [16, [8,8], 4],
            [32, [4,4], 2],
        ]
        super(PacNet, self).__init__(obs_space, action_space, num_outputs,
                                     model_config, name)
        self.model = VisionNetwork(obs_space, action_space, num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()
    """
    
if __name__ == "__main__":
    args = parser.parse_args()
    config = build_training_config(args)
    log_config = build_log_config(args, config)
    setup_wandb(args, log_config)

    # Now we setup our custom wrapper
    registry.register_env(
        "wrapped_pacman_env",
        lambda config: build_env()
    )

    # And custom model 
    ModelCatalog.register_custom_model("pacnet", PacNet)    

    # Start ray and load a training instance.
    ray.init()
    trainer = apex.ApexTrainer(config=config, env="wrapped_pacman_env")

    # Find the new folder and make sure we can upload videos
    base_dir = '/home/ubuntu/ray_results/'
    expdir = max([base_dir + d for d in os.listdir(base_dir)], key=os.path.getmtime)
    print("Exp dir detected: {}".format(expdir))
    
    # Begin training
    timesteps = 0 
    for i in range(args.iterations):
        start_time = time.time()

        result = trainer.train()
        print("Finished iter {}".format(i), result)

        elapsed_time = time.time() - start_time
        current_steps = result['timesteps_total'] - timesteps
        
        # Monitor results
        wandb.log(
            {
                'episode_reward_mean':result['episode_reward_mean'],
                'episode_len_mean':result['episode_len_mean'],
                'timesteps_total':result['timesteps_total'],
                'learner_dequeue_time_ms':result['timers']['learner_dequeue_time_ms'],
                'learner_grad_time_ms':result['timers']['learner_grad_time_ms'],
                'learner_overall_time_ms':result['timers']['learner_overall_time_ms'],
                'elapsed_time':elapsed_time,
                'steps_per_unit_time':float(current_steps / elapsed_time)
            }
        )

        timesteps = result['timesteps_total']
        
        # Upload the new videos
        videos = glob.glob(expdir + "/*.mp4")
        print("Videos:", videos)
        if len(videos) > 0 and i % args.video_log_freq == 0:
            wandb.log({'Video {}'.format(i):wandb.Video(videos[0])})

        for video in videos:
            os.system('rm {}'.format(video))
    

