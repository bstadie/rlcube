from baselines.common import tf_util as U
#from super_expert.experts.pposgd_simple import traj_segment_generator, env_render_gen
import gym
import copy
import numpy as np
from baselines import deepq
#from ml_logger import logger

from rlcube import RLCube

def train_deep_q(name, env, num_timesteps, render=False):
    max_timesteps = num_timesteps
    max_timesteps = 1000
    env = env
    # Enabling layer_norm here is import for parameter space noise!
    model = deepq.models.mlp([16], layer_norm=False)
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=max_timesteps,
        buffer_size=50,
        exploration_fraction=0.1,
        exploration_final_eps=0.1,
        print_freq=1,
        param_noise=False,
        train_freq=50

    )



if __name__ == '__main__':
    env = RLCube()
    train_deep_q(name='cube', env=env, num_timesteps=int(1e6), render=False)

    #env = gym.make('MountainCar-v0')
    #kkk
    #num_timesteps = 1000000
    #name = 'mountain_car'
    #train_deep_q(name=name, env=env, num_timesteps=num_timesteps)
