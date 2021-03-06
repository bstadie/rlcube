#!/usr/bin/env python3

from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
from rlcube import RLCube

def train():
    env = RLCube()
    num_timesteps = 10000
    timesteps_per_actorbatch = 1000
    import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                                    hid_size=64, num_hid_layers=2)

    pi = pposgd_simple.learn(env, policy_fn,
                             max_timesteps=num_timesteps,
                             timesteps_per_actorbatch=timesteps_per_actorbatch,
                             clip_param=0.2, entcoeff=0.0,
                             optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                             gamma=0.99, lam=0.95, schedule='linear',
                             )

if __name__ == '__main__':
    logger.configure()
    train()