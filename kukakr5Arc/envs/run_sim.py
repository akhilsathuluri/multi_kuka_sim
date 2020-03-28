import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

import os
import pybullet_envs
import kukakr5Arc

env = gym.make('kukakr5Arc-v1')

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
model.learn(total_timesteps=400000)
model.save("/home/nightmareforev/git/multi_kuka_sim/kukakr5Arc/envs/saved_policies/kukakr5Arc_reacher")
print('Saving model.... Model saved')

del model # remove to demonstrate saving and loading

model = DDPG.load("/home/nightmareforev/git/multi_kuka_sim/kukakr5Arc/envs/saved_policies/kukakr5Arc_reacher", env = env)
print('Loading model.....Model loaded')

#env.render() goes before env.reset() for the render to work
#env.render()

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
#    env.render()

while True:
	obs = env.reset()
	for moves in range(80):
		action, _states = model.predict(obs)
		obs, rewards, dones, info = env.step(action)
		moves += 1


