"""
This file implements the gym like environment of the kukakr5Arc
"""

import os, inspect
import math
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import kukakr5Arc

class kukakr5ArcEnv(gym.Env):

	def __init__(self, urdfRootPath = "/home/nightmareforev/catkin_ws/src/kuka_experimental-indigo-devel/kuka_kr5_support/urdf", timeStep=0.01):
		self._time_step = timeStep
		self._num_bullet_solver_iterations = 300
		self.urdfRootPath = urdfRootPath
		#Make marker radius change with distance_threshold
		self.distance_threshold = 0.2
		self._observation = []
		self._env_step_counter = 0
		self._is_render = False
		self._action_bound = 0.3
		self.reward_type = "Dense"
		# self.maxVelocity = .35
		# self.maxForce = 200
		self.useInverseKinematics = 1
		self.flangeIndex = 6
		#Number of steps to reach the given goal for a single step
		self.goalPos = [0.0,0.0,0.0]
		self.orn = [0.0,0.0,0.0,1.0]
		self.sub_steps = 80
		self._max_episode_steps = 100
		self.time_to_sleep = 1./100.
		self.homePos = [0,-0.5*math.pi,0.5*math.pi,0,0.5*math.pi,0]
		# self._pybullet_client = bc.BulletClient(connection_mode=p.GUI)
		if self._is_render:
			self._pybullet_client = bc.BulletClient(connection_mode=p.GUI)
		else:
			self._pybullet_client = bc.BulletClient()

		self._pybullet_client.setPhysicsEngineParameter(
			numSolverIterations=int(self._num_bullet_solver_iterations))
		self._pybullet_client.setTimeStep(self._time_step)
		#Load the robot
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0,0,-10)
		planeId = p.loadURDF("plane.urdf")
		cubeStartPos = [0,0,0]
		cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
		self.robotId = p.loadURDF(os.path.join(self.urdfRootPath, "kr5.urdf"), cubeStartPos, cubeStartOrientation, useFixedBase = 1)

		self.numJoints = p.getNumJoints(self.robotId)
		# for jointIndex in range(self.numJoints-2):
		# #Two fixed joints hence total_joints-2
		# 	p.resetJointState(self.robotId, jointIndex, self.homePos[jointIndex])
		#Load the marker of the goal
		self.markerId = p.loadURDF(os.path.join(self.urdfRootPath, "sphere.urdf"), self.goalPos, cubeStartOrientation, useFixedBase = 1)

		self.seed()
		self.reset()

		obs = self._get_obs()
		action_dim = 3
		action_high = np.array([self._action_bound]*action_dim)
		self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
		# self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)
		self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))


	def reset(self):
		#Resets without completely destroying the simulation
		for jointIndex in range(self.numJoints-2):
		#Two fixed joints hence total_joints-2
			p.resetJointState(self.robotId, jointIndex, self.homePos[jointIndex])
		self._env_step_counter = 0
		#Reset the targetPos
		while True:
			self.goalPos = np.concatenate([np.random.uniform(low=0.8, high=2, size=1), np.random.uniform(low=-2, high=2, size=1), np.random.uniform(low=0.2, high=2, size=1)])
			if np.linalg.norm(self.goalPos) > 0.3 and np.linalg.norm(self.goalPos) < 1.5:
				break
		#Reset the marker position
		p.resetBasePositionAndOrientation(self.markerId, self.goalPos, self.orn)
		# return self._get_observation()
		obs = self._get_obs()
		return obs


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	"""Stepping the simulation forward using the given action. There is a more complicated sleep scheme in minitaur_gym_env.py"""
	def step(self, action):
		#Clean this my putting the robot specific functions in a different function
		self._get_obs()
		# for ii in range(np.size(action)):
		self._observation[:3] += action
		#Clipping action to not touch the ground
		if self._observation[2] < 0.2:
			self._observation[2] = 0.2
		targetPos = p.calculateInverseKinematics(self.robotId, self.flangeIndex, self._observation[:3], self._observation[3:6])
		p.setJointMotorControlArray(self.robotId, range(self.numJoints-2), p.POSITION_CONTROL, targetPositions=targetPos)
		if self._is_render:
			for _ in range(self.sub_steps):
				self._pybullet_client.stepSimulation()
				# time.sleep(self.time_to_sleep)
		else:
			for _ in range(self.sub_steps):
				self._pybullet_client.stepSimulation()
		obs = self._get_obs()
		done = False
		info = {
		'is_success': self._is_success(obs['achieved_goal'], self.goalPos),
		}
		reward = self.compute_reward(obs['achieved_goal'], self.goalPos, info)
		# info = {
		# 	'is_success': self._is_success(obs['achieved_goal'], self.goalPos),
		# }
		# reward = self.compute_reward(obs['achieved_goal'], self.goalPos, info)
		return obs, reward, done, info

		# self._env_step_counter += 1
		# reward = self._reward()
		# #Add a safety condition that the end_effector should never run against the ground
		# #Termination condition based on the number of environment steps
		# if self._env_step_counter > self._max_episode_steps:
		# 	done = True
		# else:
		# 	done = False
		# #Passing the return for HER needs to be appended with the desired_goal
		# return self._observation, reward, done, {}

	def goal_distance(self, pos1, pos2):
		return np.sqrt(np.sum(np.square(np.array(pos1) - np.array(pos2))))

	def compute_reward(self, achieved_goal, desired_goal, info):
		#Reward assuming the reacher task
		dist = self.goal_distance(achieved_goal, desired_goal)
		#Trying with a much denser reward
		if self.reward_type == "sparse":
			return -(dist>self.distance_threshold).astype(np.float32)
		else:
			return -dist

	def _get_obs(self):
		#Considering the position and orientation of the flange for now
		observation = []
		state = p.getLinkState(self.robotId, self.flangeIndex)[:2]
		pos = state[0]
		achieved_goal = pos
		ori = state[1]
		rel_dist = pos - self.goalPos
		euler = p.getEulerFromQuaternion(ori)
		observation.extend(list(pos))
		observation.extend(list(euler))
		#Added relative distance vector as an observation
		observation.extend(list(rel_dist))
		self._observation = np.array(observation)
		# return self._observation
		return {
		'observation': self._observation,
		'achieved_goal': np.array(achieved_goal),
		'desired_goal': self.goalPos,
		}

	def _is_success(self, achieved_goal, desired_goal):
		dist = self.goal_distance(achieved_goal, desired_goal)
		return (dist<self.distance_threshold).astype(np.float32)

	# def GetObservationUpperBound(self):
	# 	upper_bound = np.array([0.0]*self.GetObservationDimension())
	# 	#Assuming the observation involves only the position and orientation of the end_effector
	# 	upper_bound[0:3] = 5
	# 	#Assuming euler rotations
	# 	upper_bound[3:] = math.pi
	# 	return upper_bound

	# def GetObservationDimension(self):
	# 	return np.size(self._get_obs())

	# def GetObservationLowerBound(self):
	# 	return -self.GetObservationUpperBound()
