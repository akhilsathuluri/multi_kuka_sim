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
		self._is_render = True
		self._action_bound = 0.3
		# self.maxVelocity = .35
		# self.maxForce = 200
		self.useInverseKinematics = 1
		self.flangeIndex = 6
		#Number of steps to reach the given goal for a single step
		self.goalPos = [0.0,0.0,0.0]
		self.orn = [0.0,0.0,0.0,1.0]
		self.sub_steps = 80
		self.max_sim_steps = 100
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

		observation_high = self.GetObservationUpperBound()
		observation_low = self.GetObservationLowerBound()
		action_dim = 3
		action_high = np.array([self._action_bound]*action_dim)
		self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
		self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

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

		return self._get_observation()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	"""Stepping the simulation forward using the given action. There is a more complicated sleep scheme in minitaur_gym_env.py"""
	def step(self, action):
		#Clean this my putting the robot specific functions in a different function
		self._observation = self._get_observation()
		for ii in range(len(action)):
			self._observation[ii] += action[ii]
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

		self._env_step_counter += 1
		reward = self._reward()
		#Add a safety condition that the end_effector should never run against the ground
		#Termination condition based on the number of environment steps
		if self._env_step_counter > self.max_sim_steps:
			done = True
		else:
			done = False
		return self._observation, reward, done, {}

	def dist(self, pos1, pos2):
		return np.sqrt(np.sum(np.square(np.array(pos1) - np.array(pos2))))

	def _reward(self):
		#Reward assuming the reacher task
		#Get distance from the targetPosition
		# flangePosition = self._pybullet_client.getLinkState(self.robotId, self.flangeIndex)[:2][0]
		dist = self.dist(self._observation[:3], self.goalPos)
		if dist < self.distance_threshold:
			reward = 1
		else:
			reward = -1
		print(dist, reward)
		return reward

	def _get_observation(self):
		#Considering the position and orientation of the flange for now
		observation = []
		state = p.getLinkState(self.robotId, self.flangeIndex)[:2]
		pos = state[0]
		ori = state[1]
		rel_dist = pos - self.goalPos
		euler = p.getEulerFromQuaternion(ori)
		observation.extend(list(pos))
		observation.extend(list(euler))
		#Added relative distance vector as an observation
		observation.extend(list(rel_dist))
		self._observation = np.array(observation)
		return self._observation

	def GetObservationUpperBound(self):
		upper_bound = np.array([0.0]*self.GetObservationDimension())
		#Assuming the observation involves only the position and orientation of the end_effector
		upper_bound[0:3] = 5
		#Assuming euler rotations
		upper_bound[3:] = math.pi
		return upper_bound

	def GetObservationDimension(self):
		return np.size(self._get_observation())

	def GetObservationLowerBound(self):
		return -self.GetObservationUpperBound()
