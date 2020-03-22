#import os, inspect
#currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#parentdir = os.path.dirname(os.path.dirname(currentdir))
#os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data

class KR5:
	
	def __init__(self, urdfRootPath = "/home/nightmareforev/catkin_ws/src/kuka_experimental-indigo-devel/kuka_kr5_support/urdf", timeStep=0.01):
		self.urdfRootPath = urdfRootPath
		self.timeStep = timeStep
		self.maxVelocity = .35
		self.maxForce = 200
		self.useInverseKinematics = 1
		self.flangeIndex = 6;
#		self.useNullSpace = 21 
#		self.useOrientation = 1 
		#Setting limits for null space

		self.homePos = [0,0.5*math.pi,-0.5*math.pi,0,0,0]
		self.reset()

	def reset(self):
		robot = p.loadURDF(os.path.join(self.urdfRootPath, "kr5.urdf", usedFixedBase = 1)
		self.robotId = robot[0]
		self.numJoints = p.getNumJoints(self.robotId)
		for jointIndex in range(self.numJoints-2):
		#Two fixed joints hence total_joints-2
			p.resetJointState(self.robotId, jointIndex, self.homePos[jointIndex])
		#This resets simulation without having to use dynamics. Given that it is dangerous to use in the simulation, but I think this as the right way to reset a simulation.
		#Add additional objects required for the simulation, like the table and a cube to be manipulated.		
		#self.objects  = p.loadURDF("r2d2.urdf", pos, ori, useFixedBase=0)
		#You can print information similar to this one: https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/kuka.py
	
	def getActionDimension(self):
		return 3 #Only flange position #Position and Orientation of the end-effector. The actions can be increased based on the implementation.
	
	def getObservationDimension(self):
		return len(self.getObservation())

	def getObservation(self):
		observation = []
		#Considering the position and orientation of the flange for now
		state = p.getLinkState(self.robotId, self.flangeIndex)
		pos = state[0]
		ori = state[1]
		euler = p.getEulerFromQuaternion(ori)
		observation.extend(list(pos))
		observation.extend(list(euler))
		return observation

	def applyAction(self, actions):
		#There are several if-else conditions considered in the above reference file. We are using a simpler method.
		currentflangePose = p.getLinkState(self.robotId, self.flangeIndex)
		currentflangePos = currentflangePose[0]
		currentflangeOri = currentflangePose[1]
		
		self.flagePos[0] = currentflangePos[0]+actions[0]
		self.flagePos[1] = currentflangePos[1]+actions[1]
		self.flagePos[2] = currentflangePos[2]+actions[2]
		# Add more based on the change in action dim
		#self.flageOri = currentflangeOri+actions[3]

		targetjointPose = self.calculateInverseKinematics(self.robotId, self.flangeIndex, self.flangePos, currentflangeOri)
		#Given code handles the case where simulation is off
		p.setJointMotorControlArray(self.robotId, range(self.numJoints), p.POSITION_CONTROL, targetPositions=targetjointPose)
