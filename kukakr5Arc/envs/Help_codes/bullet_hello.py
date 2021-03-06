#The same works with python3 too!
import pybullet as p
import pybullet_utils.bullet_client as bc
import time
import pybullet_data
import gym
import kukakr5Arc
import math
_pybullet_client = bc.BulletClient()
env = gym.make('kukakr5Arc-v0')
#physicsClient = p.DIRECT()
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
#Loading robot
cubeStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
robot = p.loadURDF("/home/nightmareforev/catkin_ws/src/kuka_experimental-indigo-devel/kuka_kr5_support/urdf/kr5.urdf", cubeStartPos, cubeStartOrientation, useFixedBase=1)

#Loading a marker
pos = [0.9, 0.2, 0.5]
marker = p.loadURDF("/home/nightmareforev/catkin_ws/src/kuka_experimental-indigo-devel/kuka_kr5_support/urdf/sphere.urdf", pos, cubeStartOrientation, useFixedBase=1)

pos_new = [0.5,0.3,0.5]
orn_new = [0.0,0.0,0.0,1.0]
p.resetBasePositionAndOrientation(marker, pos_new, orn_new)

#r2d2 = p.loadURDF("r2d2.urdf", [0,0,0], cubeStartOrientation)
numJoints = p.getNumJoints(robot)
#Get joint angles
joint_positions = [j[0] for j in p.getJointStates(robot,range(6))]
#Set real-time simulation. This is by defaults so no need to explicitly execute this step. The sim steps only when stepSimulation is called
p.setRealTimeSimulation(0)
#To use the dynamic simulation and not just the IK use the following
useSimulation = 1
#Target position control
#p.setJointMotorControlArray(robot, range(6), p.POSITION_CONTROL, targetPositions=[0.2]*6)
#calculate the motion to reach a particular point in space
reach_ori = p.getQuaternionFromEuler([3.14, 0, 0])
reach_pos = [0.2,0.2,0.3]
targetJpos = p.calculateInverseKinematics(robot, 6, reach_pos, reach_ori)
#Can get link state (FK)
joint_number = 6
p.getLinkState(robot, joint_number)[:2]
p.setJointMotorControlArray(robot, range(6), p.POSITION_CONTROL, targetPositions=targetJpos)
#Simulation
for i in range (100):
    p.stepSimulation()
    time.sleep(1./10.)
#cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
#print(cubePos,cubeOrn)
#home position
jointPoses = [0,0,0,0,0,0]
#reset all the joints to a home position
for i in range(numJoints):
	p.resetJointState(robot, i, jointPoses[i])
time.sleep(5)
p.disconnect()
