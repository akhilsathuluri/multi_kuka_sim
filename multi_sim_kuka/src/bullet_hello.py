#The same works with python3 too!
import pybullet as p
import time
import pybullet_data
#physicsClient = p.DIRECT()
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
#Loading robot
cubeStartPos = [0,0,0]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
robot = p.loadURDF("/home/nightmareforev/catkin_ws/src/kuka_experimental-indigo-devel/kuka_kr5_support/urdf/kr5.urdf", cubeStartPos, cubeStartOrientation, useFixedBase=1)
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
reach_pos = [1,0.15,0.6]
targetJpos = p.calculateInverseKinematics(robot, 6, reach_pos, reach_ori)
#Can get link state (FK)
joint_number = 6
p.getLinkState(robot, joint_number)[:2]
p.setJointMotorControlArray(robot, range(6), p.POSITION_CONTROL, targetPositions=targetJpos)

frameSkip = 10
timeStep = 0.002

p.setPhysicsEngineParameter(fixedTimeStep=timeStep * frameSkip,         numSolverIterations=15,                                     numSubSteps=frameSkip)

#Simulation
for _ in range (50):
    p.stepSimulation()
    time.sleep(1./10.)
#cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
#print(cubePos,cubeOrn)
#home position
homePos = [0,-0.5*math.pi,0.5*math.pi,0,0.5*math.pi,0]
#reset all the joints to a home position
for i in range(numJoints-2):
	p.resetJointState(robot, i, homePos[i])
time.sleep(5)
p.disconnect()
