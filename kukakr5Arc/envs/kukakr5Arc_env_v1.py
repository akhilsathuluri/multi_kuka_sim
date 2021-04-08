"""
This file implements the gym environment of example PyBullet simulation.
"""

import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import time
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc

import pybullet_data

from pkg_resources import parse_version


class kukakr5ArcEnv_v1(gym.Env):
    """
    The gym environment to run pybullet simulations.
    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self,
               render=True,
               #Where
               render_sleep=False,
               #Where
               debug_visualization=True,
               render_width=240,
               render_height=240,
               action_repeat=1,
               time_step=1./240.,
               num_bullet_solver_iterations=50,
               urdf_root="/home/nightmareforev/git/bullet_stuff/multi_kuka_sim/kuka_kr5_support/urdf"):
        """Initialize the gym environment.
        Args:
        urdf_root: The path to the urdf data folder.
        """
        self._time_step = time_step
        self._urdf_root = urdf_root
        self._observation = []
        self._action_repeat = action_repeat
        self._num_bullet_solver_iterations = num_bullet_solver_iterations
        self._env_step_counter = 0
        self._is_render = render
        #Where
        self._debug_visualization = debug_visualization
        #Where
        self._render_sleep = render_sleep
        self._render_width = render_width
        self._render_height = render_height
        self._cam_dist = .3
        self._cam_yaw = 50
        self._cam_pitch = -35
        self._last_frame_time = 0.0

        print("urdf_root=" + self._urdf_root)

        if self._is_render:
            self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bc.BulletClient()

        #Never going to use this part anyway
        if (debug_visualization == False):
            self._pybullet_client.configureDebugVisualizer(flag=self._pybullet_client.COV_ENABLE_GUI,enable=0)
            self._pybullet_client.configureDebugVisualizer(flag=self._pybullet_client.COV_ENABLE_RGB_BUFFER_PREVIEW, enable=0)
            self._pybullet_client.configureDebugVisualizer(          flag=self._pybullet_client.COV_ENABLE_DEPTH_BUFFER_PREVIEW, enable=0)
            self._pybullet_client.configureDebugVisualizer(          flag=self._pybullet_client.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, enable=0)

        self._pybullet_client.setAdditionalSearchPath(urdf_root)
        self._pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._pybullet_client.setGravity(0,0,-10)
        #Load the env
        planeId = self._pybullet_client.loadURDF("plane.urdf")
        self.robotPos = [0,0,0]
        self.robotOri = self._pybullet_client.getQuaternionFromEuler([0,0,0])
        self.robotId = self._pybullet_client.loadURDF(os.path.join(self._urdf_root, "kr5.urdf"), self.robotPos, self.robotOri, useFixedBase = 1)
        self.flangeIndex = 6
        self.homePos = [0, -0.5*math.pi, 0.5*math.pi, 0, 0.5*math.pi, 0]
        numJoints = self._pybullet_client.getNumJoints(self.robotId)
        #Removing the fixed joints
        self.robotJoints = numJoints-2
        #Load the marker of the goal
        self.markerId = self._pybullet_client.loadURDF(os.path.join(self._urdf_root, "sphere.urdf"), self.robotPos, self.robotOri, useFixedBase = 1)
        #Doing these only once instead of everytime in reset
        self._pybullet_client.setPhysicsEngineParameter(
            numSolverIterations=int(self._num_bullet_solver_iterations))
        self._pybullet_client.setTimeStep(self._time_step)

        #Env params
        self.distance_threshold = 0.2
        self.reward_type = "dense"
        self._max_episode_steps = 100

        self.seed()
        self.reset()
        #Compatibility with gym
        observation_high = (self.GetObservationUpperBound())
        observation_low = (self.GetObservationLowerBound())
        action_dim = self.GetActionDimension()
        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)
        #Where
        self.viewer = None

    #Where
    def configure(self, args):
        self._args = args

    def reset(self):
        # if self._is_render:
        #     self._pybullet_client = bc.BulletClient(connection_mode=pybullet.GUI)
        # else:
        #     self._pybullet_client = bc.BulletClient()

        for jointIndex in range(self.robotJoints):
            self._pybullet_client.resetJointState(self.robotId, jointIndex, self.homePos[jointIndex])

        self._env_step_counter = 0
        #Sample the goal position
        self.SampleGoal()
        #Reset the marker position
        self._pybullet_client.resetBasePositionAndOrientation(self.markerId, self.goalPos, self.robotOri)
        self._env_step_counter = 0
        return self._get_observation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #I wonder how this helps as action is completed in one step
        if self._render_sleep:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self._action_repeat * self._time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

        for _ in range(self._action_repeat):
            self.ApplyAction(action)
            self._pybullet_client.stepSimulation()

        self._env_step_counter += 1
        reward = self._reward()
        done = self._termination()
        return np.array(self._observation), reward, done, {}

    def render(self, mode="rgb_array", close=False):
        self._is_render = True
        #Not changing anything as this was working for another example
        if mode != "rgb_array":
            return np.array([])
        base_pos = [0, 0, 0]
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
        fov=60, aspect=float(self._render_width) / self._render_width, nearVal=0.01, farVal=100.0)
        proj_matrix = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
            -0.02000020071864128, 0.0
            ]
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
        width=self._render_width,
        height=self._render_height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)  #ER_TINY_RENDERER)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self._render_height, self._render_width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        #Setting terminate condition to be self.steps
        if self._env_step_counter > self._max_episode_steps:
            terminate = True
        else:
            terminate = False
        return terminate

    def _get_observation(self):
        observation = []
        state = self._pybullet_client.getLinkState(self.robotId, self.flangeIndex)[:2]
        pos = state[0]
        ori = state[1]
        rel_dist = pos - self.goalPos
        euler = self._pybullet_client.getEulerFromQuaternion(ori)
        observation.extend(list(pos))
        observation.extend(list(euler))
        #Added relative distance vector as an observation
        observation.extend(list(rel_dist))
        self._observation = np.array(observation)
        return self._observation

    #Where
    if parse_version(gym.__version__) < parse_version('0.15.7'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step

    #Other essential functions required
    def GetObservationUpperBound(self):
        upper_bound = np.array([np.inf]*self.GetObservationDimension())
        return upper_bound

    def GetObservationLowerBound(self):
        return -self.GetObservationUpperBound()

    def GetObservationDimension(self):
        return np.size(self._get_observation())

    def GetActionDimension(self):
        #Both position and orientation
    	return 6

    def SampleGoal(self):
        self.goalPos = np.concatenate([np.random.uniform(low=0.8, high=1.5, size=1), np.random.uniform(low=-1.5, high=1.5, size=1), np.random.uniform(low=0.2, high=1.5, size=1)])

    def ApplyAction(self, action):
        self._observation = self._get_observation()
        self._observation[:6] += action
        #Clipping action to not touch the ground
        if self._observation[2] < 0.2:
            self._observation[2] = 0.2
        targetPos = self._pybullet_client.calculateInverseKinematics(self.robotId, self.flangeIndex, self._observation[:3], self._observation[3:6])
        self._pybullet_client.setJointMotorControlArray(self.robotId, range(self.robotJoints), self._pybullet_client.POSITION_CONTROL, targetPositions=targetPos)

    def _reward(self):
        dist = self.goal_distance(self._observation[:3], self.goalPos)
        if self.reward_type == "sparse":
            return -(dist>self.distance_threshold).astype(np.float32)
        else:
            return -dist

    def goal_distance(self, pos1, pos2):
        return np.sqrt(np.sum(np.square(np.array(pos1) - np.array(pos2))))

    def close(self):
        self._pybullet_client.disconnect()
