import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='kukakr5Arc-v0',
    entry_point='kukakr5Arc.envs:kukakr5ArcEnv',
#    timestep_limit=1000.0,
    reward_threshold=1.0,
    nondeterministic = True,
)

#Updated version of the env

register(
    id='kukakr5Arc-v1',
    entry_point='kukakr5Arc.envs:kukakr5ArcEnv_v1',
#    timestep_limit=1000.0,
    reward_threshold=1.0,
    nondeterministic = True,
)

#Use the others to make the pusher env

#register(
#    id='SoccerEmptyGoal-v0',
#    entry_point='gym_soccer.envs:SoccerEmptyGoalEnv',
#    timestep_limit=1000,
#    reward_threshold=10.0,
#    nondeterministic = True,
#)

