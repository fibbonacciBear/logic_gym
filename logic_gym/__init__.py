from gymnasium.envs.registration import register
import sys
sys.path.append('../')

register(
    id="logic_gym/LogicGym-v0",
    entry_point="logic_gym.envs:LogicGymEnv",
)
