from enum import Enum
import gymnasium as gym
import numpy as np
import string

from gymnasium.spaces import Text, MultiDiscrete
from logic_gym.flip_executor import FlipWrapper


class Rules(Enum):
    Ae = 0
    imple = 1
    contra = 2


class LogicGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}
    max_proof_length = 16
    MAX_OBSERVATION_LENGTH = 1000
    reset_count = 0
    terminated_count = 0
    truncated_count = 0

    def __init__(self, render_mode=None, max_proof_length=max_proof_length, max_steps=32):

        self.max_proof_length = max_proof_length
        self.current_proof_length_zero_indexed = -1
        self._stats_steps = 0
        self.max_steps = max_steps

        # Define the observation space
        vocab = string.ascii_letters + string.digits + string.punctuation + " " + "\n"
        self.observation_space = Text(min_length=0, max_length=self.MAX_OBSERVATION_LENGTH, charset=vocab)
        self.observation = ""

        # Define the action space
        self.action_space = MultiDiscrete(
            [len(Rules), max_proof_length, max_proof_length, 1]
        )

        self.truncated = False
        self.terminated = False
        self._state = ""

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize the FLiP wrapper
        self._flip_wrapper = FlipWrapper()
        with open("logic_gym/default_task.txt", "r") as file:
            self._task = file.read()

    def _get_info(self) -> dict:
        return {}

    def set_task(self, task: str) -> None:
        """
        Set the task for the Logic Gym environment.
        Parameters:
        - task (str): The task to be set for the environment.
        Returns:
        - None
        """
        self._task = task

    def get_task(self) -> str:
        """
        Returns the task associated with the environment.
        Returns:
            str: The task associated with the environment.
        """
        return self._task

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        axioms_list = self.get_task().split("\n")
        self.current_proof_length_zero_indexed = 0
        self.terminated = False
        self.truncated = False
        self._stats_steps = 0
        self.reset_count += 1

        try:
            self.observation = self._flip_wrapper.start(axioms_list)
            self._set_max_premise_index()
            self._state = self._flip_wrapper.get_state_using_pp()
        except:
            raise Exception("Error in setting the task")

        return self.observation, self._get_info()

    def _action_to_flip_statement(self, action):

        rule = Rules(action[0])
        premise1 = action[1]
        premise2 = action[2]
        variable = action[3]

        ## TODO: Add support for multiple variables
        variable_name = "Sally"

        if rule == Rules.Ae:
            move = f"rapply({rule.name}, {premise1}, {variable_name})"
        else:
            move = f"rapply({rule.name}, {premise1}, {premise2})"

        if (
            premise1 > self.current_proof_length_zero_indexed
            or premise2 > self.current_proof_length_zero_indexed
        ):
            move = None
        return move

    def step(self, action):
        self._set_max_premise_index()
        self._stats_steps += 1
        reward = 0
        flip_statement = self._action_to_flip_statement(action)

        if self._stats_steps >= self.max_steps:
            self.truncated_count += 1
            self.truncated = True

        if flip_statement is None:
            return (
                self.observation,
                reward,
                self.terminated,
                self.truncated,
                self._get_info(),
            )
        try:

            goal_state, self.observation = self._flip_wrapper.run_proof_step(
                flip_statement
            )
            self._state = self._flip_wrapper.get_state_using_pp()

        except:
            return (
                self.observation,
                reward,
                self.terminated,
                self.truncated,
                self._get_info(),
            )

        self._set_max_premise_index()
        if not goal_state == "unknown":
            reward = 1
            self.terminated = True
            self.terminated_count += 1

        if (
            not self.terminated
            and self.current_proof_length_zero_indexed >= self.max_proof_length
        ):
            self.truncated = True
            self.truncated_count += 1


        return (
            self.observation,
            reward,
            self.terminated,
            self.truncated,
            self._get_info(),
        )

    def _set_max_premise_index(self):
        """
        Set the maximum premise index.
        """
        self.current_proof_length_zero_indexed = len(self.observation.split("\n")) - 2

    def close(self):
        self._flip_wrapper.terminate()

    def get_state_for_humans(self):
        return self._state

    def get_stats(self):
        return {
            "reset_count": self.reset_count,
            "terminated_count": self.terminated_count,
            "truncated_count": self.truncated_count,
            "steps": self._stats_steps,
        }

    def action_masks(self):
        return (
            [True] * 3
            + [True] * self.current_proof_length_zero_indexed
            + [False] * (self.max_proof_length - self.current_proof_length_zero_indexed)
            + [True] * self.current_proof_length_zero_indexed
            + [False] * (self.max_proof_length - self.current_proof_length_zero_indexed)
            + [True] * 1
        )
