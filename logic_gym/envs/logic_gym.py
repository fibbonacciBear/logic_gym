import copy
from enum import Enum
import gymnasium as gym
import numpy as np
import string
import random

from gymnasium.spaces import Text, MultiDiscrete
from logic_gym.flip_executor import FlipExecutor


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

    def __init__(
        self, render_mode=None, max_proof_length=max_proof_length, max_steps=32
    ):

        self.max_proof_length = max_proof_length
        self.max_steps = max_steps
        self.max_variables = 1
        self.variables = []

        self.current_proof_length_zero_indexed = -1
        self._stats_steps = 0
        self._info = {"bad_action": False}
        
        
        # Define the observation space
        vocab = string.ascii_letters + string.digits + string.punctuation + " " + "\n"
        self.observation_space = Text(
            min_length=0, max_length=self.MAX_OBSERVATION_LENGTH, charset=vocab
        )
        self.observation = ""
        
        self.executed_flip_statements = []

        # Define the action space
        self.action_space = MultiDiscrete(
            [len(Rules), max_proof_length, max_proof_length, 1]
        )

        self.truncated = False
        self.terminated = False
        self._pp_state = ""

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize the FLiP wrapper
        self._flip_wrapper = FlipExecutor()
        with open("logic_gym/default_task.txt", "r") as file:
            self._task = file.read()
        try:
            self._flip_wrapper.start([])
        except:
            raise Exception("Error in starting the FLiP process")

    def _get_info(self) -> dict:
        return self._info

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

    def get_state(self) -> dict:
        """
        Returns the state for the Logic Gym environment.
        Returns:
            str: The state for the Logic Gym environment.
        """
        # self.current_proof_length_zero_indexed
        # self._stats_steps
        # self._info
        # self.observation = ""
        # self.truncated = False
        # self.terminated = False
        # self._state = ""

        return {
            "current_proof_length_zero_indexed": self.current_proof_length_zero_indexed,
            "stats_steps": self._stats_steps,
            "info": copy.deepcopy(self._info),
            "observation":  copy.deepcopy(self.observation),
            "truncated": self.truncated,
            "terminated": self.terminated,
            "state":  copy.deepcopy(self._pp_state),
            "executed_flip_statements":  copy.deepcopy(self.executed_flip_statements),
        }

    def set_state(self, state: dict) -> None:
        """
        Set the state for the Logic Gym environment.
        Parameters:
        - state (str): The state to be set for the environment.
        Returns:
        - None
        """
        self.current_proof_length_zero_indexed = state["current_proof_length_zero_indexed"]
        self._stats_steps = state["stats_steps"]
        self._info = copy.deepcopy(state["info"])
        self.observation = copy.deepcopy(state["observation"])
        self.truncated = state["truncated"]
        self.terminated = state["terminated"]
        self._pp_state = copy.deepcopy(state["state"])
        self.executed_flip_statements = copy.deepcopy(state["executed_flip_statements"])
        try:
            self._flip_wrapper.reset( self.get_task().split("\n") + self.executed_flip_statements)
        except:
            raise Exception("Error in setting the state")

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        axioms_list = self.get_task().split("\n")
        self.current_proof_length_zero_indexed = 0
        self.terminated = False
        self.truncated = False
        self._stats_steps = 0
        self.reset_count += 1
        self.variables = self.get_variables(axioms_list)
        self._info = {"bad_action": False}

        try:
            _, self.observation = self._flip_wrapper.reset(axioms_list)
            self._set_max_premise_index()
            self._pp_state = self._flip_wrapper.get_state_using_pp()
        except:
            raise Exception("Error in setting the task")

        return self.observation, self._get_info()

    def _action_to_flip_statement(self, action):

        rule = Rules(action[0])
        premise1 = action[1]
        premise2 = action[2]
        variable = action[3]

        ## TODO: Add support for multiple variables
        variable_name = self.variables[variable]

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
            self._info = {"bad_action": True}
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
            self._pp_state = self._flip_wrapper.get_state_using_pp()
            self.executed_flip_statements.append(flip_statement)
        except:
            self._info = {"bad_action": True}
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
        
        
    def get_variables(self, axioms_list):
        variables = []
        for axiom in axioms_list:
            if "Variable(" in axiom:
                variables.append(axiom.split(" = ")[0])
        return variables

    def get_state_for_humans(self):
        return self._pp_state

    def get_stats(self):
        return {
            "reset_count": self.reset_count,
            "terminated_count": self.terminated_count,
            "truncated_count": self.truncated_count,
            "steps": self._stats_steps,
        }
        
    def get_all_actions(self):
        actions_one_two =  [
            [rule, premise1, premise2, variable]
            for rule in range(1, len(Rules))
            for premise1 in range(self.max_proof_length)
            for premise2 in range(self.max_proof_length)
            for variable in range(self.max_variables)
        ]
        
        actions_zero = [
            [0, premise1, 0, variable]
            for premise1 in range(self.max_proof_length)
            for variable in range(self.max_variables)
        ]
        
        actions = actions_zero + actions_one_two
        
        random.shuffle(actions)
        return actions

    def action_masks(self):
        return (
            [True] * 3
            + [True] * self.current_proof_length_zero_indexed
            + [False] * (self.max_proof_length - self.current_proof_length_zero_indexed)
            + [True] * self.current_proof_length_zero_indexed
            + [False] * (self.max_proof_length - self.current_proof_length_zero_indexed)
            + [True] * 1
        )
