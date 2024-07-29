import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as SpaceDict
import numpy as np
import torch
from transformers import pipeline


class CodeBertWrapper(gym.ObservationWrapper):
    codebert_pipeline = None

    def __init__(self, env):
        super().__init__(env)
        self.codebert_pipeline = CodeBertWrapper._initialize_codebert()

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32
        )

        self._last_observation_len = 0
        self._last_embedding = None

    @classmethod
    def _initialize_codebert(cls):
        if CodeBertWrapper.codebert_pipeline is not None:
            return CodeBertWrapper.codebert_pipeline

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        if torch.backends.mps.is_available():
            device = "mps"

        # print("---------------------------------------------")
        # print("Creating a new pipeline for CodeBertWrapper")
        # print("---------------------------------------------")
        CodeBertWrapper.codebert_pipeline = pipeline(
            task="feature-extraction",
            model="microsoft/codebert-base",
            device=device,
        )
        return CodeBertWrapper.codebert_pipeline

    def observation(self, obs):
        if self._last_observation_len == len(obs):
            embedding = self._last_embedding
        else:
            embedding = torch.Tensor(self.codebert_pipeline(obs)[0]).mean(0).numpy()
            self._last_embedding = embedding
            self._last_observation_len = len(obs)
        return embedding

    
    
    #########################################################################    
    # Old version without the singleton variable for the pipeline
    #########################################################################
    # class CodeBertWrapper(gym.ObservationWrapper):
    # def __init__(self, env):
    #     super().__init__(env)
    #     self._initialize_codebert()

    #     self.observation_space = Box(
    #         low=-np.inf, high=np.inf, shape=(768,), dtype=np.float32
    #     )

    # def _initialize_codebert(self):
    #     device = "cpu"
    #     if torch.cuda.is_available():
    #         device = "cuda"
    #     if torch.backends.mps.is_available():
    #         device = "mps"

    #     self.codebert_pipeline = pipeline(
    #         task="feature-extraction",
    #         model="microsoft/codebert-base",
    #         device=device,
    #     )

    # def observation(self, obs):
    #     return torch.Tensor(self.codebert_pipeline(obs)[0]).mean(0).numpy()


