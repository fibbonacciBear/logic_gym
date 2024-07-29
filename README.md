
# Natural Deduction Solver for First Order Logic

There are many first order logic solvers available such as Vampire, E, iProver, ect. However all of them are [saturation based provers](https://www.sciencedirect.com/science/article/pii/S0168007222000823). Our goal is to develop a natural deduction based prover which produces explainable proof steps and mirrors human based thinking. To do this, we will be exploring learning an algorithm via reinforcement learning to solve the problem rather than explicitly coding it. 

We shall be restricting ourselves to simple first order logic problems for the time being. Towards that, we will be using a [ProntoQA dataset](https://arxiv.org/pdf/2210.01240). We will start with 1 hop dataset from ProntoQA and gradually build towards multihop ProntoQA datasets. The dataset folder contains code for formalizing the prontoQA dataset which is written in natural language. We formalize the dataset with the use of openAI's GPT. 

We use [FLiP](https://jon-jacky.github.io/FLiP/www/reference-nd.html) as a logical framework for defining logic and as a proof checker for when the proof is built. FLiP is an extension of python built to understand first order logic. 

Gymnasium is an open source python library for developing and comparing reinforcement learning applications by providing a standard API to communicate between learning algorithms and environments. We will build a custom environment that wraps FLiP and provides a standard set of gymnasium interfaces that could be used by popular frameworks such as [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/), [coreRL](https://docs.cleanrl.dev/) and [RLlib](https://docs.ray.io/en/latest/rllib/index.html). We experiment with classic reinforcement learning algorithms such as PPO and DQN. In addition to this we would like to model reinforcement learning as a sequence to sequence modelling problem using LLMS similar to the work done in [decision transformers](https://arxiv.org/abs/2106.01345). 

We are interested in using decision transformers to solve the RL problem. While using decision transformers, we would like to see how small the language model can be made and still continue solving first order logic problems (in similar lines along [TinyStories investigation](https://arxiv.org/pdf/2305.07759)).

## Get started:

Create a .env file with the following environment variables, adjust to match your environment.

```shell
OPEN_AI_API_KEY=<your-openai-api-key>
FLIP_HOME=/Users/akashganesan/anaconda3/envs/drlzh/lib/python3.11/site-packages/flip
PYTHON_BINARY=/Users/akashganesan/anaconda3/envs/drlzh/bin/python
```

## Installation

To install your new environment, run the following commands:

```{shell}
cd logic_gym
pip install -e .
```

