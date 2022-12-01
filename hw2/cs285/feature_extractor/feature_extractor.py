import abc
import itertools
from torch import nn

import numpy as np
import torch

from cs285.infrastructure import pytorch_util as ptu


class FeatureExtractor(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, ac_dim, ob_dim, learning_rate=1e-4, **kwargs):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.learning_rate = learning_rate
        self.conv_head = torch.nn.Sequential(
            torch.nn.Conv2d(3, 5, 5, stride=2),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU(),
            torch.nn.Conv2d(5, 5, 10, stride=2),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU(),
            torch.nn.Conv2d(5, 5, 10, stride=2),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((20, 20)),
            torch.nn.Flatten(),
            torch.nn.Linear(2000, 99),  # FIXME: to 99
            torch.nn.ReLU(),
        )
        self.mlp_classifier = ptu.build_mlp(
            input_size=100,
            output_size=1,
            n_layers=3,
            size=30,
            output_activation="sigmoid",
        )

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_feature(self, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if isinstance(obs, list):
            print("list")
            obs = np.array(obs)

        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        if isinstance(actions, list):
            print("list")
            actions = np.array(actions)

        if len(actions.shape) > 1:
            actions = actions
        else:
            actions = actions[None]

        # TODO return the action that the policy prescribes
        observation = ptu.from_numpy(observation)
        actions = ptu.from_numpy(actions)
        rew = self(observation, actions)
        return ptu.to_numpy(rew)

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor, actions: torch.FloatTensor):
        conv_head = self.conv_head(observation)
        input_to_mlp = torch.cat((conv_head, actions), 1)
        # input_to_mlp = conv_head
        output = self.mlp_classifier(input_to_mlp)
        return output
