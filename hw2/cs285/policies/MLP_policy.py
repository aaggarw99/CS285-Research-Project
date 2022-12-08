import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy
from cs285.infrastructure.utils import normalize


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        ac_dim,
        ob_dim,
        n_layers,
        size,
        discrete=False,
        learning_rate=1e-4,
        training=True,
        nn_baseline=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        self.conv_head = None
        self.conv_head_baseline = None

        if isinstance(self.ob_dim, tuple):
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
                torch.nn.Linear(2000, 99),
            )
            self.conv_head_baseline = torch.nn.Sequential(
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
                torch.nn.Linear(2000, 99),
                torch.nn.ReLU(),
            )
            self.conv_head.to(ptu.device)
            self.conv_head_baseline.to(ptu.device)
            self.ob_dim = 99

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim + 1,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(
                self.logits_na.parameters()
                if self.conv_head is None
                else itertools.chain(
                    self.logits_na.parameters(), self.conv_head.parameters()
                ),
                self.learning_rate,
            )
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim + 1,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters())
                if self.conv_head is None
                else itertools.chain(
                    [self.logstd],
                    self.mean_net.parameters(),
                    self.conv_head.parameters(),
                ),
                self.learning_rate,
            )

        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim + 1,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters()
                if self.conv_head is None
                else itertools.chain(
                    self.baseline.parameters(), self.conv_head_baseline.parameters()
                ),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray, target_feature):
        if isinstance(obs, list):
            print("list")
            obs = np.array(obs)

        if not (len(obs.shape) == 1 or len(obs.shape) == 3):
            observation = obs
        else:
            observation = obs[None]
        if len(target_feature.shape) > 1:
            target_features = target_feature
        else:
            target_features = target_feature[:, None]

        # TODO return the action that the policy prescribes
        observation = ptu.from_numpy(observation)
        target_features = ptu.from_numpy(target_features)
        action = self(observation, target_features)
        return ptu.to_numpy(action.sample()), action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, obs: torch.FloatTensor, target_feature):
        if self.conv_head is not None:
            head_res = self.conv_head(obs)
            observation = torch.cat((head_res, target_feature), dim=1)
        else:
            observation = torch.cat((obs, target_feature), dim=1)
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution


#####################################################
#####################################################


class MLPPolicyPG(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, target_feature, q_values=None):
        # q_values (1005,)
        # advantages (1005,)
        # observations (1005, 4)
        # actions (1005,)

        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)
        if len(target_feature.shape) == 1:
            target_feature = target_feature[:, None]
        target_feature = ptu.from_numpy(target_feature)

        # TODO: update the policy using policy gradient
        # HINT1: Recall that the expression that we want to MAXIMIZE
        # is the expectation over collected trajectories of:
        # sum_{t=0}^{T-1} [grad [log pi(a_t|s_t) * (Q_t - b_t)]]
        # HINT2: you will want to use the `log_prob` method on the distribution returned
        # by the `forward` method

        action_logits = -self.forward(observations, target_feature).log_prob(actions)

        weighted_action_logits = torch.mul(action_logits, advantages)

        loss = torch.mean(weighted_action_logits)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.nn_baseline:
            ## TODO: update the neural network baseline using the q_values as
            ## targets. The q_values should first be normalized to have a mean
            ## of zero and a standard deviation of one.
            q_values = normalize(q_values, 0, 1)

            ## Note: You will need to convert the targets into a tensor using
            ## ptu.from_numpy before using it in the loss
            q_values = ptu.from_numpy(q_values)

            if self.conv_head_baseline is not None:
                conv_baseline = self.conv_head_baseline(observations)
                baselines = self.baseline(
                    torch.cat((conv_baseline, target_feature), dim=1)
                ).squeeze()
            else:
                baselines = self.baseline(
                    torch.cat((observations, target_feature), dim=1)
                ).squeeze()
            baseline_loss = self.baseline_loss(baselines, q_values)

            self.baseline_optimizer.zero_grad()
            baseline_loss.backward()
            self.baseline_optimizer.step()

        train_log = {
            "Training Loss": ptu.to_numpy(loss),
            "Baseline Loss": ptu.to_numpy(baseline_loss) if self.nn_baseline else 0,
        }
        return train_log

    def run_baseline_prediction(self, observations, target_feature):
        """
        Helper function that converts `observations` to a tensor,
        calls the forward method of the baseline MLP,
        and returns a np array

        Input: `observations`: np.ndarray of size [N, 1]
        Output: np.ndarray of size [N]

        """
        if len(target_feature.shape) == 1:
            target_feature = target_feature[:, None]
        observations = ptu.from_numpy(observations)
        target_feature = ptu.from_numpy(target_feature)
        if self.conv_head_baseline is not None:
            conv_baseline = self.conv_head_baseline(observations)
            pred = self.baseline(torch.cat((conv_baseline, target_feature), dim=1))
        else:
            pred = self.baseline(torch.cat((observations, target_feature), dim=1))
        return ptu.to_numpy(pred.squeeze())
