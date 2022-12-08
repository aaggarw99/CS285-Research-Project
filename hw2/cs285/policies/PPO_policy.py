from torch import nn
import torch

from cs285.infrastructure import pytorch_util as ptu
from cs285.infrastructure.utils import normalize
from cs285.policies.MLP_policy import MLPPolicy


class MLPPolicyPPO(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, epsilon=0.2, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        self.epsilon = epsilon
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions, advantages, target_feature, q_values=None, old_logprobs=None):
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
        dist = self.forward(observations, target_feature)
        logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        ratios = torch.exp(logprobs - old_logprobs.detach())
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages

        loss = -torch.mean(torch.minimum(surr1, surr2))

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


    # def load_weights(
    #         self, 
    #         action_logits_params, 
    #         conv_head_params, 
    #         baseline_params, 
    #         conv_head_baseline_params
    #         ):
    #     self.logits_na.load_state_dict(action_logits_params)
    #     self.conv_head.load_state_dict(conv_head_params)
    #     self.baseline_params.load_state_dict(baseline_params)
    #     self.conv_head_baseline.load_state_dict(conv_head_baseline_params)
