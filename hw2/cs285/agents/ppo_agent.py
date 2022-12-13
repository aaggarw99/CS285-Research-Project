import torch

from .base_agent import BaseAgent
from .pg_agent import PGAgent
from cs285.policies.PPO_policy import MLPPolicyPPO
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import normalize, gen_random_feature
from cs285.infrastructure import pytorch_util as ptu


class PPOAgent(PGAgent):
    """
    Need to implement:
        1. Minibatching, and taking the gradients with respect to minibatch. 
        2. Caching the previous logprobs... 



    """
    def __init__(self, env, agent_params):

        # init vars
        self.env = env
        self.feature_extractor = None
        self.agent_params = agent_params
        self.gamma = self.agent_params["gamma"]
        self.standardize_advantages = self.agent_params["standardize_advantages"]
        self.nn_baseline = self.agent_params["nn_baseline"]
        self.reward_to_go = self.agent_params["reward_to_go"]
        self.gae_lambda = self.agent_params["gae_lambda"]

        self.mini_batch_size = self.agent_params["train_mini_batch_size"]

        # actor/policy
        self.actor = MLPPolicyPPO(
            self.agent_params["ac_dim"],
            self.agent_params["ob_dim"],
            self.agent_params["n_layers"],
            self.agent_params["size"],
            discrete=self.agent_params["discrete"],
            learning_rate=self.agent_params["learning_rate"],
            nn_baseline=self.agent_params["nn_baseline"],
            epsilon=self.agent_params["ppo_epsilon"],
            load_model_path=self.agent_params["load_model_path"]
        )
        # For evaluating reward ratios. 
        self.old_actor = MLPPolicyPPO(
            self.agent_params["ac_dim"],
            self.agent_params["ob_dim"],
            self.agent_params["n_layers"],
            self.agent_params["size"],
            discrete=self.agent_params["discrete"],
            learning_rate=self.agent_params["learning_rate"],
            nn_baseline=self.agent_params["nn_baseline"],
            epsilon=self.agent_params["ppo_epsilon"],
            load_model_path=self.agent_params["load_model_path"]
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['replay_buffer_size'])

    def set_feature_extractor(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def train(
        self,
        observations,
        actions,
        rewards_list,
        next_observations,
        terminals,
        target_feature,
    ):

        """
        Training a PG agent refers to updating its actor using the given observations/actions
        and the calculated qvals/advantages that come from the seen rewards.
        """

        # TODO: update the PG actor/policy using the given batch of data
        # using helper functions to compute qvals and advantages, and
        # return the train_log obtained from updating the policy

        q_values = self.calculate_q_vals(rewards_list, terminals)
        advantages = self.estimate_advantage(
            observations, rewards_list, q_values, terminals, target_feature
        )
        
        n_mini_batches = observations.shape[0] // self.mini_batch_size
        training_loss = 0
        baseline_loss = 0

        for i in range(n_mini_batches):
            start, end = i * self.mini_batch_size, (i + 1) * self.mini_batch_size
            _, action_dist = self.old_actor.get_action(
                observations[start:end],
                target_feature[start:end]
            )
            old_logprobs = action_dist.log_prob(ptu.from_numpy(actions[start:end]))
            log = self.actor.update(
                observations[start:end],
                actions[start:end],
                advantages[start:end],
                target_feature[start:end],
                q_values[start:end],
                old_logprobs=old_logprobs
            )
            training_loss += log["Training Loss"]
            baseline_loss += log["Baseline Loss"]

        self.old_actor.load_state_dict(self.actor.state_dict())

        train_log = {
            "Training Loss": (training_loss / n_mini_batches),
            "Baseline Loss": (baseline_loss / n_mini_batches)
        }

        return train_log


    def save(self, path):
        torch.save(self.actor.state_dict(), path)
