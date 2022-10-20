import numpy as np

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import normalize


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # TODO: update the PG actor/policy using the given batch of data 
        # using helper functions to compute qvals and advantages, and
        # return the train_log obtained from updating the policy

        q_values = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_values, terminals)

        # print(q_values.shape, advantages.shape, observations.shape, actions.shape, next_observations.shape)
        # print(q_values, advantages)
        train_log = self.actor.update(observations, actions, advantages, q_values)

        return train_log

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # TODO: return the estimated qvals based on the given rewards, using
            # either the full trajectory-based estimator or the reward-to-go
            # estimator

        # Note: rewards_list is a list of lists of rewards with the inner list
        # being the list of rewards for a single trajectory.
        
        # HINT: use the helper functions self._discounted_return and
        # self._discounted_cumsum (you will need to implement these).

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory

        # Note: q_values should first be a 2D list where the first dimension corresponds to 
        # trajectories and the second corresponds to timesteps, 
        # then flattened to a 1D numpy array.

        q_values = []

        # q_values strictly is a flattened array of all q_values
        # when normalizing comes, we look at the length of rewards_list to find the length
        # of each trajectory
        if not self.reward_to_go:
            for rewards in rewards_list:
              q_values.append(np.array(self._discounted_return(rewards)).astype(float))

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            for rewards in rewards_list:
              q_values.append(np.array(self._discounted_cumsum(rewards)).astype(float))

        q_values = np.array(q_values)
        q_values = np.hstack(q_values)
        
        return q_values

    def estimate_advantage(self, obs: np.ndarray, rews_list: np.ndarray, q_values: np.ndarray, terminals: np.ndarray):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_unnormalized.ndim == q_values.ndim
            ## TODO: values were trained with standardized q_values, so ensure
                ## that the predictions have the same mean and standard deviation as
                ## the current batch of q_values
            values = normalize(values_unnormalized, q_values.mean(), q_values.std())

            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rews = np.concatenate(rews_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for t in reversed(range(batch_size)):
                    ## TODO: recursively compute advantage estimates starting from
                        ## timestep T.
                    ## HINT: use terminals to handle edge cases. terminals[i]
                        ## is 1 if the state is the last in its trajectory, and
                        ## 0 otherwise.

                    if terminals[t]:  # start of traj
                      advantages[t] = rews[t] - values[t]

                    else:
                      advantages[t] = (rews[t] + self.gamma * values[t + 1] - values[t]) + (self.gae_lambda * self.gamma * advantages[t + 1])

                advantages = advantages[:-1]

            else:
                ## TODO: compute advantage estimates using q_values, and values as baselines
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages to have a mean of zero
        # and a standard deviation of one
        if self.standardize_advantages:
            advantages = normalize(advantages, 0, 1)

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        discounted_return = 0

        for t, r_t in enumerate(rewards):
          discounted_return += np.power(self.gamma, t) * r_t

        list_of_discounted_returns = [discounted_return] * len(rewards)

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        list_of_discounted_cumsums = []

        for t in range(len(rewards)):
          t_sum = 0
          for t_prime in range(t, len(rewards)):
            r_t_prime = rewards[t_prime]
            t_sum += np.power(self.gamma, t_prime - t) * r_t_prime

          list_of_discounted_cumsums.append(t_sum)

        return list_of_discounted_cumsums
