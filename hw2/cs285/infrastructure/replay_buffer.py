from cs285.infrastructure.utils import *


class ReplayBuffer(object):
    def __init__(self, max_size=1000000):

        self.max_size = max_size
        self.paths = []
        self.obs = None
        self.acs = None
        self.concatenated_rews = None
        self.unconcatenated_rews = None
        self.next_obs = None
        self.terminals = None

    def add_rollouts(self, paths, noised=False):

        # add new rollouts into our list of rollouts
        # print("adding path")
        for path in paths:
            self.paths.append(path)
            # self.paths = self.paths[-self.max_size//100:]
        self.paths = self.paths[-100:]
        # print("added path", len(self.paths), len(paths))

        # convert new rollouts into their component arrays, and append them onto our arrays
        # print("convert rollouts")
        (
            observations,
            actions,
            next_observations,
            terminals,
            concatenated_rews,
            unconcatenated_rews,
        ) = convert_listofrollouts(paths)
        # print("converted rollouts")

        if noised:
            observations = add_noise(observations)
            next_observations = add_noise(next_observations)
        # print("added noise")

        if self.obs is None:
            self.obs = observations[-self.max_size :]
            self.acs = actions[-self.max_size :]
            self.next_obs = next_observations[-self.max_size :]
            self.terminals = terminals[-self.max_size :]
            self.concatenated_rews = concatenated_rews[-self.max_size :]
            self.unconcatenated_rews = unconcatenated_rews[-self.max_size :]
        else:
            # print("obs", len(self.obs))
            self.obs = np.concatenate([self.obs, observations])[-self.max_size :]
            # print("acs")
            self.acs = np.concatenate([self.acs, actions])[-self.max_size :]
            # print("next_obs")
            self.next_obs = np.concatenate([self.next_obs, next_observations])[
                -self.max_size :
            ]
            # print("terminals")
            self.terminals = np.concatenate([self.terminals, terminals])[
                -self.max_size :
            ]
            # print("cat re")
            self.concatenated_rews = np.concatenate(
                [self.concatenated_rews, concatenated_rews]
            )[-self.max_size :]
            # print("uncat rew")
            if isinstance(unconcatenated_rews, list):
                # print(1)
                self.unconcatenated_rews += (
                    unconcatenated_rews  # TODO keep only latest max_size around
                )
            else:
                # print(2)
                self.unconcatenated_rews.append(
                    unconcatenated_rews
                )  # TODO keep only latest max_size around
            # print(len(self.unconcatenated_rews))
            self.unconcatenated_rews = self.unconcatenated_rews[-100:]
        # print("concatenated")

    ########################################
    ########################################

    def sample_random_rollouts(self, num_rollouts):
        rand_indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return self.paths[rand_indices]

    def sample_recent_rollouts(self, num_rollouts=1):
        return self.paths[-num_rollouts:]

    ########################################
    ########################################

    def sample_random_data(self, batch_size):

        assert (
            self.obs.shape[0]
            == self.acs.shape[0]
            == self.concatenated_rews.shape[0]
            == self.next_obs.shape[0]
            == self.terminals.shape[0]
        )
        rand_indices = np.random.permutation(self.obs.shape[0])[:batch_size]
        return (
            self.obs[rand_indices],
            self.acs[rand_indices],
            self.concatenated_rews[rand_indices],
            self.next_obs[rand_indices],
            self.terminals[rand_indices],
        )

    def sample_recent_data(self, batch_size=1, concat_rew=True):

        if concat_rew:
            return (
                self.obs[-batch_size:],
                self.acs[-batch_size:],
                self.concatenated_rews[-batch_size:],
                self.next_obs[-batch_size:],
                self.terminals[-batch_size:],
            )
        else:
            num_recent_rollouts_to_return = 0
            num_datapoints_so_far = 0
            index = -1
            while num_datapoints_so_far < batch_size:
                recent_rollout = self.paths[index]
                index -= 1
                num_recent_rollouts_to_return += 1
                num_datapoints_so_far += get_pathlength(recent_rollout)
            rollouts_to_return = self.paths[-num_recent_rollouts_to_return:]
            (
                observations,
                actions,
                next_observations,
                terminals,
                concatenated_rews,
                unconcatenated_rews,
            ) = convert_listofrollouts(rollouts_to_return)
            return (
                observations,
                actions,
                unconcatenated_rews,
                next_observations,
                terminals,
            )
