import numpy as np
import time
import torch
from cs285.feature_extractor.feature_extractor import FeatureExtractor
import copy
from cs285.infrastructure import pytorch_util as ptu
import pickle

############################################
############################################


def calculate_mean_prediction_error(env, action_sequence, models, data_statistics):

    model = models[0]

    # true
    true_states = perform_actions(env, action_sequence)["observation"]

    # predicted
    ob = np.expand_dims(true_states[0], 0)
    pred_states = []
    for ac in action_sequence:
        pred_states.append(ob)
        action = np.expand_dims(ac, 0)
        ob = model.get_prediction(ob, action, data_statistics)
    pred_states = np.squeeze(pred_states)

    # mpe
    mpe = mean_squared_error(pred_states, true_states)

    return mpe, true_states, pred_states


def perform_actions(env, actions):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    for ac in actions:
        obs.append(ob)
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        # add the observation after taking a step to next_obs
        next_obs.append(ob)
        rewards.append(rew)
        steps += 1
        # If the episode ended, the corresponding terminal value is 1
        # otherwise, it is 0
        if done:
            terminals.append(1)
            break
        else:
            terminals.append(0)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def mean_squared_error(a, b):
    return np.mean((a - b) ** 2)


############################################
############################################


def sample_trajectory(
    env,
    policy,
    max_path_length,
    render=False,
    render_mode=("rgb_array"),
    feature_extractor=None,
):
    # initialize env for the beginning of a new rollout
    ob = env.reset(seed=None)  # HINT: should be the output of resetting the env
    ob = np.transpose(ob[0], (2, 0, 1))
    target_feature = gen_random_feature(0, 1)

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:

        # render image of the simulated env
        if render:
            if hasattr(env, "sim"):
                image_obs.append(
                    env.sim.render(camera_name="track", height=500, width=500)[::-1]
                )
            else:
                image_obs.append(env.render())

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac, _ = policy.get_action(ob, np.array([target_feature]))
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _, _ = env.step(ac)
        ob = np.transpose(ob, (2, 0, 1))

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # TODO end the rollout if the rollout ended
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = done or steps > max_path_length  # HINT: this is either 0 or 1
        terminals.append(rollout_done)

        if rollout_done:
            break
    # # print(ptu.from_numpy(np.array(obs)), ptu.from_numpy(np.array(acs)).shaope)
    # print("get features")
    # features = feature_extractor(
    #     ptu.from_numpy(np.array(obs)), ptu.from_numpy(np.array(acs)[:, None])
    # ).squeeze()
    # # print(acs)
    # print(target_feature, torch.mean((features)))
    # # mse_features = ptu.to_numpy(10 * (target_feature - features) ** 2)
    # # for i in range(len(rewards)):
    # #     rewards[i] += 10 - (mse_features[i])
    # # 0 / 0

    # # - sum([ob[5] for ob in obs]) / len(obs)) ** 2,
    # #     len(obs),
    # # )
    # filename = ".pkl"
    # if target_feature == 0:
    #     print("saving 0")
    #     filename = "0.pkl"
    # else:
    #     print("saving 1")
    #     filename = "1.pkl"
    # with open(filename, "wb") as handle:
    #     pickle.dump(obs, handle)
    # print("finish get feature")
    return Path(obs, image_obs, acs, rewards, next_obs, terminals)


def sample_trajectories(
    env,
    policy,
    min_timesteps_per_batch,
    max_path_length,
    render=False,
    render_mode=("rgb_array"),
    feature_extractor=None,
):
    """
    Collect rollouts until we have collected min_timesteps_per_batch steps.

    TODO implement this function
    Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:

        path = sample_trajectory(
            env=env,
            policy=policy,
            max_path_length=max_path_length,
            render=render,
            feature_extractor=feature_extractor,
        )
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)

    return paths, timesteps_this_batch


def sample_n_trajectories(
    env,
    policy,
    ntraj,
    max_path_length,
    render=False,
    render_mode=("rgb_array"),
    feature_extractor=None,
):
    """
    Collect ntraj rollouts.

    TODO implement this function
    Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    paths = []

    for _ in range(ntraj):
        path = sample_trajectory(
            env=env,
            policy=policy,
            max_path_length=max_path_length,
            render=render,
            feature_extractor=feature_extractor,
        )
        paths.append(path)

    return paths


############################################
############################################


def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
    Take info (separate arrays) from a single rollout
    and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
    }


def convert_listofrollouts(paths):
    """
    Take a list of rollout dictionaries
    and return separate arrays,
    where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


############################################
############################################


def get_pathlength(path):
    return len(path["reward"])


def normalize(data, mean, std, eps=1e-8):
    return (data - mean) / (std + eps)


def unnormalize(data, mean, std):
    return data * std + mean


def add_noise(data_inp, noiseToSignal=0.01):

    data = copy.deepcopy(data_inp)  # (num data points, dim)

    # mean of data
    mean_data = np.mean(data, axis=0)

    # if mean is 0,
    # make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    # width of normal distribution to sample noise from
    # larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noiseToSignal
    for j in range(mean_data.shape[0]):
        data[:, j] = np.copy(
            data[:, j]
            + np.random.normal(0, np.absolute(std_of_noise[j]), (data.shape[0],))
        )

    return data


def gen_random_feature(feature_lower_bound=-1, feature_upper_bound=1.5):
    return np.random.randint(0, 2)
    # return (
    #     np.random.rand() * (feature_upper_bound - feature_lower_bound)
    #     + feature_lower_bound
    # )
