import gym
from gym.utils.play import play
import numpy as np
import pickle

style = "agressive"

observations = None
actions = None
rewards = None
terminal = None

current_observations = None
current_actions = None
current_rewards = None
current_terminal = None


def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    global current_observations
    global current_actions
    global current_rewards
    global current_terminal
    global observations
    global actions
    global rewards
    global terminal
    # print(len(obs_t))
    if type(obs_t) is tuple:
        obs_t = obs_t[0]
    action = [action]
    rew = [rew]
    if terminated:
        if current_terminal.shape[0] > 1000:
            if actions is None:
                observations = current_observations
                actions = current_actions
                rewards = current_rewards
                terminal = current_terminal
            else:
                observations = np.concatenate(
                    (observations, current_observations), axis=0
                )
                actions = np.concatenate((actions, current_actions), axis=0)
                rewards = np.concatenate((rewards, current_rewards), axis=0)
                terminal = np.concatenate((terminal, current_terminal), axis=0)

        current_observations = None
        current_actions = None
        current_rewards = None
        current_terminal = None
    if current_actions is None:
        current_observations = np.expand_dims(np.asarray(obs_t), axis=0)
        current_actions = np.expand_dims(np.asarray(action), axis=0)
        current_rewards = np.expand_dims(np.asarray(rew), axis=0)
        current_terminal = np.asarray([[1]]) if terminated else np.asarray([[0]])
    else:
        print(
            current_observations.shape,
            current_actions.shape,
            current_rewards.shape,
            current_terminal.shape,
        )
        current_observations = np.concatenate(
            (current_observations, np.expand_dims(np.asarray(obs_t), axis=0)), axis=0
        )
        current_actions = np.concatenate(
            (current_actions, np.expand_dims(np.asarray(action), axis=0)), axis=0
        )
        current_rewards = np.concatenate(
            (current_rewards, np.expand_dims(np.asarray(rew), axis=0)), axis=0
        )
        current_terminal = np.concatenate(
            (current_terminal, np.asarray([[1]]) if terminated else np.asarray([[0]])),
            axis=0,
        )


env = gym.make("Berzerk-v4", render_mode="rgb_array")
play(env, zoom=3, callback=callback)

data = {
    "state": observations,
    "action": actions,
    "reward": rewards,
    "terminal": terminal,
    "label": np.asarray(
        [[1] for i in range(observations.shape[0])]
        if style == "agressive"
        else [[0] for i in range(observations.shape[0])]
    ),
}


with open(f"{style}.pickle", "wb") as handle:
    pickle.dump(data, handle)
