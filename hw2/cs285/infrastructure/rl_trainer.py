from collections import OrderedDict
import pickle
import os
import sys
import time

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu

from cs285.infrastructure import utils
from cs285.feature_extractor.feature_extractor import FeatureExtractor
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.action_noise_wrapper import ActionNoiseWrapper

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):
    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params["logdir"])

        # Set random seeds
        seed = self.params["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(use_gpu=not self.params["no_gpu"], gpu_id=self.params["which_gpu"])

        #############
        ## ENV
        #############

        # Make the gym environment
        if self.params["video_log_freq"] == -1:
            render_mode = None
        else:
            render_mode = "rgb_array"
        self.env = gym.make(
            self.params["env_name"],
            render_mode=render_mode,
            mode=7,
            # forward_reward_weight=0,
            # ctrl_cost_weight=0,
            # healthy_reward=1.3,
        )
        self.env.seed(seed)

        # Add noise wrapper
        if params["action_noise_std"] > 0:
            self.env = ActionNoiseWrapper(self.env, seed, params["action_noise_std"])

        # import plotting (locally if 'obstacles' env)
        if not (self.params["env_name"] == "obstacles-cs285-v0"):
            import matplotlib

            matplotlib.use("Agg")

        # Maximum length for episodes
        self.params["ep_len"] = self.params["ep_len"] or self.env.spec.max_episode_steps
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params["ep_len"]

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params["agent_params"]["discrete"] = discrete

        # Observation and action sizes

        ob_dim = (
            self.env.observation_space.shape
            if img
            else self.env.observation_space.shape[0]
        )
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params["agent_params"]["ac_dim"] = ac_dim
        self.params["agent_params"]["ob_dim"] = ob_dim

        # simulation timestep, will be used for video saving
        if "model" in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        elif "env_wrappers" in self.params:
            self.fps = 30  # This is not actually used when using the Monitor wrapper
        elif "video.frames_per_second" in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata["video.frames_per_second"]
        else:
            self.fps = 10

        #############
        ## AGENT
        #############

        agent_class = self.params["agent_class"]
        self.agent = agent_class(self.env, self.params["agent_params"])

    def run_training_loop(
        self,
        n_iter,
        collect_policy,
        eval_policy,
        initial_expertdata=None,
        relabel_with_expert=False,
        start_relabel_with_expert=1,
        expert_policy=None,
    ):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        load_saved_model = True

        # train feature extractor or load it from saved model
        feature_extractor = FeatureExtractor(1, (210, 160))
        if load_saved_model:
            feature_extractor.load_state_dict(torch.load("models/berzerk.pt"))
        else:
            optimizer = torch.optim.Adam(feature_extractor.parameters(), lr=1e-3)
            n_epoch = 1
            criterion = torch.nn.BCELoss(reduction="mean")
            batch_size = 128

            # read data
            print("Reading Data")
            import pickle

            with open("data_collection/atari/agressive.pickle", "rb") as file:
                f = pickle.load(file)
            obs = f["state"]
            acs = f["action"]
            labels = np.zeros(f["label"].shape)
            print(obs.shape[0], f["label"][0])
            num_sample = obs.shape[0]
            test_obs = obs[num_sample - 25 * batch_size :]
            test_acs = acs[num_sample - 25 * batch_size :]
            test_labels = labels[num_sample - 25 * batch_size :]
            obs = obs[: num_sample - 25 * batch_size]
            print(obs.shape[0])
            acs = acs[: num_sample - 25 * batch_size]
            labels = labels[: num_sample - 25 * batch_size]
            print(f"train samples: {obs.shape[0]} test samples: {test_obs.shape[0]}")
            print(f"Successfully Read Agressive Data")
            with open("data_collection/atari/passive.pickle", "rb") as file:
                f = pickle.load(file)
            obs_temp = f["state"]
            acs_temp = f["action"]
            labels_temp = np.ones(f["label"].shape)
            print(f["label"][0])
            num_sample = obs_temp.shape[0]
            test_obs = np.concatenate(
                (obs_temp[num_sample - 25 * batch_size :], test_obs), axis=0
            )
            test_acs = np.concatenate(
                (acs_temp[num_sample - 25 * batch_size :], test_acs), axis=0
            )
            test_labels = np.concatenate(
                (labels_temp[num_sample - 25 * batch_size :], test_labels), axis=0
            )

            obs = np.concatenate(
                (obs_temp[: num_sample - 25 * batch_size], obs), axis=0
            )
            acs = np.concatenate(
                (acs_temp[: num_sample - 25 * batch_size], acs), axis=0
            )
            labels = np.concatenate(
                (labels_temp[: num_sample - 25 * batch_size], labels), axis=0
            )
            print(f"train samples: {obs.shape[0]} test samples: {test_obs.shape[0]}")
            print(f"Successfully Read Passive Data")

            # shuffle the data
            test_obs = ptu.from_numpy(np.transpose(test_obs, (0, 3, 1, 2)))
            test_acs = ptu.from_numpy(test_acs)
            test_labels = ptu.from_numpy(test_labels)
            rand_indices = np.random.permutation(obs.shape[0])
            obs = np.transpose(obs[rand_indices], (0, 3, 1, 2))
            acs = acs[rand_indices]
            labels = labels[rand_indices]
            num_sample = obs.shape[0]
            best_loss = float("inf")
            for i in range(n_epoch):  # num epochs
                for batch in range(0, obs.shape[0], batch_size):
                    print(batch, obs.shape[0])
                    batched_obs = ptu.from_numpy(obs[batch : batch + batch_size])
                    batched_acs = ptu.from_numpy(acs[batch : batch + batch_size])
                    batched_labels = ptu.from_numpy(labels[batch : batch + batch_size])
                    optimizer.zero_grad()
                    # print("Get Prediction")
                    prediction = feature_extractor(batched_obs, batched_acs)
                    # print("Calculate Loss")
                    loss = criterion(prediction.squeeze(), batched_labels.squeeze())
                    # print("step")
                    loss.backward()
                    optimizer.step()
                    # print("Calculate Test Loss")
                    prediction = feature_extractor(test_obs, test_acs)
                    loss = criterion(prediction.squeeze(), test_labels.squeeze())
                    if loss < best_loss:
                        print("save")
                        torch.save(feature_extractor.state_dict(), "models/berzerk.pt")
                        best_loss = loss
                    print(
                        f"{i}: loss:{loss}, correct pred {torch.sum(torch.where(torch.round(prediction)==test_labels, 1, 0))}/{torch.sum(torch.ones(prediction.shape))}"
                    )

        feature_extractor.eval()
        self.feature_extractor = feature_extractor
        self.agent.set_feature_extractor(feature_extractor)
        # 0 / 0
        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************" % itr)

            # decide if videos should be rendered/logged at this iteration
            if (
                itr % self.params["video_log_freq"] == 0
                and self.params["video_log_freq"] != -1
            ):
                self.logvideo = True
            else:
                self.logvideo = False

            # decide if metrics should be logged
            if self.params["scalar_log_freq"] == -1:
                self.logmetrics = False
            elif itr % self.params["scalar_log_freq"] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            # print("collecting trajs")
            training_returns = self.collect_training_trajectories(
                itr, initial_expertdata, collect_policy, self.params["batch_size"]
            )
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            # print("adding to buffer")
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            # print("training")
            train_logs = self.train_agent()

            # log/save
            if self.logvideo or self.logmetrics:
                # perform logging
                print("\nBeginning logging procedure...")
                self.perform_logging(
                    itr, paths, eval_policy, train_video_paths, train_logs
                )

                if self.params["save_params"]:
                    self.agent.save(
                        "{}/agent_itr_{}.pt".format(self.params["logdir"], itr)
                    )

    ####################################
    ####################################

    def collect_training_trajectories(
        self,
        itr,
        load_initial_expertdata,
        collect_policy,
        batch_size,
    ):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        # TODO decide whether to load training data or use the current policy to collect more data
        # HINT: depending on if it's the first iteration or not, decide whether to either
        # (1) load the data. In this case you can directly return as follows
        # ``` return loaded_paths, 0, None ```

        # (2) collect `self.params['batch_size']` transitions

        # if itr == 0 and load_initial_expertdata:
        if itr == 0:
            with open("data_collection/atari/agressive.pickle", "rb") as file:
                f = pickle.load(file)
            print(
                f["state"].shape,
                f["action"].shape,
                f["reward"].shape,
                f["terminal"].shape,
            )
            obs = np.transpose(f["state"], (0, 3, 1, 2))
            obs = [obs[i] for i in range(obs.shape[0])]
            acs = [f["action"][i][0] for i in range(f["action"].shape[0])]
            rews = [f["reward"][i][0] for i in range(f["reward"].shape[0])]
            term = [f["terminal"][i][0] for i in range(f["terminal"].shape[0])]
            paths = []
            old_idx = 0
            count = 3
            for i in range(len(term)):
                if term[i] == 1:
                    count -= 1
                    paths.append(
                        utils.Path(
                            obs[old_idx : i + 1],
                            obs[old_idx : i + 1],
                            acs[old_idx : i + 1],
                            rews[old_idx : i + 1],
                            obs[old_idx : i + 1],
                            term[old_idx : i + 1],
                        )
                    )
                    if count == 0:
                        break
            print(len(paths))
            return (
                paths,
                0,
                None,
            )
            # with open(load_initial_expertdata, "rb") as f:
            #     loaded_paths = pickle.loads(f.read())

            # return loaded_paths, 0, None

        # TODO collect `batch_size` samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = utils.sample_trajectories(
            self.env,
            collect_policy,
            batch_size,
            self.params["ep_len"],
            False,
            feature_extractor=self.feature_extractor,
        )

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.logvideo:
            print("\nCollecting train rollouts to be used for saving videos...")
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = utils.sample_n_trajectories(
                self.env,
                collect_policy,
                MAX_NVIDEO,
                MAX_VIDEO_LEN,
                True,
                feature_extractor=self.feature_extractor,
            )

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        print("\nTraining agent using sampled data from replay buffer...")
        all_logs = []
        for train_step in range(self.params["num_agent_train_steps_per_iter"]):

            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            (
                ob_batch,
                ac_batch,
                re_batch,
                next_ob_batch,
                terminal_batch,
                target_feature_batch,
            ) = self.agent.sample(self.params["train_batch_size"])

            # TODO use the sampled data to train an agent
            # HINT: use the agent's train function
            # HINT: keep the agent's training log for debugging
            train_log = self.agent.train(
                ob_batch,
                ac_batch,
                re_batch,
                next_ob_batch,
                terminal_batch,
                target_feature_batch,
            )
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
            self.env,
            eval_policy,
            self.params["eval_batch_size"],
            self.params["ep_len"],
            feature_extractor=self.feature_extractor,
        )

        # save eval rollouts as videos in tensorboard event file
        if self.logvideo and train_video_paths != None:
            print("\nCollecting video rollouts eval")
            eval_video_paths = utils.sample_n_trajectories(
                self.env,
                eval_policy,
                MAX_NVIDEO,
                MAX_VIDEO_LEN,
                True,
                feature_extractor=self.feature_extractor,
            )

            # save train/eval videos
            print("\nSaving train rollouts as videos...")
            self.logger.log_paths_as_videos(
                train_video_paths,
                itr,
                fps=self.fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="train_rollouts",
            )
            self.logger.log_paths_as_videos(
                eval_video_paths,
                itr,
                fps=self.fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )

        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                self.logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            self.logger.flush()
