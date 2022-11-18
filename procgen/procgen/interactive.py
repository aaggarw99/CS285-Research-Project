#!/usr/bin/env python
import argparse
import pickle
from datetime import datetime
import numpy as np

from procgen import ProcgenGym3Env
from .env import ENV_NAMES
from gym3 import Interactive, VideoRecorderWrapper, unwrap

# last_terminal_step = 

class ProcgenInteractive(Interactive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_state = None
        self._last_episode_step = -1
        self.is_done = 0
        self.done_logging = False

        self.states = []
        self.actions = []
        self.rewards = []
        self.terminals = []


    def set_play_style(self, play_style):
        self.play_style = play_style
        self.game_log_time = datetime.today().isoformat()


    def _save_game_to_pickle(self):
        filename = f"../data_collection/starpilot/{self.play_style}_{self.game_log_time}"
        with open(filename, 'wb') as handle:
            game = {
                "states": np.stack(self.states),
                "actions": np.array(self.actions),
                "rewards": np.array(self.rewards),
                "terminals": np.array(self.terminals).astype(int)
            }
            pickle.dump(game, handle)

            print("Saved Game to " + filename)

            self.states = []
            self.actions = []
            self.rewards = []
            self.terminals = []
            self.done_logging = True


    def _update(self, dt, keys_clicked, keys_pressed):
        if "LEFT_SHIFT" in keys_pressed and "F1" in keys_clicked:
            print("save state")
            self._saved_state = unwrap(self._env).get_state()
            
        elif "F1" in keys_clicked:
            print("load state")
            if self._saved_state is not None:
                unwrap(self._env).set_state(self._saved_state)

        last_rew = self._last_rew
        last_state = self._env.get_info()[0]["rgb"]
        last_action = self._last_ac

        self.states.append(last_state)
        self.actions.append(last_action)
        self.rewards.append(last_rew)
        
        super()._update(dt, keys_clicked, keys_pressed)

        # The update function will give us the done signal. 
        self.terminals.append(self.is_done)

        if self.is_done and not self.done_logging:
            self._save_game_to_pickle()
        elif self.done_logging and not self.is_done:
            self.game_log_time = datetime.today().isoformat()
            self.done_logging = False


def make_interactive(vision, record_dir, play_style, **kwargs):
    info_key = None
    ob_key = None
    if vision == "human":
        info_key = "rgb"
        kwargs["render_mode"] = "rgb_array"
    else:
        ob_key = "rgb"

    env = ProcgenGym3Env(num=1, **kwargs)
    if record_dir is not None:
        env = VideoRecorderWrapper(
            env=env, directory=record_dir, ob_key=ob_key, info_key=info_key
        )
    h, w, _ = env.ob_space["rgb"].shape
    interactive = ProcgenInteractive(
        env,
        ob_key=ob_key,
        info_key=info_key,
        width=w * 12,
        height=h * 12,
    )
    interactive.set_play_style(play_style)
    return interactive


def main():
    default_str = "(default: %(default)s)"
    parser = argparse.ArgumentParser(
        description="Interactive version of Procgen allowing you to play the games"
    )
    parser.add_argument(
        "--vision",
        default="human",
        choices=["agent", "human"],
        help="level of fidelity of observation " + default_str,
    )
    parser.add_argument("--record-dir", help="directory to record movies to")
    parser.add_argument(
        "--distribution-mode",
        default="hard",
        help="which distribution mode to use for the level generation " + default_str,
    )
    parser.add_argument(
        "--env-name",
        default="coinrun",
        help="name of game to create " + default_str,
        choices=ENV_NAMES + ["coinrun_old"],
    )
    parser.add_argument(
        "--level-seed", type=int, help="select an individual level to use"
    )
    parser.add_argument(
        "--play-style",
        default="normal",
        help="aggressiveness when collecting data"
    )

    advanced_group = parser.add_argument_group("advanced optional switch arguments")
    advanced_group.add_argument(
        "--paint-vel-info",
        action="store_true",
        default=False,
        help="paint player velocity info in the top left corner",
    )
    advanced_group.add_argument(
        "--use-generated-assets",
        action="store_true",
        default=False,
        help="use randomly generated assets in place of human designed assets",
    )
    advanced_group.add_argument(
        "--uncenter-agent",
        action="store_true",
        default=False,
        help="display the full level for games that center the observation to the agent",
    )
    advanced_group.add_argument(
        "--disable-backgrounds",
        action="store_true",
        default=False,
        help="disable human designed backgrounds",
    )
    advanced_group.add_argument(
        "--restrict-themes",
        action="store_true",
        default=False,
        help="restricts games that use multiple themes to use a single theme",
    )
    advanced_group.add_argument(
        "--use-monochrome-assets",
        action="store_true",
        default=False,
        help="use monochromatic rectangles instead of human designed assets",
    )

    args = parser.parse_args()
    kwargs = {
        "paint_vel_info": args.paint_vel_info,
        "use_generated_assets": args.use_generated_assets,
        "center_agent": not args.uncenter_agent,
        "use_backgrounds": not args.disable_backgrounds,
        "restrict_themes": args.restrict_themes,
        "use_monochrome_assets": args.use_monochrome_assets,
    }
    if args.env_name != "coinrun_old":
        kwargs["distribution_mode"] = args.distribution_mode
    if args.level_seed is not None:
        kwargs["start_level"] = args.level_seed
        kwargs["num_levels"] = 1

    ia = make_interactive(
        args.vision, record_dir=args.record_dir, play_style=args.play_style, env_name=args.env_name, **kwargs
    )
    ia.run()


if __name__ == "__main__":
    main()
