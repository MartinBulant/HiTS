import numpy as np
import inspect
import logging
from typing import Any

from gymnasium import spaces
from graph_rl.envs.goal_env import GoalEnv
from hac_envs.environment import Environment

LOG = logging.getLogger(__name__)


class GymWrapper(GoalEnv):
    """Wraps HAC environment in gym environment.

    Assumes hac_env is an instance of Environment as defined in the
    original Hierarchical Actor-Critic implementation at
    https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-
    """

    def __init__(self, hac_env: Environment, observation_space_bounds=None):
        self.hac_env: Environment = hac_env
        self.max_episode_length: int = hac_env.max_actions
        self.viewer = None

        # action space
        action_low = np.astype(
            hac_env.action_offset - hac_env.action_bounds, np.float32
        )
        action_high = np.astype(
            hac_env.action_offset + hac_env.action_bounds, np.float32
        )
        self.action_space = spaces.Box(
            low=action_low, high=action_high, dtype=np.float32
        )

        # partial observation space
        if observation_space_bounds is None:
            # appropriate for UR5 and Pendulum
            partial_obs_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(hac_env.state_dim,), dtype=np.float32
            )
        else:
            partial_obs_space = spaces.Box(
                low=observation_space_bounds[:, 0],
                high=observation_space_bounds[:, 1],
                dtype=np.float32,
            )

        # goal spaces (Use goal space used for training in original paper)
        # desired goal
        goal_low = np.array(hac_env.goal_space_train, dtype=np.float32)[:, 0]
        goal_high = np.array(hac_env.goal_space_train, dtype=np.float32)[:, 1]
        desired_goal_space = spaces.Box(low=goal_low, high=goal_high, dtype=np.float32)

        # achieved goal
        achieved_low = np.array(hac_env.subgoal_bounds, dtype=np.float32)[:, 0]
        achieved_high = np.array(hac_env.subgoal_bounds, dtype=np.float32)[:, 1]
        achieved_goal_space = spaces.Box(
            low=achieved_low, high=achieved_high, dtype=np.float32
        )

        # observation space, including desired and achieved goal
        self.observation_space = spaces.Dict(
            {
                "observation": partial_obs_space,
                "desired_goal": desired_goal_space,
                "achieved_goal": achieved_goal_space,
            }
        )

        self.reset()

    def _get_obs(self, state: np.ndarray) -> dict[str, np.ndarray]:
        achieved_goal = self.hac_env.project_state_to_end_goal(self.hac_env.sim, state)

        obs = {
            "observation": np.array(state, dtype=np.float32),
            "desired_goal": np.array(self.desired_goal, dtype=np.float32),
            "achieved_goal": np.array(achieved_goal, dtype=np.float32),
        }

        return obs

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict
    ):
        tolerance = self.hac_env.end_goal_thresholds
        reward = 0.0
        for a_goal, d_goal, tol in zip(achieved_goal, desired_goal, tolerance):
            if np.absolute(a_goal - d_goal) > tol:
                reward = -1.0
                break
        return reward

    def step(self, action):
        info = {}
        state = self.hac_env.execute_action(action)
        self.n_steps += 1

        info["n_steps"] = self.n_steps
        obs = self._get_obs(state)
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)

        terminated = reward == 0.0
        truncated = self.n_steps >= self.max_episode_length
        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        options = {} if options is None else options
        self.n_steps = 0
        self.desired_goal = self.hac_env.get_next_goal(test=options.get("test", False))
        state = self.hac_env.reset_sim()

        self.hac_env.display_end_goal(self.desired_goal)
        obs = self._get_obs(state)

        return obs, {"n_steps": self.n_steps}

    def update_subgoals(self, subgoals):
        self.hac_env.display_subgoals(subgoals + [None])

    def update_timed_subgoals(self, timed_subgoals, tolerances):
        subgoals = [tg.goal for tg in timed_subgoals if tg is not None]
        # NOTE: Visualization of time component of timed subgoals is not supported
        # by HAC environments.
        self.update_subgoals(subgoals)

    def render(self, mode=None):
        return self.hac_env.render()
