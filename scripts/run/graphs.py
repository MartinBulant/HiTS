import numpy as np
import logging

from graph_rl.graphs import HiTSGraph, HACGraph
from .interruption_policies.string_to_ip import get_ip_callable

LOG = logging.getLogger(__name__)


def create_graph(env, graph_params, run_params, subtask_specs, level_algo_kwargs_list):
    """Create and return graph based on parameters, subtask specs and mapping to environment goal."""

    graph_func_dict = {"HiTS": create_graph_hits, "HAC": create_graph_hac}

    assert graph_params["algorithm"] in graph_func_dict

    # construct graph
    graph = graph_func_dict[graph_params["algorithm"]](
        env=env,
        n_layers=graph_params["n_layers"],
        n_steps=run_params["n_steps"],
        subtask_specs=subtask_specs,
        level_algo_kwargs_list=level_algo_kwargs_list,
    )

    # add interruption policies if indicated in parameters
    nodes = list(reversed(graph.entry_node.get_descendants_recursively()))
    for l, node in zip(graph_params["level_params_list"], nodes):
        if "interruption_policy" in l and l["interruption_policy"] != "None":
            node.interruption_policy = get_ip_callable(l["interruption_policy"])

    return graph


def create_graph_hits(env, n_layers, n_steps, subtask_specs, level_algo_kwargs_list):
    """Create and return HiTS graph."""

    if "child_failure_penalty" not in level_algo_kwargs_list[-1]:
        level_algo_kwargs_list[-1]["child_failure_penalty"] = -subtask_specs[
            -1
        ]._max_n_actions

    # If buffer size not specified, set it such that all transitions will fit into buffer
    # (assuming that the episodes always last for env.max_episode_length which is not the
    # case in general).
    # This requires the env to have an attribute max_apisode_length.
    # NOTE: The calculation max number of episodes = n_steps/max_episode_length is optimistic because it
    # disregards the possibility of having much shorter episodes which still make use of the maximum
    # number of subgoals.

    max_episode_length = env.get_wrapper_attr("max_episode_length")
    for i in range(n_layers):
        if "buffer_size" not in level_algo_kwargs_list[i]:
            n_goals = level_algo_kwargs_list[i]["n_hindsight_goals"] + 1
            bs_factor = (
                level_algo_kwargs_list[i]["buffer_size_factor"]
                if "buffer_size_factor" in level_algo_kwargs_list[i]
                else 1.0
            )
            if i == n_layers - 1:
                level_algo_kwargs_list[i]["buffer_size"] = (
                    int(
                        bs_factor
                        * n_steps
                        / max_episode_length
                        * subtask_specs[i]._max_n_actions
                    )
                    * n_goals
                )
            else:
                # TODO: Can this be optimized?
                level_algo_kwargs_list[i]["buffer_size"] = (
                    int(bs_factor * n_steps) * n_goals
                )
            LOG.info(
                f"Buffer size level {i}: {level_algo_kwargs_list[i]['buffer_size']}"
            )
            if "buffer_size_factor" in level_algo_kwargs_list[i]:
                del level_algo_kwargs_list[i]["buffer_size_factor"]

    update_tsgs_rendering = env.get_wrapper_attr("update_timed_subgoals")

    graph = HiTSGraph(
        name="hits_graph",
        n_layers=n_layers,
        env=env,
        subtask_specs=subtask_specs,
        HAC_kwargs=level_algo_kwargs_list[-1],
        HiTS_kwargs=level_algo_kwargs_list[:-1],
        update_tsgs_rendering=update_tsgs_rendering,
    )

    return graph


def create_graph_hac(env, n_layers, n_steps, subtask_specs, level_algo_kwargs_list):
    """Create and return HAC graph."""

    # If buffer size not specified, set it such that all transitions will fit into buffer
    # (assuming that the episodes always last for env.max_episode_length which is not the
    # case in general).
    # This requires the env to have an attribute max_apisode_length.
    max_n_actions_list = [ss._max_n_actions for ss in subtask_specs]
    for i in range(n_layers):
        if "buffer_size" not in level_algo_kwargs_list[i]:
            n_goals = level_algo_kwargs_list[i]["n_hindsight_goals"] + 1
            bs_factor = (
                level_algo_kwargs_list[i]["buffer_size_factor"]
                if "buffer_size_factor" in level_algo_kwargs_list[i]
                else 1.0
            )
            level_algo_kwargs_list[i]["buffer_size"] = (
                int(
                    bs_factor
                    * n_steps
                    / env.max_episode_length
                    * np.prod(max_n_actions_list[i:])
                )
                * n_goals
            )
            LOG.info(
                f"Buffer size level {i}: {level_algo_kwargs_list[i]['buffer_size']}"
            )
            if "buffer_size_factor" in level_algo_kwargs_list[i]:
                del level_algo_kwargs_list[i]["buffer_size_factor"]

    update_sgs_rendering = env.get_wrapper_attr("update_subgoals")
    graph = HACGraph(
        name="hac_graph",
        n_layers=n_layers,
        env=env,
        subtask_specs=subtask_specs,
        HAC_kwargs=level_algo_kwargs_list,
        update_sgs_rendering=update_sgs_rendering,
    )

    return graph
