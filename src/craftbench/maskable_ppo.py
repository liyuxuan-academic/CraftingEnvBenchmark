import time


from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO

from hebg.metrics.complexity import learning_complexity
from hebg.metrics.histograms import behaviors_histograms
from hebg.behavior import Behavior

import wandb

from crafting.task import RewardShaping, TaskObtainItem

from craftbench.wandbench import WandbCallback
from craftbench.plots import save_requirement_graph, save_heb_graph

from craftbench.make_env import make_env, record_wrap_env

PROJECT = "neural_network_size_sweep"

DEFAULT_CONFIG = {
    "agent": "MaskablePPO",
    "agent_seed": 0,
    "policy_type": "MlpPolicy",
    "pi_n_layers": 3,
    "pi_units_per_layer": 64,
    "vf_n_layers": 3,
    "vf_units_per_layer": 64,
    "total_timesteps": 1e6,
    "max_n_consecutive_successes": 200,
    "env_name": "RandomCrafting-v1",
    "env_seed": 1,
    "task_seed": None,
    "task_complexity": 243,
    "reward_shaping": RewardShaping.ALL_USEFUL.value,
    "max_episode_steps": 200,
    "time_factor": 2.0,
    "n_items": 20,
    "n_tools": 0,
    "n_findables": 1,
    "n_zones": 1,
}

# Define sweep config
sweep_configuration = {
    'method': 'grid',
    'name': PROJECT, 
    'metric': {'goal': 'maximize', 'name': 'mean_ep_return'}, 
    'parameters': 
    {   
        'pi_n_layers': {'values': [1, 2, 3, 4, 5, 6, 7, 8]}, 
        'pi_units_per_layer': {'values': [4, 8, 16, 32, 64, 128, 256]}, 
        'vf_n_layers': {'values': [1, 2, 3, 4, 5, 6, 7, 8]}, 
        'vf_units_per_layer': {'values': [4, 8, 16, 32, 64, 128, 256]}, 
        'agent': {'value': 'MaskablePPO'}, 
        'agent_seed': {'value': 0}, 
        'policy_type': {'value': 'MlpPolicy'}, 
        'total_timesteps': {'value': 1e6}, 
        'max_n_consecutive_successes': {'value': 200}, 
        'env_name': {'value': 'RandomCrafting-v1'}, 
        'env_seed': {'value': 1}, 
        'task_seed': {'value': None}, 
        'task_complexity': {'value': 243}, 
        'reward_shaping': {'value': RewardShaping.ALL_USEFUL.value},
        'max_episode_steps': {'value': 200},
        'time_factor': {'value': 2.0}, 
        'n_items': {'value': 20}, 
        'n_tools': {'value': 0}, 
        'n_findables': {'value': 1}, 
        'n_zones': {'value': 1}
    }
}


# Temporary solution for pygame "No available video device"
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

def benchmark_mskppo(
    save_req_graph: bool = False,
    save_sol_graph: bool = False,
):
    run = wandb.init(project=PROJECT, monitor_gym=True)
    print(wandb.config)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dirname = f"{timestamp}-{run.id}"
    config = wandb.config
    params_logs = {}

    # Build env
    crafting_env = make_env(config)
    video_path = f"videos/{run.id}"
    env = record_wrap_env(crafting_env, video_path)
    task: TaskObtainItem = crafting_env.tasks[0]  # Assume only one task
    params_logs["task"] = str(task)

    if save_req_graph:
        # Get & save requirements graph
        requirement_graph_path = save_requirement_graph(
            run_dirname,
            crafting_env.world,
            title=str(crafting_env.world),
            figsize=(32, 18),
        )
        params_logs["requirement_graph"] = wandb.Image(requirement_graph_path)

    # Get solving behavior
    all_behaviors = crafting_env.world.get_all_behaviors()
    all_behaviors_list = list(all_behaviors.values())
    solving_behavior: Behavior = all_behaviors[f"Get {task.goal_item}"]
    params_logs["solving_behavior"] = str(solving_behavior)

    # Save goal solving graph
    if save_sol_graph:
        solving_heb_graph_path = save_heb_graph(solving_behavior, run_dirname)
        params_logs["solving_heb_graph"] = wandb.Image(solving_heb_graph_path)

    # Compute complexities
    used_nodes_all = behaviors_histograms(all_behaviors_list)
    lcomp, comp_saved = learning_complexity(solving_behavior, used_nodes_all)
    print(
        f"BEHAVIOR: {str(solving_behavior)}:"
        f"Complexities total={lcomp + comp_saved},"
        f" saved={comp_saved}, learn={comp_saved}"
    )
    params_logs.update(
        {
            "learning_complexity": lcomp,
            "total_complexity": lcomp + comp_saved,
            "saved_complexity": comp_saved,
        }
    )

    # Build neural networks architecture from config
    pi_arch = [config["pi_units_per_layer"] for _ in range(config["pi_n_layers"])]
    vf_arch = [config["vf_units_per_layer"] for _ in range(config["vf_n_layers"])]
    net_arch = [dict(pi=pi_arch, vf=vf_arch)]

    # Build agent
    agent = MaskablePPO(
        config["policy_type"],
        env,
        policy_kwargs={"net_arch": net_arch},
        seed=config["agent_seed"],
        verbose=1,
    )

    wandb.log(params_logs)

    # pylint: disable=unexpected-keyword-arg
    agent.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            verbose=2,
            max_n_consecutive_successes=config["max_n_consecutive_successes"],
        ),
    )

    run.finish()

def start_wandb_agent():
    wandb.agent("ny7b5y0x", function=benchmark_mskppo, count=1, project=PROJECT)

if __name__ == "__main__":
    # benchmark_mskppo()

    # Initialize sweep by passing in config.
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT)
    # print(sweep_id)

    start_wandb_agent()