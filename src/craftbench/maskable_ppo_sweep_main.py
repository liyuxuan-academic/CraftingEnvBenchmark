from craftbench.maskable_ppo import benchmark_mskppo, DEFAULT_CONFIG
from crafting.task import RewardShaping, TaskObtainItem
import yaml
import wandb 

if __name__ == "__main__":
    
    with open('./maskable_ppo_sweep_configuration.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print("Sweep Config:\n", config)
    benchmark_mskppo(run_config = config)
