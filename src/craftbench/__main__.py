from craftbench.maskable_ppo import benchmark_mskppo


def main():
    # print(f"{'CraftingEnvBenchmark': -^40}")
    print("CraftingEnvBenchmark")
    benchmark_mskppo(save_req_graph=True, save_sol_graph=True)
