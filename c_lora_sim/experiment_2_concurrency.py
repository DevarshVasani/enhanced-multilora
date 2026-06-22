import copy
import matplotlib.pyplot as plt
from c_lora_sim.clora_job import CLoraJob
from c_lora_sim.clora_env import CLoraEnv
from c_lora_sim.clora_rl_agent import CLoraPPOAgent

def generate_infinite_workload(num_jobs: int = 200) -> list[CLoraJob]:
    # All same base model to isolate the concurrency effect
    jobs = []
    for i in range(num_jobs):
        job = CLoraJob(
            job_id=i,
            base_model_id="Llama-3",
            adapter_rank=16,
            tokens_to_process=10000,
            job_type="trainer",
            arrival_time=0.0
        )
        jobs.append(job)
    return jobs

def run_scheduler_with_cap(env: CLoraEnv, jobs: list[CLoraJob], N_cap: int):
    env.reset(jobs)
    
    # We measure total time to finish all jobs
    while True:
        state, done = env.step()
        if done:
            break
            
        actions = {}
        for job in list(env.pending_jobs):
            # Find any GPU under the N_cap
            best_gpu = None
            for gpu in env.cluster.gpus.values():
                if gpu.num_adapters() < N_cap:
                    best_gpu = gpu.gpu_id
                    break
            
            if best_gpu is not None:
                actions[job.job_id] = (best_gpu, 16)
                    
        env.step(actions)
        
    total_time = env.wall_time
    total_tokens = sum(j.tokens_to_process for j in env.finished_jobs)
    global_throughput = total_tokens / total_time
    return global_throughput

if __name__ == "__main__":
    env = CLoraEnv(num_nodes=2, gpus_per_node=4) # 8 GPUs
    workload = generate_infinite_workload(200)
    
    n_caps = list(range(1, 13)) # N=1 to N=12
    throughputs = []
    
    for cap in n_caps:
        wl = copy.deepcopy(workload)
        tp = run_scheduler_with_cap(env, wl, cap)
        throughputs.append(tp)
        print(f"Cap N={cap}, Global Throughput={tp:.2f} tokens/s")
        
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(n_caps, throughputs, marker='o', linestyle='-', color='g')
    plt.axvline(x=6, color='r', linestyle='--', label='RL Agent Optimal Cap (N=6)')
    plt.xlabel('Concurrency Cap (N adapters per GPU)')
    plt.ylabel('Global System Throughput (Tokens/sec)')
    plt.title('Concurrency Sweet-Spot Test (N-Packing)')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/devarsh/Work/ResearchProject/c_lora_sim/experiment_2_results.png')
    print("Saved plot to experiment_2_results.png")
