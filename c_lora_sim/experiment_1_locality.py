import random
import matplotlib.pyplot as plt
from c_lora_sim.clora_job import CLoraJob
from c_lora_sim.clora_env import CLoraEnv
from c_lora_sim.clora_rl_agent import CLoraPPOAgent

def generate_workload(num_jobs: int = 100) -> list[CLoraJob]:
    base_models = ["Llama-3", "Qwen-2.5", "Mistral"]
    jobs = []
    for i in range(num_jobs):
        model_id = random.choice(base_models)
        # All jobs arrive at time 0 (flood)
        job = CLoraJob(
            job_id=i,
            base_model_id=model_id,
            adapter_rank=16,
            tokens_to_process=10000,
            job_type="trainer",
            arrival_time=0.0
        )
        jobs.append(job)
    return jobs

def run_fifo_scheduler(env: CLoraEnv, jobs: list[CLoraJob]):
    env.reset(jobs)
    total_cold_start = 0.0
    
    while True:
        actions = {}
        if env.pending_jobs:
            job = env.pending_jobs[0] # Schedule one job at a time
            best_gpu = None
            for gpu in env.cluster.gpus.values():
                if gpu.num_adapters() < 6:
                    best_gpu = gpu.gpu_id
                    break
            
            if best_gpu is not None:
                actions[job.job_id] = (best_gpu, 16)
                gpu = env.cluster.get_gpu(best_gpu)
                if gpu.base_model_id != job.base_model_id:
                    total_cold_start += 900.0 if gpu.base_model_id is not None else 10.0
                    
        state, done = env.step(actions)
        if done:
            break
            
    jct = sum(j.completion_time - j.arrival_time for j in env.finished_jobs) / len(env.finished_jobs)
    return total_cold_start, jct

def run_rl_scheduler(env: CLoraEnv, jobs: list[CLoraJob]):
    env.reset(jobs)
    agent = CLoraPPOAgent(num_gpus=len(env.cluster.gpus))
    total_cold_start = 0.0
    
    while True:
        actions = {}
        if env.pending_jobs:
            # We only pass the first pending job to simulate sequential scheduling
            agent_actions = agent.select_action([env.pending_jobs[0]], env.cluster)
            actions.update(agent_actions)
        
        # Calculate cold start from actions
        for job_id, (gpu_id, _) in actions.items():
            job = next(j for j in env.pending_jobs if j.job_id == job_id)
            gpu = env.cluster.get_gpu(gpu_id)
            if gpu.base_model_id != job.base_model_id:
                total_cold_start += 900.0 if gpu.base_model_id is not None else 10.0
                
        state, done = env.step(actions)
        if done:
            break
        
    jct = sum(j.completion_time - j.arrival_time for j in env.finished_jobs) / len(env.finished_jobs)
    return total_cold_start, jct

if __name__ == "__main__":
    env = CLoraEnv(num_nodes=2, gpus_per_node=4) # 8 GPUs total
    workload = generate_workload(100)
    
    # Deep copy workload
    import copy
    workload_fifo = copy.deepcopy(workload)
    workload_rl = copy.deepcopy(workload)
    
    fifo_cold_start, fifo_jct = run_fifo_scheduler(env, workload_fifo)
    rl_cold_start, rl_jct = run_rl_scheduler(env, workload_rl)
    
    print(f"FIFO Scheduler: Cold Start Time = {fifo_cold_start}s, Avg JCT = {fifo_jct:.2f}s")
    print(f"RL-BSBF Scheduler: Cold Start Time = {rl_cold_start}s, Avg JCT = {rl_jct:.2f}s")
    
    # Plot results
    labels = ['FIFO', 'RL-BSBF (GAT)']
    cold_starts = [fifo_cold_start, rl_cold_start]
    
    plt.figure(figsize=(8, 6))
    plt.bar(labels, cold_starts, color=['red', 'blue'])
    plt.ylabel('Total Cluster Idle Time (seconds)')
    plt.title('Total Cluster Idle Time Due to Weight Loading')
    plt.savefig('/home/devarsh/Work/ResearchProject/c_lora_sim/experiment_1_results.png')
    print("Saved plot to experiment_1_results.png")
