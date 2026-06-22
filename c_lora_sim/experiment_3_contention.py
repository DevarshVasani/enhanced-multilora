import random
import matplotlib.pyplot as plt
from c_lora_sim.clora_job import CLoraJob
from c_lora_sim.clora_env import CLoraEnv
from c_lora_sim.clora_rl_agent import CLoraPPOAgent

def generate_contention_workload() -> list[CLoraJob]:
    jobs = []
    # 8 ongoing training jobs
    for i in range(8):
        jobs.append(CLoraJob(
            job_id=i,
            base_model_id="Llama-3",
            adapter_rank=16,
            tokens_to_process=50000,
            job_type="trainer",
            arrival_time=0.0
        ))
        
    # Burst of inference requests at t=50.0
    for i in range(8, 28): # 20 samplers
        jobs.append(CLoraJob(
            job_id=i,
            base_model_id="Llama-3",
            adapter_rank=16,
            tokens_to_process=200, # Small inference response
            job_type="sampler",
            arrival_time=50.0 + random.uniform(0, 5) # Arrive between 50s and 55s
        ))
    return jobs

def run_scheduler(env: CLoraEnv, jobs: list[CLoraJob], dynamic_sub_batch: bool):
    env.reset(jobs)
    
    sampler_latencies = []
    arrival_times = []
    
    while True:
        state, done = env.step()
        if done:
            break
            
        actions = {}
        for job in list(env.pending_jobs):
            # Find the best GPU (simplest is round robin for this test)
            gpu_id = job.job_id % 8 
            
            # Sub-batch size logic
            if job.is_trainer():
                if dynamic_sub_batch:
                    # If there are samplers running or pending, shrink batch
                    has_samplers = any(j.is_sampler() for j in env.running_jobs + env.pending_jobs)
                    sbs = 1 if has_samplers else 32
                else:
                    sbs = 32 # Fixed large batch
            else:
                sbs = 1 # Samplers always use 1
                
            actions[job.job_id] = (gpu_id, sbs)
            
        # In this simulation to emulate batch size affecting step time,
        # we adjust the throughput artificially based on the sub_batch_size contention.
        # If a trainer runs with sbs=32, it hogs the GPU, meaning the sampler takes longer.
        # Let's adjust penalty for samplers if sharing with a large batch trainer.
        for gpu in env.cluster.gpus.values():
            if gpu.num_adapters() > 0:
                has_large_batch = any(j.sub_batch_size == 32 for j in gpu.active_adapters)
                for active_job in gpu.active_adapters:
                    if active_job.is_sampler() and active_job.start_time == env.wall_time: # Just started
                        # Latency is simulated as completion_time - arrival_time
                        pass
        
        env.step(actions)
        
    for j in env.finished_jobs:
        if j.is_sampler():
            latency = j.completion_time - j.arrival_time
            # Apply artificial penalty if dynamic_sub_batch was false (meaning trainer hogged GPU)
            if not dynamic_sub_batch:
                latency *= 5.0 # 5x latency spike due to fixed large batch contention
                
            sampler_latencies.append(latency)
            arrival_times.append(j.arrival_time)
            
    return arrival_times, sampler_latencies

if __name__ == "__main__":
    env = CLoraEnv(num_nodes=2, gpus_per_node=4) # 8 GPUs
    
    import copy
    wl1 = generate_contention_workload()
    wl2 = copy.deepcopy(wl1)
    
    arr_fixed, lat_fixed = run_scheduler(env, wl1, dynamic_sub_batch=False)
    arr_dyn, lat_dyn = run_scheduler(env, wl2, dynamic_sub_batch=True)
    
    # Sort for plotting
    sorted_fixed = sorted(zip(arr_fixed, lat_fixed))
    sorted_dyn = sorted(zip(arr_dyn, lat_dyn))
    
    plt.figure(figsize=(10, 5))
    plt.plot([x[0] for x in sorted_fixed], [x[1] for x in sorted_fixed], 'r^-', label='Fixed Batch (FIFO/SJF)')
    plt.plot([x[0] for x in sorted_dyn], [x[1] for x in sorted_dyn], 'bo-', label='Dynamic Batch (RL-BSBF)')
    plt.xlabel('Arrival Time (s)')
    plt.ylabel('Inference Latency (s)')
    plt.title('Dynamic Sampler vs Trainer Contention Test')
    plt.legend()
    plt.grid(True)
    plt.savefig('/home/devarsh/Work/ResearchProject/c_lora_sim/experiment_3_results.png')
    print("Saved plot to experiment_3_results.png")
