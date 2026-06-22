import random
from typing import List, Dict, Tuple, Optional
from c_lora_sim.clora_job import CLoraJob
from c_lora_sim.clora_cluster import GPUCluster, GPU

# A simple timeline event queue
class Timeline:
    def __init__(self):
        self.events = []

    def push(self, time: float, event: object):
        self.events.append((time, event))
        self.events.sort(key=lambda x: x[0], reverse=True)

    def pop(self) -> Tuple[float, object]:
        return self.events.pop()

    def __len__(self) -> int:
        return len(self.events)

def get_concurrency_penalty(N: int) -> float:
    """
    N=1 -> 1.0x
    N=4 -> 1.5x
    N=8 -> 2.5x
    Linear interpolation between points.
    """
    if N <= 1:
        return 1.0
    elif N <= 4:
        # N=1 -> 1.0, N=4 -> 1.5 => slope = 0.5/3 = 0.1667
        return 1.0 + (N - 1) * (0.5 / 3.0)
    elif N <= 8:
        # N=4 -> 1.5, N=8 -> 2.5 => slope = 1.0/4 = 0.25
        return 1.5 + (N - 4) * (1.0 / 4.0)
    else:
        # Penalize heavily beyond N=8
        return 2.5 + (N - 8) * 0.5

class CLoraEnv:
    def __init__(self, num_nodes: int = 4, gpus_per_node: int = 4):
        self.cluster = GPUCluster(num_nodes, gpus_per_node)
        self.timeline = Timeline()
        self.wall_time = 0.0
        self.jobs: List[CLoraJob] = []
        self.pending_jobs: List[CLoraJob] = []
        self.running_jobs: List[CLoraJob] = []
        self.finished_jobs: List[CLoraJob] = []
        
        self.base_throughput = 100.0 # tokens per second at N=1
        
    def reset(self, jobs: List[CLoraJob]):
        self.cluster.reset()
        self.timeline = Timeline()
        self.wall_time = 0.0
        self.jobs = jobs
        self.pending_jobs = []
        self.running_jobs = []
        self.finished_jobs = []
        
        for job in self.jobs:
            self.timeline.push(job.arrival_time, ("arrival", job.job_id))
            
    def _update_gpu_completion_events(self, gpu: GPU):
        """Recalculate completion events for all jobs on a given GPU."""
        # First, remove old completion events for these jobs
        active_ids = [j.job_id for j in gpu.active_adapters]
        self.timeline.events = [e for e in self.timeline.events if not (isinstance(e[1], tuple) and e[1][0] == "complete" and e[1][1] in active_ids)]
        
        N = gpu.num_adapters()
        if N == 0:
            return
            
        penalty = get_concurrency_penalty(N)
        throughput_per_job = (self.base_throughput / penalty) / N
        
        for job in gpu.active_adapters:
            # Time to finish remaining tokens
            time_needed = job.remaining_tokens() / throughput_per_job
            completion_time = self.wall_time + time_needed
            self.timeline.push(completion_time, ("complete", job.job_id, gpu.gpu_id))

    def _advance_all_progress(self, current_time: float):
        delta = current_time - self.wall_time
        if delta <= 0:
            return
            
        for gpu in self.cluster.gpus.values():
            N = gpu.num_adapters()
            if N > 0:
                penalty = get_concurrency_penalty(N)
                throughput_per_job = (self.base_throughput / penalty) / N
                tokens_processed = throughput_per_job * delta
                for job in gpu.active_adapters:
                    job.advance_progress(tokens_processed)

    def step(self, action: Optional[Dict[int, Tuple[int, int]]] = None):
        """
        Action format: {job_id: (gpu_id, sub_batch_size)}
        """
        if action:
            for job_id, (gpu_id, sbs) in action.items():
                job = next((j for j in self.pending_jobs if j.job_id == job_id), None)
                if job:
                    self.pending_jobs.remove(job)
                    gpu = self.cluster.get_gpu(gpu_id)
                    delay = gpu.add_adapter(job)
                    
                    job.start_time = self.wall_time + delay
                    job.sub_batch_size = sbs
                    self.running_jobs.append(job)
                    
                    # Schedule job actual start after cold start delay
                    self.timeline.push(job.start_time, ("start_exec", job.job_id, gpu.gpu_id))
        
        # Fast forward to next event
        if len(self.timeline) > 0:
            next_time, event = self.timeline.pop()
            self._advance_all_progress(next_time)
            self.wall_time = next_time
            
            if isinstance(event, tuple) and event[0] == "arrival":
                job_id = event[1]
                job = next((j for j in self.jobs if j.job_id == job_id), None)
                if job:
                    self.pending_jobs.append(job)
                    
            elif isinstance(event, tuple) and event[0] == "start_exec":
                gpu_id = event[2]
                gpu = self.cluster.get_gpu(gpu_id)
                self._update_gpu_completion_events(gpu)
                
            elif isinstance(event, tuple) and event[0] == "complete":
                job_id = event[1]
                gpu_id = event[2]
                job = next((j for j in self.running_jobs if j.job_id == job_id), None)
                if job:
                    job.mark_completed(self.wall_time)
                    self.running_jobs.remove(job)
                    self.finished_jobs.append(job)
                    gpu = self.cluster.get_gpu(gpu_id)
                    gpu.remove_adapter(job)
                    self._update_gpu_completion_events(gpu)
                    
        done = len(self.finished_jobs) == len(self.jobs)
        return self.get_state(), done
        
    def get_state(self):
        return {
            "time": self.wall_time,
            "pending": len(self.pending_jobs),
            "running": len(self.running_jobs),
            "finished": len(self.finished_jobs),
            "cluster": self.cluster
        }
