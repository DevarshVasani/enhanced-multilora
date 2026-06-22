from typing import List, Optional, Dict
from c_lora_sim.clora_job import CLoraJob

class GPU:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.base_model_id: Optional[str] = None
        self.active_adapters: List[CLoraJob] = []
        
    def is_free(self) -> bool:
        return len(self.active_adapters) == 0
        
    def num_adapters(self) -> int:
        return len(self.active_adapters)
        
    def add_adapter(self, job: CLoraJob) -> float:
        """
        Adds a job to this GPU.
        Returns the cold start delay required.
        """
        delay = 0.0
        if self.base_model_id != job.base_model_id:
            # Massive simulated penalty (e.g., 15 minutes) for base model swap
            delay = 900.0 if self.base_model_id is not None else 10.0 # 10s if initially empty
            self.base_model_id = job.base_model_id
            self.active_adapters.clear() # Switching base model clears old adapters
        else:
            # Near zero penalty for swapping/adding adapters
            delay = 0.1 
            
        self.active_adapters.append(job)
        return delay
        
    def remove_adapter(self, job: CLoraJob) -> None:
        if job in self.active_adapters:
            self.active_adapters.remove(job)

class GPUCluster:
    def __init__(self, num_nodes: int, gpus_per_node: int):
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.gpus: Dict[int, GPU] = {i: GPU(i) for i in range(num_nodes * gpus_per_node)}
        
    def get_gpu(self, gpu_id: int) -> GPU:
        return self.gpus[gpu_id]
        
    def reset(self) -> None:
        for gpu in self.gpus.values():
            gpu.base_model_id = None
            gpu.active_adapters.clear()
