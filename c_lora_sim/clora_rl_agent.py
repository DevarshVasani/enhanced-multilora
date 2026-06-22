import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
from c_lora_sim.clora_job import CLoraJob
from c_lora_sim.clora_cluster import GPUCluster

class CLoraGAT(nn.Module):
    """
    A mock Graph Attention Network feature extractor for C-LoRA.
    """
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(feature_dim, hidden_dim)
        
    def build_graph_edges(self, jobs: List[CLoraJob]) -> List[Tuple[int, int]]:
        """
        Connects jobs in the queue if they share the exact same Base Model ID.
        This forces the GAT to see 'clusters' of similar LLM workloads.
        """
        edges = []
        for i in range(len(jobs)):
            for j in range(i + 1, len(jobs)):
                if jobs[i].base_model_id == jobs[j].base_model_id:
                    edges.append((i, j))
                    edges.append((j, i))
        return edges
        
    def forward(self, job_features: torch.Tensor, edges: List[Tuple[int, int]]) -> torch.Tensor:
        # In a full PyTorch Geometric implementation, this would use GATConv.
        # For our simulation/POC, we return transformed embeddings.
        return torch.relu(self.fc(job_features))

class CLoraPPOAgent:
    """
    PPO Agent wrapper that produces the C-LoRA specific action space:
    (Placement GPU, Co-location target N, Sub-batch size)
    """
    def __init__(self, num_gpus: int, max_sub_batch: int = 32):
        self.num_gpus = num_gpus
        self.max_sub_batch = max_sub_batch
        self.gat = CLoraGAT(feature_dim=5, hidden_dim=64)
        
    def extract_features(self, jobs: List[CLoraJob], cluster: GPUCluster) -> Dict[str, Any]:
        """
        Extracts features for the RL policy.
        """
        edges = self.gat.build_graph_edges(jobs)
        # Mock feature tensor
        features = torch.zeros((len(jobs), 5))
        for i, job in enumerate(jobs):
            features[i, 0] = job.tokens_to_process
            features[i, 1] = 1.0 if job.is_trainer() else 0.0
            
        embeddings = self.gat(features, edges)
        return {
            "embeddings": embeddings,
            "edges": edges
        }
        
    def select_action(self, pending_jobs: List[CLoraJob], cluster: GPUCluster) -> Dict[int, Tuple[int, int]]:
        """
        Output:
        Placement: Which GPU node to assign it to.
        Co-location target (N): Which existing adapter pool to group it with (implicit via GPU choice).
        Sub-batch size / Gradient Accumulation: Micro-batch configuration.
        """
        actions = {}
        for job in pending_jobs:
            # Mock policy: Find a GPU with the same base model if possible,
            # and limit packing to the optimal N (e.g., N=4 or N=6)
            best_gpu = None
            for gpu in cluster.gpus.values():
                if gpu.base_model_id == job.base_model_id and gpu.num_adapters() < 6:
                    best_gpu = gpu.gpu_id
                    break
            
            if best_gpu is None:
                # Find an empty GPU
                for gpu in cluster.gpus.values():
                    if gpu.is_free():
                        best_gpu = gpu.gpu_id
                        break
                        
            if best_gpu is not None:
                # Dynamic sub-batch size logic:
                # If there are samplers running, reduce trainer batch size
                sub_batch = 16 if job.is_trainer() else 1
                actions[job.job_id] = (best_gpu, sub_batch)
                
        return actions
