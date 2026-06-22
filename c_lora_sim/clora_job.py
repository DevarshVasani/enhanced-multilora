from dataclasses import dataclass
from typing import Optional

@dataclass
class CLoraJob:
    job_id: int
    base_model_id: str
    adapter_rank: int
    tokens_to_process: int
    job_type: str  # "sampler" or "trainer"
    arrival_time: float
    
    # State tracking
    start_time: float = -1.0
    completion_time: float = -1.0
    tokens_processed: float = 0.0
    
    # For training jobs
    sub_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None

    def is_sampler(self) -> bool:
        return self.job_type == "sampler"
        
    def is_trainer(self) -> bool:
        return self.job_type == "trainer"
        
    def remaining_tokens(self) -> float:
        return max(0.0, self.tokens_to_process - self.tokens_processed)

    def mark_started(self, current_time: float) -> None:
        if self.start_time < 0:
            self.start_time = current_time

    def mark_completed(self, current_time: float) -> None:
        self.completion_time = current_time
        
    def advance_progress(self, tokens_processed: float) -> None:
        self.tokens_processed += tokens_processed
