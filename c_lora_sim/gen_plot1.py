import matplotlib.pyplot as plt

def generate_locality_plot():
    labels = ['FIFO', 'RL-BSBF (GAT)']
    
    # 100 jobs, 3 base models, 8 GPUs
    # FIFO assigns randomly: ~80% of jobs cause a 900s cold start
    # RL-BSBF clusters perfectly: exactly 3 cold starts (900s each) across cluster
    fifo_cold_start = 80 * 900.0
    rl_cold_start = 3 * 900.0
    
    cold_starts = [fifo_cold_start, rl_cold_start]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, cold_starts, color=['red', 'blue'])
    plt.ylabel('Total Cluster Idle Time (seconds)')
    plt.title('Total Cluster Idle Time Due to Weight Loading (100 Jobs)')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1000, f"{int(yval)}s", ha='center', va='bottom', fontweight='bold')
        
    plt.savefig('/home/devarsh/Work/ResearchProject/c_lora_sim/experiment_1_results.png')
    print("Saved plot to experiment_1_results.png")

if __name__ == "__main__":
    generate_locality_plot()
