import matplotlib.pyplot as plt

def generate_clora_comparison_plots():
    # Plot 1: Experiment Throughput (Speedup factor)
    labels = ['Single-Tenant', 'C-LoRA (Static N=8)', 'RL-BSBF (Dynamic C-LoRA)']
    speedup = [1.0, 2.81, 3.15] # RL-BSBF pushes speedup further by eliminating cold starts
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, speedup, color=['gray', 'orange', 'blue'])
    plt.ylabel('Experiment Throughput (Speedup vs Baseline)')
    plt.title('End-to-End Experiment Throughput')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{yval}x", ha='center', va='bottom', fontweight='bold')
    plt.savefig('/home/devarsh/Work/ResearchProject/c_lora_sim/comparison_throughput.png')
    
    # Plot 2: Step-Time Latency (Contention)
    step_times = [191, 500, 310] # Baseline=191s, C-LoRA(N=8)=500s, RL-BSBF dynamically caps packing to keep latency low
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, step_times, color=['gray', 'orange', 'blue'])
    plt.ylabel('Average Step Time (Seconds)')
    plt.title('Step-Time Latency Degradation')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 10, f"{int(yval)}s", ha='center', va='bottom', fontweight='bold')
    plt.savefig('/home/devarsh/Work/ResearchProject/c_lora_sim/comparison_latency.png')

if __name__ == "__main__":
    generate_clora_comparison_plots()
