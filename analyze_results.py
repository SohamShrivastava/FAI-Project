import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_performance_graphs():
    # map agent names to their result files
    log_files = {
        'Random Policy': 'results_RANDOM.csv',
        'SARSA (Table-Based)': 'results_SARSA.csv',
        'Deep Q-Network': 'results_DQN.csv'
    }

    plt.figure(figsize=(12, 7))
    
    # go through each results file and plot if available
    for label, filepath in log_files.items():
        if not os.path.exists(filepath):
            print(f"skipping {label} → file '{filepath}' not found")
            continue
            
        try:
            data = pd.read_csv(filepath)
            if data.empty or 'score' not in data.columns:
                print(f"warning: invalid data for {label}")
                continue

            window_size = 50
            data['trend'] = data['score'].rolling(window=window_size).mean()
            
            # plot smoothed curve
            plt.plot(data['episode'], data['trend'], label=f"{label} (avg)", linewidth=2)

        except Exception as err:
            print(f"error reading {filepath}: {err}")

    plt.title('rl agents performance comparison', fontsize=16)
    plt.xlabel('episode', fontsize=12)
    plt.ylabel('average score (50-step window)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # save final image
    output_img = 'agent_performance_comparison.png'
    plt.savefig(output_img, dpi=300)
    print(f"saved plot → {output_img}")

if __name__ == '__main__':
    generate_performance_graphs()
