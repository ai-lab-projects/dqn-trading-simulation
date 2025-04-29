import matplotlib.pyplot as plt
import pickle

metrics = {"total profit": 0, "win rate": 1, "average return": 2, "holding rate": 4, "average holding days": 5, "p_value": 6, "total_reward_over_mean": 7}

file2 = "results_July27_1.pkl"
file3 = "results_July27_2.pkl"

with open(file2, 'rb') as f:
    results2 = pickle.load(f)

with open(file3, 'rb') as f:
    results3 = pickle.load(f)

results = results2+results3

for i, result in enumerate(results):
    print(result["files"])
    print(result["params"])
    train_results = result["train_results"]
    val_results = result["val_results"]
    
    fig, axs = plt.subplots(1, 7, figsize=(20, 4))  
    fig.suptitle(f'Training Curves for Learning {i+1}')  

    for ax, (metric, idx) in zip(axs, metrics.items()):
        train_scores = [score[idx] for score in train_results]
        val_scores = [score[idx] for score in val_results]
        
        episodes = list(range(-1, len(train_scores) - 1))
        
        ax.scatter(episodes, train_scores, label='Train', s=10)  
        ax.scatter(episodes, val_scores, label='Validation', s=10)  
        ax.set_xlabel('Episode')  
        ax.set_ylabel(metric)  

        # Set grid and ticks
        ax.set_xticks(range(0, max(episodes) + 1, 5))  
        ax.set_xticks(range(min(episodes), max(episodes) + 1), minor=True)  

        ax.grid(which='both')  

        ax.legend()  
        ax.set_title(metric)  
        
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()  