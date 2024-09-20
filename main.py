import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CentralLimitTheorem:
    def __init__(self, distribution):
        self.distribution = distribution
        self.dist_min = min(distribution)
        self.dist_max = max(distribution)

    def _sample(self, N):
        sample = random.choices(self.distribution, k=N)
        return np.mean(sample)

    def run_sample_demo(self, N, num_samples=1000):
        return [self._sample(N) for _ in range(num_samples)]

def generate_distribution(distribution_type, size):
    if distribution_type == 'uniform':
        return list(np.random.uniform(0, 100, size))
    elif distribution_type == 'normal':
        return list(np.random.normal(50, 15, size))
    elif distribution_type == 'exponential':
        return list(np.random.exponential(50, size))


def plot_distribution(distribution, title=None, bin_min=None, bin_max=None, num_bins=None, ax=None):
    sns.histplot(distribution, bins=num_bins, kde=True, color="skyblue", ax=ax)
    
    if title:
        ax.set_title(title)
    ax.set_xlim(bin_min, bin_max)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Frequency")

def main():
    fig, axes = plt.subplots(3, 4, figsize=(10, 10))
    fig.suptitle("Central Limit Theorem with Various Distributions", fontsize=20)

    distribution_types = ['uniform', 'normal', 'exponential']
    n_vals = [2, 10, 30, 100]

    for i, dist_type in enumerate(distribution_types):
        sample_distribution = generate_distribution(dist_type, 10000)
        clt_demo = CentralLimitTheorem(sample_distribution)
        
        for j, N in enumerate(n_vals):
            means = clt_demo.run_sample_demo(N=N)
            plot_distribution(means, f"{dist_type.capitalize()} - N = {N}", 
                              bin_min=min(sample_distribution), bin_max=max(sample_distribution), 
                              num_bins=40, ax=axes[i, j])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    main()
