import numpy as np
import matplotlib.pyplot as plt
from greedy import * 
from online import * 


#plots for online algorithms with different sets of sub sampled reviewers. 

def obj_score(scores, assign):    
    return np.sum(scores * assign)

print("\n Making plots ")
sim_scores = np.loadtxt('similarity_result.txt')
num_papers = sim_scores.shape[1]
total_reviewers = sim_scores.shape[0]

d, lambd = 2, 2
factors = [10**(-i) for i in range(8, 11, 3)]

def run_trial_for_num_reviewers(num_reviewers):
    max_trials = total_reviewers // num_reviewers
    obj_scores = []
    
    for trial in range(1, max_trials):
        scores = sim_scores[trial* num_reviewers : (trial*num_reviewers) + num_reviewers, :].T
        greedy_assign = assign(scores, review_time=d, min_reviewer_per_paper=lambd)
        rank_only_results = rank_assign(scores, review_time=d, min_reviewer_per_paper=lambd)
        rank_greedy_results = [rank_greedy_assign(scores, review_time=d, min_reviewer_per_paper=lambd, factor=f) for f in factors]
        greedy_rt_results = greedy_rt_assign(scores, review_time=d, min_reviewer_per_paper=lambd)
        variable_threshold_results = variable_threshold_assign(scores, review_time=d, min_reviewer_per_paper=lambd)
        obj_scores_for_trial = [obj_score(scores, res) for res in [greedy_assign, rank_only_results, greedy_rt_results, variable_threshold_results] + rank_greedy_results]
        obj_scores.append(obj_scores_for_trial)
    
    return obj_scores

markers = ['o', 's', '^', 'D', '*', 'p', 'v', '>', '<', 'H', '+', 'x', '|', '_']

fig, axes = plt.subplots(2, 2, figsize=(15, 15))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

reviewers = [15]

for idx, num in enumerate(reviewers):
    obj_scores = run_trial_for_num_reviewers(num)
    ax = axes[idx//2, idx%2]
    greedy_scores, rank_only, greedy_rt_results, variable_threshold_result,  *rank_greedy = list(zip(*obj_scores))
    
    ax.plot(greedy_scores, label='Greedy', marker=markers[0])
    ax.plot(rank_only, label='Rank Only', marker=markers[1])
    ax.plot(greedy_rt_results, label='Greedy RT', marker=markers[2])
    ax.plot(variable_threshold_result, label = "Variable Threshold", marker = markers[3])

    for i, factor in enumerate(factors):
        ax.plot(rank_greedy[i], label=f'Rank Greedy {factor:.0e}', marker=markers[i+3])

    ax.legend(title="Policy")
    ax.set_xlabel("Trials with different random reviewer pools", y=1.1, fontsize=12)
    ax.set_ylabel("Objective scores")
    ax.set_ylim(ymin=0)
    ax.set_title(f'Policy comparison for {num} reviewers', y=0.9, fontsize=12)

plt.tight_layout(pad=2.0)
plt.show()
