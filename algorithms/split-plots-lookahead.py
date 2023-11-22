import numpy as np
import matplotlib.pyplot as plt
from greedy import * 
from online import * 
import oracle

# Plots for online algorithms with different sets of sub-sampled reviewers.
def obj_score(scores, assign):    
    return np.sum(scores * assign)

print("\n Making plots ")
sim_scores = np.loadtxt('similarity_result.txt')
num_papers = sim_scores.shape[1]

d, lambd = 2, 2
factors = [10**(-i) for i in range(8, 11, 3)]
lookahead = 20 

split_index = int(num_papers * 0.8)

# Splitting along the number of papers dimension
actual_scores = sim_scores[:, :split_index]
sampling_scores = sim_scores[:, split_index:] 

x, y = sampling_scores.shape 

# for i in range(x): 
#     for j in range(y): 
#         sampling_scores[i, j] = np.random.uniform(0, 1)


def run_trial_for_num_reviewers():
    obj_scores = []
    
    for trial in range(1, 27):
        scores = actual_scores[trial*15 : (trial*15) + 15, :].T

        lp_assign = oracle.lp(scores, review_time = d, min_reviewer_per_paper =lambd)
    
        greedy_assign = assign(scores, review_time=d, min_reviewer_per_paper=lambd)
        rank_only_results = rank_assign(scores, review_time=d, min_reviewer_per_paper=lambd)
        rank_greedy_results = [rank_greedy_assign(scores, review_time=d, min_reviewer_per_paper=lambd, factor=f) for f in factors]
        greedy_rt_results = greedy_rt_assign(scores, review_time=d, min_reviewer_per_paper=lambd)
        past_ones_results = online_past_ones_with_lookahead(scores, sampling_scores, review_time=d, min_reviewer_per_paper=lambd, lookahead=lookahead)


        for time_step in range(len(scores)):
            print(f"\nTime Step {time_step + 1}:")
            print("Greedy Assignment:")
            print(greedy_assign[time_step])  # Replace with appropriate variable
            print("Lookahead Assignment:")
            print(past_ones_results[time_step]) 
            print("Oracle assignment") 
            print(lp_assign[time_step])

        obj_scores_for_trial = [obj_score(scores, res) for res in [lp_assign, greedy_assign, rank_only_results, greedy_rt_results, past_ones_results] + rank_greedy_results]
        obj_scores.append(obj_scores_for_trial)
    
    return obj_scores

markers = ['o', 's', '^', 'D', '*', 'p', 'v', '>', '<', 'H', '+', 'x', '|', '_']

fig, axes = plt.subplots(2, 2, figsize=(15, 15))
plt.subplots_adjust(wspace=0.3, hspace=0.5)

reviewers = [15]

for idx, num in enumerate(reviewers):
    obj_scores = run_trial_for_num_reviewers()
    ax = axes[idx//2, idx%2]

    # Unpack the objective scores for different algorithms
    lp_assign, greedy_scores, rank_only, greedy_rt_results, past_ones, *rank_greedy = list(zip(*obj_scores))
    
    # Plot the results for each algorithm
    ax.plot(lp_assign, label = "LP oracle", marker = markers[4])
    ax.plot(greedy_scores, label='Greedy', marker=markers[0])
    ax.plot(rank_only, label='Rank Only', marker=markers[1])
    ax.plot(greedy_rt_results, label='Greedy RT', marker=markers[2])
    ax.plot(past_ones, label='Past Ones with Lookahead', marker=markers[3])

    for i, factor in enumerate(factors):
        ax.plot(rank_greedy[i], label=f'Rank Greedy {factor:.0e}', marker=markers[i+4])  # Adjusted index for markers

    ax.legend(title="Policy")
    ax.set_xlabel("Trials with different random reviewer pools", y=1.1, fontsize=12)
    ax.set_ylabel("Objective scores")
    ax.set_ylim(ymin=0)
    ax.set_title(f'Policy comparison for {num} reviewers', y=0.9, fontsize=12)

plt.tight_layout(pad=2.0)
plt.show()
