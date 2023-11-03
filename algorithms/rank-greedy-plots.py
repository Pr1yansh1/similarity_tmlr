import numpy as np
import matplotlib.pyplot as plt
from greedy import * 
from online import * 

def obj_score(scores, assign):    
    return np.sum(scores * assign)

print("\n Making plots ")
sim_scores = np.loadtxt('similarity_result.txt')
print(sim_scores.shape)

d, lambd = 2, 2
factors = [10**(-i) for i in range(1, 9)]

def run_trial_for_num_reviewers(num_reviewers):
    obj_scores = []
    
    for trial in range(1, 27):
        scores = sim_scores[trial*15 : (trial*15) + 15, :].T
        rank_greedy_results = [rank_greedy_assign(scores, review_time=d, min_reviewer_per_paper=lambd, factor=f) for f in factors]
        obj_scores_for_trial = [obj_score(scores, res) for res in rank_greedy_results]
        obj_scores.append(obj_scores_for_trial)
    
    return obj_scores

markers = ['o', 's', '^', 'D', '*', 'p', 'v', '>', '<', 'H', '+', 'x', '|', '_']

num = 15 
obj_scores = run_trial_for_num_reviewers(num)
   
fig, ax = plt.subplots()
rank_greedy = list(zip(*obj_scores))

for i, factor in enumerate(factors):
    ax.plot(rank_greedy[i], label=f'Rank Greedy {factor:.0e}', marker=markers[i+3])

ax.legend(title="Policy")
ax.set_xlabel("Trials with different random reviewer pools", fontsize=12)
ax.set_ylabel("Objective scores")
# Removed the line setting the y-axis minimum to 0
ax.set_title(f'Policy comparison for {num} reviewers', fontsize=12)

plt.show()
