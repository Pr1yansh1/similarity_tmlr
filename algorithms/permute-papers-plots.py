import greedy 
from online import * 

import numpy as np
import matplotlib.pyplot as plt

def obj_score(scores, assign):    
    return np.sum(scores * assign)

sim_scores = np.loadtxt('similarity_result.txt')
d, lambd = 2, 2
factors = [0.0001, 0.00001, 0.000001]  # Multiple factors in the desired notation
num_permutations = 50  # Increased permutations
w_max = np.max(sim_scores)  # Maximum edge weight

# Lists to store the total similarities for each permutation
greedy_scores = []
rank_scores = []
greedy_rt_scores = []
rank_greedy_scores = {factor: [] for factor in factors}  # Dictionary to store scores for each factor

# Loop for each permutation
for _ in range(num_permutations):
    permuted_scores = np.random.permutation(sim_scores)  # Permute the order of papers

    greedy_assign = greedy.assign(permuted_scores, review_time=d, min_reviewer_per_paper=lambd)
    rank_assign_result = rank_assign(permuted_scores, review_time=d, min_reviewer_per_paper=lambd)
    greedy_rt_assign_result = greedy_rt_assign(permuted_scores, review_time=d, min_reviewer_per_paper=lambd, w_max=w_max)
    
    greedy_scores.append(obj_score(permuted_scores, greedy_assign))
    rank_scores.append(obj_score(permuted_scores, rank_assign_result))
    greedy_rt_scores.append(obj_score(permuted_scores, greedy_rt_assign_result))
    
    # Loop over each factor
    for factor in factors:
        rank_greedy_result = rank_greedy_assign(permuted_scores, review_time=d, min_reviewer_per_paper=lambd, factor=factor)
        rank_greedy_scores[factor].append(obj_score(permuted_scores, rank_greedy_result))

# Plotting
plt.plot(greedy_scores, label='Greedy')
plt.plot(rank_scores, label='Rank Assign')
plt.plot(greedy_rt_scores, label=f'Greedy RT (w_max={w_max:.2f})')

# Plotting rank_greedy scores for each factor
for factor in factors:
    plt.plot(rank_greedy_scores[factor], label=f'Rank Greedy {factor}')

plt.xlabel('Permutations')
plt.ylabel('Total Similarity Sum')
plt.legend()
plt.title('Comparison across different paper arrival permutations')
plt.show()
