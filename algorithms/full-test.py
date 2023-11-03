import numpy as np
import random
import matplotlib.pyplot as plt
import oracle, greedy, mdp, randomassign
import online 

def obj_score(scores, assign):	
    return np.sum(scores * assign)

print("\n Making assignments on the full similarity matrix ")
sim_scores = np.loadtxt('similarity_result.txt')

d, lambd = 2, 2
factors = [0.00001, 0.000001, 0.0000001, 0.00000001]  # Example values

# Assignments
lp_assign = oracle.lp(sim_scores, review_time=d, min_reviewer_per_paper=lambd)
ilp_assign = oracle.ilp(sim_scores, review_time=d, min_reviewer_per_paper=lambd)
greedy_assign = greedy.assign(sim_scores, review_time=d, min_reviewer_per_paper=lambd)
mdp_assign = mdp.assign(sim_scores, reviews_per_paper=lambd)
random_assign = randomassign.assign(sim_scores, review_time=d, min_reviewer_per_paper=lambd)
rank_only_results = online.rank_assign(sim_scores, review_time=d, min_reviewer_per_paper=lambd)
rank_greedy_results = [online.rank_greedy_assign(sim_scores, review_time=d, min_reviewer_per_paper=lambd, factor=f) for f in factors]

# Objective scores
obj_scores = [obj_score(sim_scores, assign) for assign in [lp_assign, ilp_assign, greedy_assign, mdp_assign, rank_only_results, *rank_greedy_results]]

# Print Results
methods = ['Oracle LP', 'Oracle ILP', 'Greedy', 'MDP', 'Rank Only'] + [f'Rank Greedy {factor}' for factor in factors]
for method, score in zip(methods, obj_scores):
    print(f"{method}: {score}")

# If you want to save the assignments, you can save it like this:
np.savetxt('full_assignments.txt', mdp_assign)
