import numpy as np
import matplotlib.pyplot as plt
from greedy import assign 
from online import rank_greedy_assign 


factors = [0.00001, 0.000001, 0.0000001, 0.00000001]  

def cumulative_obj_scores(scores, assignments):
    # Calculates the cumulative objective scores for given scores and assignments
    cum_scores = np.cumsum(np.sum(scores * assignments, axis=1))
    return cum_scores

# Load the similarity scores
sim_scores = np.loadtxt('similarity_result.txt')

# Assign using both algorithms
greedy_assignments = assign(sim_scores)
rank_greedy_assignments = rank_greedy_assign(sim_scores, factor = factors[3])

# Calculate cumulative scores
cumulative_greedy_scores = cumulative_obj_scores(sim_scores, greedy_assignments)
cumulative_rank_greedy_scores = cumulative_obj_scores(sim_scores, rank_greedy_assignments)

# Plotting
plt.plot(cumulative_greedy_scores, label='Greedy')
plt.plot(cumulative_rank_greedy_scores, label='Rank-Greedy')
plt.xlabel('Number of Papers Arrived')
plt.ylabel('Cumulative Similarity Score')
plt.legend()
plt.title('Comparison of Cumulative Similarity Score for Greedy vs. Rank-Greedy')
plt.show()