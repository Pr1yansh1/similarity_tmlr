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


print("\n Making plots for full dataset ")
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