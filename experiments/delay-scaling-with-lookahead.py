import numpy as np
import matplotlib.pyplot as plt
from importmonkey import add_path
add_path("../algorithms")
import greedy, rankgreedy, greedyrt
from online import rank_greedy_assign, greedy_rt_assign, online_past_ones_with_lookahead

# real scores
scores = np.loadtxt('../similarity_result.txt')[:100, :50]
# exponentiated scores
#scores = np.exp(scores) / np.exp(np.max(scores))
# random scores
#scores = np.random.rand(*scores.shape)
num_papers, num_reviewers = scores.shape
# low rank random scores
#scores = np.random.rand(num_papers, 1) @ np.random.rand(1, num_reviewers)
#scores /= np.max(scores)
r0 = 1

# d_max s.t. R > r0 * d
d_values = range(1, num_reviewers * 4 // 5 // r0 , num_papers//10)
f_values = [10**(-i) for i in range(3, 6, 1)]
obj_scores = []

for d in d_values:
    print(f"RUNNING COMPUTATIONS FOR d={d}")
    greedy_assignment = greedy.eval(scores, review_time=d, min_reviewer_per_paper=r0)
    rank_greedy_assignment = [rankgreedy.eval(scores, review_time=d, min_reviewer_per_paper=r0, factor=f) for f in f_values]
    greedy_rt_assignment = greedyrt.eval(scores, review_time=d, min_reviewer_per_paper=r0)
    lookahead_assignment = np.sum(online_past_ones_with_lookahead(scores, scores, review_time=d, min_reviewer_per_paper=r0, lookahead=10) * scores)
    obj_scores.append(#list(map(lambda assign: obj_score(scores, assign),
        [greedy_assignment, greedy_rt_assignment, lookahead_assignment] + rank_greedy_assignment)

obj_scores = np.array(obj_scores) / num_papers / r0
greedy, greedy_rt, lookahead, rg1, rg2, rg3 = list(zip(*obj_scores))
np.savetxt('delay_scaling_fast_results_iid.txt', obj_scores)
plt.plot(d_values, greedy, label='Greedy', alpha=0.5, marker='o', zorder=10)
plt.plot(d_values, greedy_rt, label='Greedy RT', alpha=0.5, marker='o')
plt.plot(d_values, lookahead, label='Lookahead', alpha=0.5, marker='o')
plt.plot(d_values, rg1, label='Rank Greedy 1e-03', alpha=0.5, marker='o')#+str(f_values[0]))
plt.plot(d_values, rg2, label='Rank Greedy 1e-04', alpha=0.5, marker='o')#+str(f_values[1]))
plt.plot(d_values, rg3, label='Rank Greedy '+str(f_values[2]), alpha=0.5, marker='o')
plt.xlabel('Review time')
plt.ylabel('Mean similarity score per assignment')
plt.legend(title='Policy')
#plt.ylim(ymin=0)
#plt.title('Delay scaling')
plt.gca().set_facecolor('lightgray')
plt.grid(True, color='white', linestyle='-', linewidth=0.5)
plt.show()
