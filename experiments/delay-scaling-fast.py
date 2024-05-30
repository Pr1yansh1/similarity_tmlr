import numpy as np
import matplotlib.pyplot as plt
from importmonkey import add_path
add_path("../algorithms")
import greedy, rankgreedy, greedyrt
from online import rank_greedy_assign, greedy_rt_assign

scores = np.loadtxt('../similarity_result.txt')
#scores = np.exp(scores) / np.exp(np.max(scores))
num_papers, num_reviewers = scores.shape
#scores = np.random.rand(*scores.shape)
scores = np.random.rand(num_papers, 1) @ np.random.rand(1, num_reviewers)
scores /= np.max(scores)
lambd = 2

d_values = range(1, num_papers, num_papers//20)
f_values = [10**(-i) for i in range(3, 6, 1)]
obj_scores = []

for d in d_values:
    greedy_assignment = greedy.eval(scores, review_time=d, min_reviewer_per_paper = lambd)
    rank_greedy_assignment = [rankgreedy.eval(scores, review_time=d, min_reviewer_per_paper=lambd, factor=f) for f in f_values]
    greedy_rt_assignment = greedyrt.eval(scores, review_time=d, min_reviewer_per_paper=lambd)
    obj_scores.append(#list(map(lambda assign: obj_score(scores, assign),
                  [greedy_assignment, greedy_rt_assignment] + rank_greedy_assignment)

obj_scores = np.array(obj_scores) / num_papers / lambd
greedy, greedy_rt, rg1, rg2, rg3 = list(zip(*obj_scores))
np.savetxt('delay_scaling_fast_results_iid.txt', obj_scores)
plt.plot(d_values, greedy, label='Greedy')
plt.plot(d_values, greedy_rt, label='Greedy RT')
plt.plot(d_values, rg1, label='Rank Greedy 1e-03')#+str(f_values[0]))
plt.plot(d_values, rg2, label='Rank Greedy 1e-04')#+str(f_values[1]))
plt.plot(d_values, rg3, label='Rank Greedy '+str(f_values[2]))
plt.xlabel('Review time')
plt.ylabel('Mean similarity score per assignment')
plt.legend(title='Policy')
#plt.ylim(ymin=0)
#plt.title('Delay scaling')
plt.gca().set_facecolor('lightgray')
plt.grid(True, color='white', linestyle='-', linewidth=0.5)
plt.show()
