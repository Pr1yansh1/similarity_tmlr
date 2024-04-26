import numpy as np
import matplotlib.pyplot as plt
from importmonkey import add_path
add_path("../algorithms")
import greedy
from online import rank_greedy_assign, greedy_rt_assign

scores = np.loadtxt('../similarity_result.txt')
P, R = scores.shape
lambd = 2
def obj_score(scores, assign):
    return np.sum(scores * assign)


d_values = range(1, P,  P//20)
obj_scores = []

for d in d_values:
    greedy_assignment = greedy.assign(scores, review_time=d, min_reviewer_per_paper = lambd)
    rank_greedy_assignment = [rank_greedy_assign(scores, review_time=d, min_reviewer_per_paper=lambd, factor=f) for f in [10**(-i) for i in range(5, 10, 2)]]
    greedy_rt_assignment = greedy_rt_assign(scores, review_time=d, min_reviewer_per_paper=lambd)
    obj_scores.append(list(map(lambda assign: obj_score(scores, assign),
                  [greedy_assignment, greedy_rt_assignment] + rank_greedy_assignment)))

greedy, greedy_rt, rg1, rg2, rg3 = list(zip(*obj_scores))
plt.plot(d_values, greedy, label='Greedy')
plt.plot(d_values, greedy_rt, label='Greedy RT')
plt.plot(d_values, rg1, label='Rank Greedy 1')
plt.plot(d_values, rg2, label='Rank Greedy 2')
plt.plot(d_values, rg3, label='Rank Greedy 3')
plt.xlabel('Review time(d)')
plt.ylabel('Objective Score (Aggregate similarity match)')
plt.legend(title='Policy')
#plt.ylim(ymin=0)
#plt.title('Delay scaling')
plt.show()
