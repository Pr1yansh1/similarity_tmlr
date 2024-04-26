import numpy as np
import random
import matplotlib.pyplot as plt
#import oracle, greedy, randomassign
import greedy
from online import rank_greedy_assign, greedy_rt_assign


sim_scores = np.loadtxt('../similarity_result.txt')
d, lambd = 6, 3


def obj_score(scores, assign):
    return np.sum(scores * assign)

scores = sim_scores
greedy_assignment = greedy.assign(scores, review_time=d, min_reviewer_per_paper = lambd)
rank_greedy_assignment = [rank_greedy_assign(scores, review_time=d, min_reviewer_per_paper=lambd, factor=f) for f in [10**(-i) for i in range(8, 11, 1)]]
greedy_rt_assignment = greedy_rt_assign(scores, review_time=d, min_reviewer_per_paper=lambd)

scores = list(map(lambda assign: obj_score(scores, assign),
                  [greedy_assignment, greedy_rt_assignment] + rank_greedy_assignment))
print(scores)

