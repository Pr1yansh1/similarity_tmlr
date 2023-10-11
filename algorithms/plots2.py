import numpy as np
import random
import matplotlib.pyplot as plt
import oracle, greedy, mdp, randomassign

# Assuming you have these in your environment or another module
from online import greedy_rt_assign, rank_assign

def obj_score(scores, assign):    
    return np.sum(scores * assign)

print("\n Making plots ")
sim_scores = np.loadtxt('similarity_result.txt')
d, lambd = 2, 2
obj_scores = []
assignments = []

for trial in range(20, 35):
    scores = sim_scores[:, trial * 15 : trial * 15 + 15]

    lp_assign = oracle.lp(scores, review_time=d, min_reviewer_per_paper=lambd)
    ilp_assign = oracle.ilp(scores, review_time=d, min_reviewer_per_paper=lambd)
    greedy_assign = greedy.assign(scores, review_time=d, min_reviewer_per_paper=lambd)
    greedy_rt = greedy_rt_assign(scores, review_time=d, min_reviewer_per_paper=lambd, w_max=0.15)
    rank_based = rank_assign(scores)
    mdp_assign = mdp.assign(scores, reviews_per_paper=lambd)
    random_assign = randomassign.assign(scores, review_time=d, min_reviewer_per_paper=lambd)

    randomly_drawn_scores = np.array(random.choices(scores, k=scores.shape[0]))
    assignments.append([scores, randomly_drawn_scores, lp_assign, ilp_assign, greedy_assign, greedy_rt, rank_based, mdp_assign])
    obj_scores.append(list(map(lambda assign: obj_score(randomly_drawn_scores, assign),
                               [lp_assign, ilp_assign, greedy_assign, greedy_rt, rank_based, mdp_assign, random_assign])))

lp, ilp, greedy_, greedy_rt, rank_based, mdp_, rnd = list(zip(*obj_scores))  

print(greedy_) 
print(ilp)
plt.plot(lp, label='Oracle LP')
plt.plot(ilp, label='Oracle ILP')
plt.plot(greedy_, label='Greedy')
plt.plot(greedy_rt, label='Greedy-RT')
plt.plot(rank_based, label='Rank Based')
plt.plot(mdp_, label='MDP')
plt.plot(rnd, label='Random')
plt.legend(title="Policy")
plt.xlabel("Trials with different random reviewer pools")
plt.ylabel("Objective scores")
plt.ylim(ymin=0)
plt.title('Policy comparison for real scores, drawn without replacement')
np.savetxt('assignments.txt', np.ravel(np.array(assignments)))
plt.show()
