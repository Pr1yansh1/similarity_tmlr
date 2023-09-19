import numpy as np
import random
import matplotlib.pyplot as plt
import oracle, greedy, mdp, randomassign

def obj_score(scores, assign):	
    return np.sum(scores * assign)

#scores = np.loadtxt("similarity_result.txt")
#scores = np.random.rand(30, 15)
#scores = np.random.rand(419, 1) @ np.random.rand(1, 10)
#scores += np.random.rand(419, 1) @ np.random.rand(1, 10)
#np.random.shuffle(scores)
#scores = np.array([[np.random.choice([1, 0.01]), 0, 0] for _ in range(64)])

print("\n Making plots ")
sim_scores = np.loadtxt('similarity_result.txt')
d, lambd = 2, 2
obj_scores = []
assignments = []

for trial in range(20, 35):
    scores = sim_scores[:, trial * 15 : trial * 15 + 15]
    #scores = np.array([[np.random.choice([1, 0.01]), 0, 0] for _ in range(64)])

    lp_assign = oracle.lp(scores, review_time = d, min_reviewer_per_paper =lambd)
    ilp_assign = oracle.ilp(scores, review_time = d, min_reviewer_per_paper =lambd)
    greedy_assign = greedy.assign(scores, review_time=d, min_reviewer_per_paper=lambd)
    mdp_assign = mdp.assign(scores, reviews_per_paper=lambd)
    random_assign = randomassign.assign(scores, review_time =d, min_reviewer_per_paper=lambd)

    randomly_drawn_scores = np.array(random.choices(scores, k = scores.shape[0]))
    assignments.append([scores, randomly_drawn_scores, lp_assign, ilp_assign, greedy_assign, mdp_assign])
    obj_scores.append( list(map(lambda assign: obj_score(randomly_drawn_scores, assign),
                                [lp_assign, ilp_assign, greedy_assign, mdp_assign, random_assign]))) 


lp, ilp, greedy, mdp, rnd = list(zip(*obj_scores)) 
plt.plot(lp, label='Oracle LP')
plt.plot(ilp, label='Oracle ILP')
plt.plot(greedy, label='Greedy')
plt.plot(mdp, label='MDP')
plt.plot(rnd, label='Random')
plt.legend(title="Policy")
plt.xlabel("Trials with different random reviewer pools")
plt.ylabel("Objective scores")
plt.ylim(ymin=0)
plt.title('Policy comparison for real scores, drawn without replacement')
#plt.title('Policy comparison for unfriendly scores')
#plt.title('Value histograms after each training epoch')
#plt.xlim(xmin=0.25, xmax = 0.5)
#plt.ylabel('Frequency')
#plt.xlabel('Value')
np.savetxt('assignments.txt', np.ravel(np.array(assignments)))
plt.show()
