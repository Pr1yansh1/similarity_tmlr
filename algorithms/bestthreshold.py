import numpy as np
from scipy.stats import rankdata
INF = 1E10

#num_papers, num_reviewers = 100, 1
#scores = np.random.rand(num_papers, num_reviewers)

default_d = 100

def eval(scores, eta, d = default_d, r0 = 2):
    num_papers, num_reviewers = scores.shape
    #W = np.zeros(num_reviewers)
    last_assign = np.full(num_reviewers, 10)
    obj_score = 0
    
    for paper in range(num_papers):
        threshold_reviewers = np.where((last_assign >= d) * (scores >= eta))[0]
        if len(threshold_reviewers) >= r0:
            # sufficient reviewers with similarity above threshold
            # choose randomly among them
            assign_reviewers = np.random.choice(threshold_reviewers, size=r0, replace=False)
        else:
            # greedy
            assign_reviewers = rankdata(-available_scores) <= r0
            
        available_scores = np.where(last_assign >= d and scores >= eta, 1, 0)
        W_assign = np.zeros(num_reviewers)        
        W_assign[rankdata(-available_scores) <= r0] = 1
        obj_score += np.dot(scores[paper], W_assign)
        
        last_assign += 1
        last_assign[W_assign.astype(bool)] = 0

    return obj_score

def step(scores, W0, scale,  num_trials=5, d = default_d, r0 = 2):
    best_value, best_W = -INF, np.zeros(R)
    for _ in range(num_trials):
        W = W0 + np.random.uniform(-scale, scale, R)
        value = eval(scores, W, d, r0)
        if value > best_value:
            best_value, best_W = value, W
    return best_value, best_W


import matplotlib.pyplot as plt

scores = np.loadtxt("../similarity_result.txt")
P, R = scores.shape
scale = 0.1* np.mean(scores)

num_steps = 8
W0 = np.zeros(R)
obj_score0 = eval(scores, W0)
obj_scores = [(0, obj_score0)]


for t in range(1, num_steps):
    obj_score1, W1 = step(scores, W0, scale, num_trials=50)
    if obj_score1 > obj_score0:
        W0 = W1
    obj_scores.append((t, obj_score1))

plt.plot(*tuple(zip(*obj_scores)))

#greedy_scores = []
#perturbed_scores1 = []
#perturbed_scores2 = []
#optimized_scores = []

#for d in range(1, P//2, P//20):
 #   greedy_scores.append((d, eval(scores, 0, d=d)))
  #  optimized_scores.append((d, step(scores, scale=np.mean(scores)*0.1, d=d)))

    #for _ in range(num_trials):
     #   perturbed_scores1.append((d, eval(scores, W * np.random.rand(R), d=d)))
      #  perturbed_scores2.append((d, eval(scores, 0.1 * W * np.random.rand(R), d=d)))


#plt.scatter(*tuple(zip(*greedy_scores)), label="Greedy", alpha=0.5, color='red')
#plt.scatter(*tuple(zip(*optimized_scores)), label="Optim", alpha=0.5, color='green')
#plt.scatter(*tuple(zip(*perturbed_scores1)), label="Perturb1", alpha=0.2, color='green')
#plt.scatter(*tuple(zip(*perturbed_scores2)), label="Perturb2", alpha=0.2, color='yellow')
#plt.legend(title='Policy')
plt.show()
        
        
    

