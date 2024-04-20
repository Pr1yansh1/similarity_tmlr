import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from importmonkey import add_path
add_path("..")
import oracle
EPS = 1E-4
bad_instances = []

def obj_score(scores, assign):	
    return np.sum(scores * assign)

def solve_random_instance(p, r, d):
    scores = np.random.rand(p, r)
    lp_soln = oracle.lp(scores, review_time=d, min_reviewer_per_paper=1)
    ilp_soln = oracle.ilp(scores, review_time=d, min_reviewer_per_paper=1)
    lp_score = obj_score(scores, lp_soln)
    ilp_score = obj_score(scores, ilp_soln)
    if np.linalg.norm(lp_score - ilp_score) >= EPS:
        bad_instances.append((d, lp_score, ilp_score)) #, scores)
    return lp_score, ilp_score

def compute_scores_by_d(d_max, num_trials):
    p = 20
    r = d_max*2
    
    lp_scores = {}
    ilp_scores = {}
    
    for d in range(1, d_max):
        #lp_scores[d], ilp_scores[d] = [], []
        r = d+5
        obj_scores = [solve_random_instance(p, r, d) for _ in range(num_trials)]
        lp_scores[d], ilp_scores[d] = zip(*obj_scores)

    return lp_scores, ilp_scores

def plot_integrality(d_max, num_trials):
    #d_max, num_trials = 10, 50
    lp_scores, ilp_scores = compute_scores_by_d(d_max, num_trials)
    lp_scatter = lp_scores.values()
    ilp_scatter = ilp_scores.values()
    d_scatter = [d for d in range(1,d_max) for _ in range(num_trials)]
    print(lp_scatter, len(lp_scatter))

    lp_plot, ilp_plot = [], []
    for d, obj_scores in lp_scores.items():
        lp_plot.append(np.mean(obj_scores))
        
    for d, obj_scores in ilp_scores.items():
        ilp_plot.append(np.mean(obj_scores))

    print(bad_instances)
    print(len(bad_instances))
    plt.scatter(d_scatter, lp_scatter, label='lp', alpha=.5)
    plt.scatter(d_scatter, ilp_scatter, label='ilp', alpha=.5)
    plt.plot(range(1,d_max), lp_plot)
    plt.plot(range(1,d_max), ilp_plot)

    plt.xlabel('Review time (d)')
    plt.ylabel('Aggregate similarity scores')
    plt.title('LP vs ILP objective values by review time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_bad_instances(d_max, num_trials):
    _ = compute_scores_by_d(d_max, num_trials)
    ds, lp_scores, ilp_scores = zip(*bad_instances)
    bads_by_d = defaultdict(list)
    for d, lp_score, ilp_score in bad_instances:
        bads_by_d[d].append(lp_score - ilp_score)

    prob_bads = []
    for d in range(1, d_max):
        print(d, bads_by_d[d], len(bads_by_d[d]), len(bads_by_d[d])/num_trials)
        prob_bad = len(bads_by_d[d])/ num_trials
        prob_bads.append(prob_bad)

    print(prob_bads)
    with open('integrality_probability.txt', 'w') as f:
        f.write(str(prob_bads))
            
    print('Plotting')
    plt.plot(list(range(1,d_max)), prob_bads)
    
    #plt.scatter(ds, lp_scores, label='lp', alpha=.5)
    #plt.scatter(ds, ilp_scores, label='ilp', alpha=.5)
    plt.xlabel('Review time (d)')
    #plt.ylabel('Aggregate similarity scores')
    #plt.title('Probability that LP instance is not integral')
    #plt.legend()
    #plt.grid(True)
    plt.ylabel('Probrability of non-integrality')
    plt.show()    
    
plot_bad_instances(10, 10000)
#plot_integrality(10, 50)
    
    


    
