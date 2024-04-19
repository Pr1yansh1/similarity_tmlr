import numpy as np
import matplotlib.pyplot as plt
import oracle, greedy

def obj_score(scores, assign):	
    return np.sum(scores * assign)


print("\n Making plots ")
sim_scores = np.loadtxt('similarity_result.txt').T
print(np.shape(sim_scores))

d_values = list(range(1, 9))
policy_performance = []

for d in d_values:
    obj_scores = []
    for trial in range(4):
        scores = sim_scores[:100, trial * 35 : trial*35 + 35]
        ilp_assign = oracle.ilp(scores, review_time =d)
        greedy_assign = greedy.assign(scores, review_time =d)
        obj_scores.append(list(map(lambda assign: obj_score(scores, assign),
                          [ilp_assign, greedy_assign])))
        
    perf_over_trials_for_d = np.mean(np.array(obj_scores), axis=0)
    print(np.shape(np.array(obj_scores)), np.shape(perf_over_trials_for_d))
    policy_performance.append(perf_over_trials_for_d)

ilp, greedy = list(zip(*policy_performance))
plt.plot(d_values, ilp, label='ILP')
plt.plot(d_values, greedy, label='Greedy')
plt.xlabel('Review time(d)')
plt.ylabel('Avg. objective score over trials')
plt.ylim(ymin=0)
plt.title('Oracle-Greedy gap as function of review time lag (P=100, R=35)')
        
plt.show()
