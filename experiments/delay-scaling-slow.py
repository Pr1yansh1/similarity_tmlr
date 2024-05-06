import numpy as np
import matplotlib.pyplot as plt
from importmonkey import add_path
add_path("../algorithms")
import oracle, greedy

def obj_score(scores, assign):	
    return np.sum(scores * assign)


print("\n Making plots ")
sim_scores = np.loadtxt("../similarity_result.txt")
sim_scores = np.random.rand(*sim_scores.shape)
print(np.shape(sim_scores))

policy_means, policy_stds = [], []
paper_sample_size = 100
reviewer_sample_size = 8
d_values = list(range(1, reviewer_sample_size * 4 // 5))

for d in d_values:
    obj_scores = []
    for trial in range(5):
        scores = sim_scores[:paper_sample_size,
                            reviewer_sample_size * trial : reviewer_sample_size * (trial+1)]
        ilp_score = np.sum(scores * oracle.ilp(scores, review_time =d, min_reviewer_per_paper=1))
        greedy_score = greedy.eval(scores, review_time=d, min_reviewer_per_paper=1)
        obj_scores.append([ilp_score, greedy_score])

    obj_scores = np.array(obj_scores) / paper_sample_size
    policy_means.append(np.mean(obj_scores, axis=0))
    policy_stds.append(np.std(obj_scores, axis=0))

ilp, greedy = list(zip(*policy_means))
ilp_err, greedy_err = list(zip(*policy_stds))
print("ILP", ilp, ilp_err, "GREEDY", greedy, greedy_err, sep='\n')
plt.errorbar(d_values, ilp, yerr=ilp_err, label='ILP')
plt.errorbar(d_values, greedy, yerr=greedy_err, label='Greedy')
plt.xlabel('Review time')
plt.ylabel('Mean similarity score per assignment')
plt.ylim(ymin=0)
#plt.title('Oracle-Greedy gap as function of review time lag (P=100, R=35)')
plt.legend(title='Policy')
plt.gca().set_facecolor('lightgray')
plt.grid(True, color='white', linestyle='-', linewidth=0.5)
plt.show()
