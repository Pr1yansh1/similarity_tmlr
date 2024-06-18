import numpy as np
import matplotlib.pyplot as plt
from importmonkey import add_path
add_path("../algorithms")
import oracle, greedy, mdpdelay

print("\n Making plots ")
P, R, d, r0 = 100, 50, 2, 1
real_scores = np.loadtxt("../similarity_result.txt")[:P, :R]
random_scores = np.random.rand(P, R)
unfriendly_scores = np.array([[np.random.choice([1, 0.01])]+[0]*(R-1) for _ in range(P)])
low_rank_rand_scores = np.random.rand(P, 1) @ np.random.rand(1, R)
exp_real_scores = np.exp(real_scores)
exp_real_scores /= np.max(exp_real_scores)
clustered_real_scores = np.loadtxt("../dataset/similarity_matrix_reordered.txt")[:P, :R]

score_dists = [('Random', random_scores), ('Real', real_scores),
    ('Exp.\nReal', exp_real_scores), ('Clustered\nReal', clustered_real_scores),
    ('Low-Rank\nRandom', low_rank_rand_scores), ('Unfriendly', unfriendly_scores)]
score_dist_names, score_dist_matrices = zip(*score_dists)
obj_scores = []

#scores_dists = [scores / np.mean(scores) for scores in score_dists]

for scores in score_dist_matrices:
    print(scores.shape)
    ilp_score = np.mean(scores * oracle.ilp(scores, review_time =d, min_reviewer_per_paper=r0))
    greedy_score = greedy.eval(scores, review_time=d, min_reviewer_per_paper=r0) / scores.size
    mdp_score = np.mean(scores * mdpdelay.assign(scores, delay=d))

    #obj_scores.append([ilp_score, mdp_score, greedy_score])
    obj_scores.append([1, mdp_score/ilp_score, greedy_score/ilp_score])
    #policy_means.append(np.mean(obj_scores, axis=0))
    #policy_stds.append(np.std(obj_scores, axis=0))


print(obj_scores)
obj_scores = list(zip(*obj_scores))
print(obj_scores)
#ilp_err, greedy_err = list(zip(*policy_stds))
#print("ILP", ilp, ilp_err, "GREEDY", greedy, greedy_err, sep='\n')

fig, ax = plt.subplots()
bar_width = 0.2
index = np.arange(len(score_dists))
policies = ['ILP', 'MDP', 'Greedy']

for i, policy in enumerate(policies):
    ax.bar(index + i * bar_width, obj_scores[i], bar_width, label=policy)
    
ax.set_ylabel('Scores')
#ax.set_title('Scores by Policy and Distribution')
ax.set_xticks(index + bar_width * (len(policies) - 1) / 2)
ax.set_xticklabels(score_dist_names)
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
#plt.show()
plt.tight_layout()
plt.savefig('bar-plot-delay-2v5.pgf')
plt.savefig('bar-plot-delay-2v5.pdf')
plt.show()
