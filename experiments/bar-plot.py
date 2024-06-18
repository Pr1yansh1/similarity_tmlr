import numpy as np
import matplotlib.pyplot as plt
from importmonkey import add_path
add_path("../algorithms")
import oracle, greedy, mdp

print("\n Making plots ")
real_scores = np.loadtxt("../similarity_result.txt")
random_scores = np.random.rand(100, 10)
unfriendly_scores = np.array([[np.random.choice([1, 0.01]), 0, 0] for _ in range(100)])

low_rank_rand_scores = np.random.rand(100, 1) @ np.random.rand(1, 10)



score_dist_names = ['Real', 'Random', 'Unfriendly', 'Low-rank Random']
score_dists = [real_scores[:, :10], random_scores, unfriendly_scores, low_rank_rand_scores]
r0=1
obj_scores = []

#scores_dists = [scores / np.mean(scores) for scores in score_dists]

for scores in score_dists:
    print(scores.shape)
    ilp_score = np.mean(scores * oracle.ilp(scores, review_time =1, min_reviewer_per_paper=r0))
    greedy_score = greedy.eval(scores, review_time=1, min_reviewer_per_paper=r0) / scores.size
    mdp_score = np.mean(scores * mdp.assign(scores, reviews_per_paper=r0))

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
ax.set_title('Scores by Policy and Distribution')
ax.set_xticks(index + bar_width * (len(policies) - 1) / 2)
ax.set_xticklabels(score_dist_names)
ax.legend()
plt.show()
