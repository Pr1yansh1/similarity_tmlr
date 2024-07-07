import numpy as np
import matplotlib.pyplot as plt
from importmonkey import add_path
add_path("../algorithms")
import oracle, greedy

def get_scores():
    P, R = 809, 418
    #P, R, r0 = 100, 50, 1
    real_scores = np.loadtxt("../similarity_result.txt")[:P, :R]
    random_scores = np.random.rand(P, R)
    unfriendly_scores = np.array([[np.random.choice([1, 0.01], p= [0.5, 0.5])]
                                  +[0]*(R-1) for _ in range(P)])
    low_rank_rand_scores = np.random.rand(P, 1) @ np.random.rand(1, R)
    exp_real_scores = np.exp(real_scores)
    exp_real_scores /= np.max(exp_real_scores)
    clustered_real_scores = np.loadtxt("../dataset/similarity_matrix_reordered.txt")[:P, :R]
    
    return [('Random', random_scores), ('Real', real_scores),
                   ('Exp. Real', exp_real_scores), ('Clustered Real', clustered_real_scores),
                   ('Low-Rank Random', low_rank_rand_scores), ('Unfriendly', unfriendly_scores)]

def plot_for_scores(score_dist, ax):
    plot_label, sim_scores = score_dist
    print(f"Plot for {plot_label}")
    num_papers, num_reviewers = sim_scores.shape

    paper_sample_size = 100
    reviewer_sample_size = 35
    d_values = list(range(1, reviewer_sample_size * 4 // 5))
    num_trials = 5
    policy_means, policy_stds = [], []

    for d in d_values:
        obj_scores = []
        for trial in range(num_trials):
            scores = sim_scores[:paper_sample_size,
                            reviewer_sample_size * trial : reviewer_sample_size * (trial+1)]
            ilp_score = np.sum(scores * oracle.ilp(scores, review_time =d, min_reviewer_per_paper=1))
            greedy_score = greedy.eval(scores, review_time=d, min_reviewer_per_paper=1)
            obj_scores.append([ilp_score, greedy_score])

        obj_scores = np.array(obj_scores) / paper_sample_size
        policy_means.append(np.mean(obj_scores, axis=0))
        policy_stds.append(np.std(obj_scores, axis=0))

    ilp, greedy1 = list(zip(*policy_means))
    ilp_err, greedy_err = list(zip(*policy_stds))
    #print(len(d_values), "ILP", len(ilp), len(ilp_err), "GREEDY", len(greedy1), len(greedy_err), sep='\n')
    print("ILP", ilp, ilp_err, "Greedy", greedy1, greedy_err, sep='\n')
    ax.errorbar(d_values, ilp, yerr=ilp_err, label='ILP')
    ax.errorbar(d_values, greedy1, yerr=greedy_err, label='Greedy')
    ax.set(ylabel='Avg score per assignment', title=plot_label)
    if plot_label == 'Unfriendly':
        ax.set_ylim(ymin=0)
    if plot_label == score_dists[-2][0] or plot_label == score_dists[-1][0]:
        ax.set(xlabel='Review time')
    #plt.title('Oracle-Greedy gap as function of review time lag (P=100, R=35)')
    ax.legend(title='Policy')
    #lt.gca().set_facecolor('lightgray')
    #lt.grid(True, color='white', linestyle='-', linewidth=0.5)
    ax.get_legend().remove()

score_dists = get_scores()#[-1:]
fig, axs = plt.subplots(3, 2, sharex=True)
for i, ax in enumerate(axs.flat):
    plot_for_scores(score_dists[i], ax)

lines, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(lines, labels, title='Policy', loc='center right', bbox_to_anchor=(1.0, 0.5))

plt.show()
