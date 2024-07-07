import numpy as np
import matplotlib.pyplot as plt
from importmonkey import add_path
add_path("../algorithms")
import greedy, rank, rankgreedy, greedyrt
from online import rank_greedy_assign, greedy_rt_assign, online_past_ones_with_lookahead

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
    plot_label, scores = score_dist
    
    print(f"RUNNING COMPUTATIONS FOR {plot_label}")    
    num_papers, num_reviewers = scores.shape
    # d_max s.t. R > r0 * d
    #d_values = range(1, num_reviewers * 4 // 5 // r0 , num_papers//10)
    r0 = 3
    d_values = range(1, num_reviewers // r0, num_reviewers//r0//20)
    f_values = [10**(-i) for i in range(4, 7, 1)]
    obj_scores = []

    for d in d_values:
        greedy_assignment = greedy.eval(scores, review_time=d, min_reviewer_per_paper=r0)
        rank_assignment = rank.eval(scores, review_time=d, min_reviewer_per_paper=r0)
        rank_greedy_assignment = [
            rankgreedy.eval(scores, review_time=d, min_reviewer_per_paper=r0, factor=f) for f in f_values]
        greedy_rt_assignment = greedyrt.eval(scores, review_time=d, min_reviewer_per_paper=r0)
        lookahead_assignment = 0 #np.sum(online_past_ones_with_lookahead(scores, scores, review_time=d, min_reviewer_per_paper=r0, lookahead=10) * scores)
        obj_scores.append(#list(map(lambda assign: obj_score(scores, assign),
        [greedy_assignment, greedy_rt_assignment, lookahead_assignment, rank_assignment] + rank_greedy_assignment)

    obj_scores = np.array(obj_scores) / num_papers / r0
    greedy1, greedy_rt, lookahead, r, rg1, rg2, rg3 = list(zip(*obj_scores))
    np.savetxt('delay_scaling_fast_results_iid.txt', obj_scores)
    ax.plot(d_values, greedy1, label='Greedy', alpha=0.5, marker='o', zorder=10)
    ax.plot(d_values, greedy_rt, label='Greedy RT', alpha=0.5, marker='o')
    #ax.plot(d_values, lookahead, label='Lookahead', alpha=0.5, marker='o')
    ax.plot(d_values, r, label='Rank', alpha=0.5, marker='o')
    ax.plot(d_values, rg1, label='Rank Greedy 1e-04', alpha=0.5, marker='o')
    ax.plot(d_values, rg2, label='Rank Greedy '+str(f_values[1]), alpha=0.5, marker='o')
    ax.plot(d_values, rg3, label='Rank Greedy '+str(f_values[2]), alpha=0.5, marker='o')
    ax.set(ylabel='Avg score per assignment', title=plot_label)
    if score_dist in score_dists[-2:]:
        ax.set(xlabel='Review time')
    #ax.label_outer()
    ax.legend(title='Policy')
    ax.get_legend().remove()
    #plt.ylim(ymin=0)
    #plt.title('Delay scaling')
    #ax.gca().set_facecolor('lightgray')
    #ax.grid(True, color='white', linestyle='-', linewidth=0.5)

score_dists = get_scores()
fig, axs = plt.subplots(3, 2, sharex=True) #, sharey=True)
for i, ax in enumerate(axs.flat):
    plot_for_scores(score_dists[i], ax)

lines, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(lines, labels, title='Policy', loc='center right', bbox_to_anchor=(1.0, 0.5))
plt.show()


