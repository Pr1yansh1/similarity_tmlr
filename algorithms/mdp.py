from scipy.stats import rankdata
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt



def mdp(scores, reviews_per_paper = 2):
    gamma = 0.9
    num_papers, num_reviewers = scores.shape

    print(f"# reviewers = {num_reviewers}, # reviews per paper = {reviews_per_paper}")

    # initialize
    states = list(combinations(range(num_reviewers), reviews_per_paper))
    init_expected_score = np.mean(np.max(scores, axis=1)) * reviews_per_paper
    values = { state : init_expected_score / (1 - gamma) for state in states }

    batch_size = 32
    num_batches = num_papers // batch_size

    for epoch in range(5):
        old_values = np.array(list(values.values()))

        for i in range(num_batches):
            batch = range(i * batch_size , (i +1) * batch_size)

            for state in states:
                available_reviewers = set(range(num_reviewers)) - set(state)
                actions = list(combinations(available_reviewers, reviews_per_paper))
            
                def value_in_trial(paper):
                    q_values = [np.sum(scores[paper, action]) + gamma * values[action] \
                        for action in actions]
                    return max(q_values)

                values[state] = np.mean([value_in_trial(paper) for paper in batch])

        new_values = np.array(list(values.values()))
        print(epoch, ": change in values ", np.linalg.norm(old_values - new_values))

    def policy(score_vec, state):
        # (takes (s1 .. sr), (r1,  rl)) and returns tuple of assignments
        available_reviewers = set(range(num_reviewers)) - set(state)
        actions = list(combinations(available_reviewers, reviews_per_paper))
        q_values = {action : np.sum(score_vec[list(action)]) + gamma * values[action] \
                        for action in actions}
        return max(q_values, key=q_values.get)

    return policy


def find_mdp(scores, reviews_per_paper = 2):
    num_papers, num_reviewers = scores.shape
    policy = mdp(np.random.permutation(scores))

    # assign greedy in first step
    mdp_assign = np.zeros(scores.shape)
    mdp_assign[0, rankdata(scores[0]) <= reviews_per_paper] = 1
    assigned = tuple(np.where(mdp_assign[0])[0])

    for paper in range(1, num_papers):
        assigned = policy(scores[paper], assigned)
        mdp_assign[paper, assigned] = 1

    return mdp_assign

scores = np.loadtxt('similarity_result.txt')[:, :10]
np.random.shuffle(scores)
mdp_assign = find_mdp(scores)