# compute probabilistically optimal online assignment of reviewers to papers
# using MDP with simulated score distribution
import numpy as np
from itertools import combinations
from scipy.stats import rankdata

def mdp(scores, delay = 3, gamma=0.9):
    num_papers, num_reviewers = scores.shape
    d = delay
    print(f"# reviewers = {num_reviewers}, delay = {delay}")

    #states = list(combinations(range(num_reviewers), reviews_per_paper))
    states = np.stack(np.meshgrid(*[np.arange(num_reviewers) for _ in range(d)]),
                      axis=-1).reshape(-1, d)

    init_expected_score = np.mean(np.max(scores, axis=1)) #* reviews_per_paper
    #values = { tuple(state) : init_expected_score / (1 - gamma) for state in states}
    values = np.ones(len(states)) * init_expected_score / (1 - gamma)

    def state2idx(state):
        return sum(n * d ** p for p, n in enumerate(reversed(state)))

    batch_size = 32
    num_batches = num_papers // batch_size
    err, epoch = 1, 0

    while err >= 1e-4 and epoch < 10:
        old_values = np.copy(values)

        for i in range(num_batches):
            batch = range(i * batch_size , (i +1) * batch_size)

            for idx, state in enumerate(states):
                available_reviewers = set(range(num_reviewers)) - set(state)

                def next_state(rev):
                    ns = list(state[1:]) + [rev]
                    return state2idx(ns)
                    return sum(n * d ** p for p, n in enumerate(reversed(ns)))
            
                def value_in_trial(paper):
                    q_values = [scores[paper, rev] + gamma
                                * values[next_state(rev)] for rev in available_reviewers]
                    return max(q_values)

                values[state2idx(state)] = np.mean([value_in_trial(paper) for paper in batch])

        #new_values = np.array(list(values.values()))
        #print(epoch, ": change in values ", np.linalg.norm(old_values - new_values))
        err = np.max(np.abs(old_values - values))
        print(epoch, f"value size: {np.max(old_values)}, change in values {err}")
        epoch += 1

    def policy(score_vec, state):
        # (takes score vector (s1 .. sr), busy reviewer set (r1 .. rl)) and returns tuple of assignments
        available_reviewers = set(range(num_reviewers)) - set(state)
        q_values = {rev : score_vec[rev] + gamma * values[
            state2idx(list(state[1:])+[rev])] for rev in available_reviewers}
        return max(q_values, key=q_values.get)

    return policy


def assign(scores, delay = 2):
    num_papers, num_reviewers = scores.shape
    #policy = mdp(np.random.permutation(scores))
    policy = mdp(scores, delay=delay)

    # assign greedy in first step
    mdp_assign = np.zeros(scores.shape)
    # mdp_assign[0, rankdata(scores[0]) <= 1] = 1
    assigned = [0]* delay

    for paper in range( num_papers):
        next_assign = policy(scores[paper], assigned)
        mdp_assign[paper, next_assign] = 1
        assigned = assigned[1:]+[next_assign]

    return mdp_assign

#scores = np.loadtxt('../similarity_result.txt')[:, 10:18]
scores = np.random.rand(20, 4)
#scores = np.array([[np.random.choice([1, 0.01]), 0, 0] for _ in range(64)])
mdp_assign = assign(scores, delay=2)
print(scores, mdp_assign)
