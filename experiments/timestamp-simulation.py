import numpy as np
import random
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
from importmonkey import add_path
add_path("../algorithms")
import oracle

# Assignment policies

class Greedy:
    label = "Greedy"
    
    @staticmethod
    def assign(score_vec, available_reviewers, num_reviews_needed):
        greedy_order = lambda reviewer : score_vec[reviewer]
        available_reviewers.sort(key=greedy_order, reverse=True)
        return available_reviewers[:num_reviews_needed]

class Rank:
    label = "Rank"
    rank = np.argsort(np.random.rand(418))
    
    def assign(self, score_vec, available_reviewers, num_reviews_needed):
        rank_order = lambda reviewer : self.rank[reviewer]
        available_reviewers.sort(key=rank_order, reverse=True)
        return available_reviewers[:num_reviews_needed]
    
class RankGreedy:
    label = "RankGreedy"
    
    def __init__(self, eta):
        self.eta = eta
        self.rank = np.argsort(np.random.rand(418))
        
    def assign(self, score_vec, available_reviewers, num_reviews_needed):
        random_rank = np.argsort(np.random.rand(score_vec.size))        
        rg_order = lambda reviewer : score_vec[reviewer] + self.eta * self.rank[reviewer]
        available_reviewers.sort(key=rg_order, reverse=True)
        return available_reviewers[:num_reviews_needed]

class GreedyRT:
    label = "GreedyRT"
    NUM_DEFAULTS_GREEDYRT = 0
    threshold = np.exp(np.random.rand() * np.log(1.5)) - 1
    
    def assign(self, score_vec, available_reviewers, num_reviews_needed):
        reviewers_scores = list(zip(available_reviewers, score_vec))
        above_threshold = [(rev, score) for rev, score in reviewers_scores
                           if rev in available_reviewers and score > self.threshold]
        if len(above_threshold) >= num_reviews_needed:
            return [rev for rev, score in random.sample(above_threshold, num_reviews_needed)]
        else:
            self.NUM_DEFAULTS_GREEDYRT += 1
            return Greedy.assign(score_vec, available_reviewers, num_reviews_needed)

class Lookahead:
    label = "Lookahead"
    mean_review_time = 28 # days
    mean_reviews_needed = 3
    memory_size = 0
    score_memory = []
    
    def assign(self, score_vec, available_reviewers, num_reviews_needed):
        busy_reviewers = list(set(range(418)) - set(available_reviewers))
        copy_score_vec = np.array(score_vec)
        score_vec[busy_reviewers] = -np.inf

        # compute short oracle assignment
        predicted_score_matrix = np.array([score_vec] + self.score_memory)
        lp_assign = oracle.lp(predicted_score_matrix, self.mean_review_time, self.mean_reviews_needed)
        assignment = Greedy.assign(lp_assign[0], available_reviewers, num_reviews_needed)

        # update memory
        self.score_memory.insert(0, copy_score_vec)
        if len(self.score_memory) > self.memory_size:
            self.score_memory.pop()

        return assignment
    

# Assignment Simulation

def assign_reviewers(data_series, policy):
    arrival_times, review_times, similarity_matrix = data_series
    P, R = similarity_matrix.shape
    total_similarity_score = 0

    # Dictionary to store when each reviewer will be available
    reviewer_availability = {reviewer: 0 for reviewer in range(R)}

    for p in range(P):
        num_reviews_needed = len(review_times[p])

        # Check if enough reviewers are available
        available_reviewers = [reviewer for reviewer in range(R)
                               if reviewer_availability[reviewer] <= arrival_times[p]]
        
        if len(available_reviewers) < num_reviews_needed:
            raise ValueError(f"Not enough reviewers available for paper {p} at time {arrival_times[p]}")

        # Assign reviewers by policy
        assigned_reviewers = policy.assign(similarity_matrix[p], available_reviewers, num_reviews_needed)

        # Update their availability
        for i, rev in enumerate(assigned_reviewers):
            reviewer_availability[rev] = arrival_times[p] + review_times[p][i]
            total_similarity_score += similarity_matrix[p, rev]

    avg_reviews_per_paper = np.mean([len(times_per_paper) for times_per_paper in review_times])
    avg_similarity_per_assignment = total_similarity_score / P / avg_reviews_per_paper
    #print(f"Assignment statistics for {policy.label}")                
    #interarrival_times = [arrival_times[i+1]-arrival_times[i] for i in range(P-1)]
    #lam = 1/np.mean(interarrival_times)
    #mu = 1/np.mean(review_times)
    #r0 = np.mean([len(reviews) for reviews in review_times])
    #print("load", lam * r0 / mu / R)
    #ms_in_day = 1000 * 3600 * 24
    #print(lam * ms_in_day, mu * ms_in_day, r0, R, P)
    #plt.hist(interarrival_times, bins=50)
    #plt.hist([time for paper in review_times for time in paper], bins=50)
    #plt.hist(arrival_times, bins=50)
    #plt.show()
    
    return avg_similarity_per_assignment


# Timeseries Generators

def true_timeseries():
    sim_scores = np.loadtxt('../similarity_result.txt')
    
    df = pd.read_csv('../dataset/paper_crawling/forum_times_with_review_duration.csv')
    df['review_times'] = df['review_times'].apply(literal_eval)
    arrival_times = list(df['submission_timestamp'])
    review_times = list(df['review_times'])
    

    df2 = pd.read_csv('../dataset/statistics/tmlr_notes.csv')
    arrival_times = list(df2['cdate'])[:len(df)]
    arrival_times.sort()

    ms_in_day = 1000 * 3600 * 24
    arrival_times = np.array(arrival_times)/ms_in_day
    review_times = np.array(review_times)/ms_in_day
    
    return arrival_times, review_times, sim_scores

def uniform_timeseries(P, R, d, r0):
    sim_scores = np.random.rand(P, R)
    sim_scores = np.loadtxt('../similarity_result.txt')[:P, :R]
    arrival_times = list(range(P))
    review_times = P*[(d,)*r0]

    #print("greedy eval", greedy.eval(sim_scores, review_time=d-1, min_reviewer_per_paper=r0))
    return arrival_times, review_times, sim_scores

def poisson_timeseries(P, R, lam, mu, r0):
    sim_scores = np.random.rand(P, R)
    sim_scores = np.loadtxt('../similarity_result.txt')
    interarrival_times = [np.random.exponential(lam) for _ in range(P-1)]
    arrival_times = [sum(interarrival_times[:p]) for p in range(P)]
    review_times = [[np.random.exponential(mu) for _ in range(r0)] for _ in range(P)]
    print("poisson load", lam * r0 / mu / R)
    return arrival_times, review_times, sim_scores

def bursty_poisson_timeseries(P, R, lam1, lam2, window_size, mu, r0):
    sim_scores = np.random.rand(P, R)
    sim_scores = np.loadtxt('../similarity_result.txt')[:P, :R]

    # generate arrival timeseries
    arrival_times = []
    last_arrival_time = 0
    phase_end_time = window_size
    is_high_phase = True
    
    while len(arrival_times) < P:
        next_interarrival_time = np.random.exponential(lam1 if is_high_phase else lam2)
        if last_arrival_time + next_interarrival_time < phase_end_time:
            last_arrival_time += next_interarrival_time
            arrival_times.append(last_arrival_time)
        else:
            last_arrival_time = phase_end_time
            is_high_phase = not is_high_phase
            phase_end_time += window_size
            
    print("bursty high load", lam1 * r0/mu/R)
    print("bursty low load" , lam2 * r0/mu/R)

    review_times = [[np.random.exponential(mu) for _ in range(r0)] for _ in range(P)]
    return arrival_times, review_times, sim_scores    


# Computation

timeseries_labels = ["Real", "Uniform", "Poisson", "Bursty"]
timeseries = [true_timeseries(),
              uniform_timeseries(809, 418, 62, 3),
              poisson_timeseries(809, 418, 2.5, 0.04, 3),
              bursty_poisson_timeseries(809, 418, 5, 1, 10, 0.04, 3)]
policies = [Greedy(), GreedyRT(), Rank(),  RankGreedy(0.0001) , Lookahead()]
values = [list(map(lambda ts: assign_reviewers(ts, pc), timeseries)) for pc in policies]

greedyrt = policies[1]
print(greedyrt.NUM_DEFAULTS_GREEDYRT, greedyrt.threshold)

# Plots

fig, ax = plt.subplots()
bar_width = 0.15
index = np.arange(len(timeseries))

for i, policy in enumerate(policies):
    ax.bar(index + i * bar_width, values[i], bar_width, label=policy.label)

ax.set_ylabel('Scores')
ax.set_xticks(index + bar_width * (len(policies) - 1)/2)
ax.set_xticklabels(timeseries_labels)
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.tight_layout()
plt.show()
