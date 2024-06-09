import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
#import oracle, greedy, randomassign
#from importmonkey import add_path
#add_path("../algorithms")
#import greedy
#from online import rank_greedy_assign, greedy_rt_assign

def assign_reviewers(arrival_times, review_times, similarity_matrix):
    P, R = similarity_matrix.shape
    total_similarity_score = 0

    # Dictionary to store when each reviewer will be available
    reviewer_availability = {reviewer: 0 for reviewer in range(R)}

    for p in range(P):
        # Get the arrival time and required review times for the current paper
        arrival_time = arrival_times[p]
        required_review_times = review_times[p]
        num_reviews_needed = len(required_review_times)

        # List to store the reviewers assigned for the current paper
        assigned_reviewers = []

        # Check if enough reviewers are available
        available_reviewers = [reviewer for reviewer in range(R) if reviewer_availability[reviewer] <= arrival_time]
        if len(available_reviewers) < num_reviews_needed:
            raise ValueError(f"Not enough reviewers available for paper {p} at time {arrival_time}")

        # Sort the available reviewers by similarity score for the current paper
        available_reviewers.sort(key=lambda reviewer: similarity_matrix[p][reviewer], reverse=True)

        # Assign the top reviewers based on similarity scores
        for i in range(num_reviews_needed):
            best_reviewer = available_reviewers[i]
            assigned_reviewers.append(best_reviewer)
            reviewer_availability[best_reviewer] = arrival_time + required_review_times[i]
            total_similarity_score += similarity_matrix[p][best_reviewer]

        #print(len(available_reviewers), len(assigned_reviewers), reviewer_availability)

    return total_similarity_score


def true_timeseries():
    sim_scores = np.loadtxt('../similarity_result.txt')
    
    df = pd.read_csv('../dataset/paper_crawling/forum_times.csv')[:-2]
    print("Loaded timeseries", df)
    df = df.sort_values(by='submission_timestamp')
    df['review_timestamps'] = df['review_timestamps'].apply(literal_eval)
    def calculate_review_times(row):
        return [review_time - row['submission_timestamp'] for review_time in row['review_timestamps'][:3]]
    df['review_times'] = df.apply(calculate_review_times, axis=1)
    #df.to_csv('../dataset/paper_crawling/forum_times_with_review_duration.csv')
    #df = pd.to_csv('../dataset/paper_crawling/forum_times_with_review_duration.csv')
    
    print("Processed timeseries", df)
    
    arrival_times = list(df['submission_timestamp'])
    review_times = list(df['review_times'])

    print("Computing load..")
    P, R = sim_scores.shape
    interarrival_times = [arrival_times[i+1]-arrival_times[i] for i in range(P-1)]
    lam = 1/np.mean(interarrival_times)
    mu = 1/np.mean(review_times)
    r0 = np.mean([len(reviews) for reviews in review_times])
    print("true load", lam * r0 / mu / R)
    ms_in_day = 1000 * 3600 * 24
    print(lam * ms_in_day, mu * ms_in_day, r0, R, P)
    #plt.hist(interarrival_times, bins=50)
    #plt.hist([time for paper in review_times for time in paper], bins=50)
    plt.hist(arrival_times, bins=50)
    plt.show()
    
    return arrival_times, review_times, sim_scores

def uniform_timeseries(P, R, d, r0):
    sim_scores = np.random.rand(P, R)
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
    

print("uniform time series similarity", assign_reviewers(*uniform_timeseries(10, 7, 2, 3)))
print("poisson time series similarity", assign_reviewers(*poisson_timeseries(809, 418, 2.5, 0.04, 3)))
print("true time series similarity",    assign_reviewers(*true_timeseries()))
print("bursty poisson arrival similarity", assign_reviewers(*bursty_poisson_timeseries(809, 418, 5, 1, 10, 0.04, 3)))
