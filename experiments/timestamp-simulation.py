import numpy as np
import pandas as pd
from ast import literal_eval
#import matplotlib.pyplot as plt
#import oracle, greedy, randomassign
from importmonkey import add_path
add_path("../algorithms")
add_path("../dataset/paper_crawling")
import greedy
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

        print(len(available_reviewers), len(assigned_reviewers), reviewer_availability)

    return total_similarity_score

def trial():
    sim_scores = np.loadtxt('../similarity_result.txt')
    
    df = pd.read_csv('../dataset/paper_crawling/forum_times.csv')[:-2]
    df = df.sort_values(by='submission_timestamp')
    df['review_timestamps'] = df['review_timestamps'].apply(literal_eval)
    def calculate_review_times(row):
        return [review_time - row['submission_timestamp'] for review_time in row['review_timestamps'][:3]]
    df['review_times'] = df.apply(calculate_review_times, axis=1)
    print(df)
    arrival_times = list(df['submission_timestamp'])
    review_times = list(df['review_times'])
    
    #sim_scores = np.random.rand(10, 7)
    #print(sim_scores)
    d, r0 = 2, 3
    P, R = sim_scores.shape
    #arrival_times = list(range(P))
    #review_times = P*[(d,)*r0]
    

    print("time series similarity", assign_reviewers(arrival_times, review_times, sim_scores))
    print("greedy eval", greedy.eval(sim_scores, review_time=d-1, min_reviewer_per_paper=r0))

trial()
