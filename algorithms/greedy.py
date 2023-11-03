import numpy as np
from scipy.stats import rankdata

# def assign(scores, review_time = 6, min_reviewer_per_paper = 3):
#     (num_reviewers, num_papers) = scores.shape
#     INF = 10
#     last_assign = np.full(num_reviewers, INF)
#     greedy_assign = np.zeros((num_papers, num_reviewers))

#     for paper in range(num_papers):
#         available_scores = np.where(last_assign >= review_time, scores[paper], 0)
#         greedy_assign[paper, rankdata(-available_scores) <= min_reviewer_per_paper] = 1
#         last_assign += 1    
#         last_assign[greedy_assign[paper].astype(bool)] = 0

#     return greedy_assign



def assign(scores, review_time=6, min_reviewer_per_paper=3):
    (num_papers, num_reviewers) = scores.shape
    INF = 10
    last_assign = np.full(num_reviewers, INF)
    greedy_assign = np.zeros((num_papers, num_reviewers))

    for paper in range(num_papers):
        available_scores = np.where(last_assign >= review_time, scores[paper], 0)
        
        # Rank the scores
        top_reviewers = np.argsort(-available_scores)[:min_reviewer_per_paper]
        greedy_assign[paper, top_reviewers] = 1
        
        last_assign += 1    
        last_assign[greedy_assign[paper].astype(bool)] = 0

    return greedy_assign
