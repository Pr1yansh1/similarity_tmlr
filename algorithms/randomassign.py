# compute a random assignment if reviewers to papers
import numpy as np
import random

def assign(scores, review_time = 6, min_reviewer_per_paper = 3):
    (num_papers, num_reviewers) = scores.shape
    INF = 10
    last_assign = np.full(num_reviewers, INF)
    random_assign = np.zeros((num_papers, num_reviewers))

    for paper in range(num_papers):
        available_reviewers = [r for r in range(num_reviewers) if last_assign[r] >= review_time]
        random_assign[paper, random.sample(available_reviewers, min_reviewer_per_paper)] = 1
        last_assign += 1    
        last_assign[random_assign[paper].astype(bool)] = 0

    return random_assign    
