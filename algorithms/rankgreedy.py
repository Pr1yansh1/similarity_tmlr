import numpy as np
from scipy.stats import rankdata
import math

def eval(scores, review_time=6, min_reviewer_per_paper=3, factor=1):
    num_papers, num_reviewers = scores.shape
    last_assign = np.full(num_reviewers, review_time + 10)
    obj_score = 0
    

    for paper in range(num_papers):
        available_scores = np.where(last_assign >= review_time, scores[paper], -np.inf)
        
        # Rank reviewers based on availability
        available_ranks = np.argsort(np.argsort(np.where(
            available_scores > -np.inf, np.random.rand(num_reviewers), -np.inf)))
        
        # Modify the scores based on the rank and the factor
        modified_scores = available_scores + factor * available_ranks
        
        # Rank the reviewers based on modified scores and assign based on this ranking
        top_reviewers = np.argsort(-modified_scores)[:min_reviewer_per_paper]
        obj_score += np.sum(available_scores[top_reviewers])
        last_assign += 1
        last_assign[top_reviewers] = 0

    return obj_score #ranking_assign
