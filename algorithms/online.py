import numpy as np
from scipy.stats import rankdata

def rank_assign(scores, review_time=6, min_reviewer_per_paper=3):
    (num_papers, num_reviewers) = scores.shape
    INF = 10
    
    last_assign = np.full(num_reviewers, INF)
    ranking_assign = np.zeros((num_papers, num_reviewers))

    for paper in range(num_papers):
        available_scores = np.where(last_assign >= review_time, scores[paper], -np.inf)
        
        # Rank reviewers based on availability
        available_ranks = np.argsort(np.argsort(np.where(available_scores > -np.inf, np.random.rand(num_reviewers), -np.inf)))
        
        # Rank the reviewers based on available_ranks
        top_reviewers = np.argsort(available_ranks)
        count_assigned = 0
        for reviewer_idx in top_reviewers:
            if count_assigned >= min_reviewer_per_paper:
                break
            if available_scores[reviewer_idx] != -np.inf:
                ranking_assign[paper, reviewer_idx] = 1
                last_assign[reviewer_idx] = 0
                count_assigned += 1

        last_assign += 1

    return ranking_assign

def rank_greedy_assign(scores, review_time=6, min_reviewer_per_paper=3, factor=1):
    (num_papers, num_reviewers) = scores.shape
    INF = 10
    
    last_assign = np.full(num_reviewers, INF)
    ranking_assign = np.zeros((num_papers, num_reviewers))

    for paper in range(num_papers):
        available_scores = np.where(last_assign >= review_time, scores[paper], -np.inf)
        
        # Rank reviewers based on availability
        available_ranks = np.argsort(np.argsort(np.where(available_scores > -np.inf, np.random.rand(num_reviewers), -np.inf)))
        
        # Modify the scores based on the rank and the factor
        modified_scores = available_scores + factor * available_ranks
        
        # Rank the reviewers based on modified scores and assign based on this ranking
        top_reviewers = np.argsort(-modified_scores)
        count_assigned = 0
        for reviewer_idx in top_reviewers:
            if count_assigned >= min_reviewer_per_paper:
                break
            if available_scores[reviewer_idx] != -np.inf:
                ranking_assign[paper, reviewer_idx] = 1
                last_assign[reviewer_idx] = 0
                count_assigned += 1

        last_assign += 1

    return ranking_assign

def greedy_rt_assign(scores, review_time=6, min_reviewer_per_paper=3, w_max=1):
    (num_papers, num_reviewers) = scores.shape
    INF = 10
    last_assign = np.full(num_reviewers, INF)
    greedy_rt_assign = np.zeros((num_papers, num_reviewers))
    
    # Calculate g and set the threshold T
    g = int(np.ceil(np.log(1 + w_max)))
    k = np.random.choice(list(range(g)))
    T = np.exp(k)

    for paper in range(num_papers):
        available_scores = np.where(last_assign >= review_time, scores[paper], 0)
        
        # Get the reviewers that meet the threshold and are available
        thresholded_reviewers = np.where(available_scores > T)[0]
        
        if len(thresholded_reviewers) >= min_reviewer_per_paper:
            # Select min_reviewer_per_paper reviewers from thresholded_reviewers
            chosen_reviewers = np.random.choice(thresholded_reviewers, min_reviewer_per_paper, replace=False)
            greedy_rt_assign[paper, chosen_reviewers] = 1
        else:
            # Fall back to the Greedy logic
            greedy_rt_assign[paper, rankdata(-available_scores) <= min_reviewer_per_paper] = 1

        last_assign += 1    
        last_assign[greedy_rt_assign[paper].astype(bool)] = 0

    return greedy_rt_assign