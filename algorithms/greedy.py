import numpy as np
from scipy.stats import rankdata

def eval(scores, review_time = 6, min_reviewer_per_paper = 3):
    num_papers, num_reviewers = scores.shape
    INF = 1E8
    last_assign = np.full(num_reviewers, review_time + 10)
    obj_score = 0

    for paper in range(num_papers):
        available_scores = np.where(last_assign >= review_time, scores[paper], -np.inf)
        top_reviewers = np.argsort(-available_scores)[:min_reviewer_per_paper]
        #greedy_assign[paper, rankdata(-available_scores) <= min_reviewer_per_paper] = 1
        last_assign += 1
        last_assign[top_reviewers] = 0
        obj_score += np.sum(available_scores[top_reviewers])

    return obj_score




def assign(scores, review_time = 6, min_reviewer_per_paper = 3):
    num_papers, num_reviewers = scores.shape
    INF = 1E8
    last_assign = np.full(num_reviewers, INF)
    greedy_assign = np.zeros((num_papers, num_reviewers))
    obj_score = 0

    for paper in range(num_papers):
        available_scores = np.where(last_assign >= review_time, scores[paper], 0)
        greedy_assign[paper, rankdata(-available_scores) <= min_reviewer_per_paper] = 1
        last_assign += 1
        last_assign[greedy_assign[paper].astype(bool)] = 0
        obj_score += np.dot(available_scores, greedy_assign[paper])
     

    return greedy_assign





