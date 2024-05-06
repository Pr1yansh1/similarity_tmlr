import numpy as np
from scipy.stats import rankdata
from oracle import lp, ilp 
import math

def eval(scores, review_time=6, min_reviewer_per_paper=3, w_max=1, scaling_factor = 10): 
    num_papers, num_reviewers = scores.shape
    last_assign = np.full(num_reviewers, review_time + 10)
    obj_score = 0
    chosen_reviewers = []

    w_max = np.max(scores)
    
    # Calculate g and set the threshold T
    g = int(np.ceil(np.log(1 + w_max)))
    k = np.random.choice(list(range(g)))
    T = np.exp(k) / scaling_factor

    # print(w_max)
    # print("One threshold for greedy RT:", T)

    greedy = 0
    normal = 0 

    for paper in range(num_papers):
        available_scores = np.where(last_assign >= review_time, scores[paper], -np.inf)
        
        # Get the reviewers that meet the threshold and are available
        thresholded_reviewers = np.where(available_scores > T)[0]
        
        if len(thresholded_reviewers) >= min_reviewer_per_paper:
            normal += 1
            # Select min_reviewer_per_paper reviewers from thresholded_reviewers
            chosen_reviewers = np.random.choice(thresholded_reviewers,
                                                min_reviewer_per_paper, replace=False)
            obj_score += np.sum(available_scores[chosen_reviewers])
            #print("threshold", T, available_scores[ chosen_reviewers], obj_score)
            #print(available_scores, chosen_reviewers)            
            
        else:
            # Fall back to the Greedy logic
            greedy += 1 
            chosen_reviewers = np.argsort(-available_scores)[:min_reviewer_per_paper]
            obj_score += np.sum(available_scores[chosen_reviewers])
            #print("greedy", available_scores[ chosen_reviewers],
                  #np.sum(available_scores[chosen_reviewers]), obj_score)
            #print(available_scores, chosen_reviewers)            
        last_assign += 1    
        last_assign[chosen_reviewers] = 0

    #print("algorithm has greedy, normal", greedy, normal)
    return obj_score #greedy_rt_assign
