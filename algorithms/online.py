import numpy as np
from scipy.stats import rankdata
from oracle import lp, ilp 
import math

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

def greedy_rt_assign(scores, review_time=6, min_reviewer_per_paper=3, w_max=1, scaling_factor = 10): 

    modified_scores = scores #* scaling_factor
    (num_papers, num_reviewers) = scores.shape
    INF = 10
    last_assign = np.full(num_reviewers, INF)
    greedy_rt_assign = np.zeros((num_papers, num_reviewers))

    w_max = np.max(modified_scores)
    
    # Calculate g and set the threshold T
    g = int(np.ceil(np.log(1 + w_max)))
    k = np.random.choice(list(range(g)))
    T = np.exp(k)

    # print(w_max)
    # print("One threshold for greedy RT:", T)

    greedy = 0 
    normal = 0 

    for paper in range(num_papers):
        available_modified_scores = np.where(last_assign >= review_time, modified_scores[paper], 0)
        
        # Get the reviewers that meet the threshold and are available
        thresholded_reviewers = np.where(available_modified_scores > T)[0]
        
        if len(thresholded_reviewers) >= min_reviewer_per_paper:
            normal += 1
            # Select min_reviewer_per_paper reviewers from thresholded_reviewers
            chosen_reviewers = np.random.choice(thresholded_reviewers, min_reviewer_per_paper, replace=False)
            greedy_rt_assign[paper, chosen_reviewers] = 1
        else:
            # Fall back to the Greedy logic
            greedy += 1 
            top_reviewers = np.argsort(-available_modified_scores)[:min_reviewer_per_paper]
            greedy_rt_assign[paper, top_reviewers] = 1

        last_assign += 1    
        last_assign[greedy_rt_assign[paper].astype(bool)] = 0

    #print("algorithm has greedy, normal", greedy, normal)
    return greedy_rt_assign

# def online_past_ones_with_lookahead(actual_scores, sampling_scores, review_time, min_reviewer_per_paper, lookahead=20, solve_method="lp", sample_with_replacement=True):
#     num_papers, num_reviewers = actual_scores.shape
#     assignment = np.zeros((num_papers, num_reviewers))
#     last_assign = np.full(num_reviewers, review_time)

#     for t in range(num_papers):
#         # Create an augmented score matrix for lookahead
#         augmented_scores = np.zeros((lookahead, num_reviewers))
#         augmented_scores[0] = actual_scores[t]

#         # Adjust the sampling process to match the number of reviewers
#         for i in range(1, lookahead):
#             if sample_with_replacement:
#                 # Randomly select a paper's index with replacement
#                 sample_idx = np.random.choice(sampling_scores.shape[1])
#             else:
#                 # Move through the papers without replacement, wrapping around if needed
#                 sample_idx = (t + i) % sampling_scores.shape[1]

#             # Sample scores for the selected paper, matching the number of reviewers
#             sampled_scores = sampling_scores[:, sample_idx]
#             augmented_scores[i, :len(sampled_scores)] = sampled_scores[:num_reviewers]

#         # Solve the assignment using an offline method
#         if solve_method == "lp":
#             lookahead_assignment = lp(augmented_scores, review_time, min_reviewer_per_paper)
#         else:
#             lookahead_assignment = ilp(augmented_scores, review_time, min_reviewer_per_paper)

#         # Assign reviewers for the current paper
#         current_assignment = lookahead_assignment[0]
#         assignment[t] = current_assignment

#         # Update last assignment time for reviewers
#         for reviewer in range(num_reviewers):
#             if current_assignment[reviewer] == 1:
#                 last_assign[reviewer] = 0
#             else:
#                 if last_assign[reviewer] < review_time:
#                     last_assign[reviewer] += 1

#     return assignment



def online_past_ones_with_lookahead(actual_scores, sampling_scores, review_time, min_reviewer_per_paper, lookahead=20, solve_method="lp", sample_with_replacement=True):
    num_papers, num_reviewers = actual_scores.shape
    assignment = np.zeros((num_papers, num_reviewers))
    last_assign = np.full(num_reviewers, review_time)

    for t in range(num_papers):
        # Check available reviewers
        available_reviewers = [i for i in range(num_reviewers) if last_assign[i] >= review_time]

        # Create an augmented score matrix for lookahead with available reviewers
        augmented_scores = np.zeros((lookahead, len(available_reviewers)))
        augmented_scores[0] = actual_scores[t, available_reviewers]

        for i in range(1, lookahead):
            if sample_with_replacement:
                sample_idx = np.random.choice(sampling_scores.shape[1])
            else:
                sample_idx = (t + i) % sampling_scores.shape[1]

            sampled_scores = sampling_scores[:, sample_idx]
            augmented_scores[i, :len(sampled_scores)] = sampled_scores[available_reviewers]

        # Solve the assignment using an offline method
        if solve_method == "lp":
            lookahead_assignment = lp(augmented_scores, review_time, min_reviewer_per_paper)
        else:
            lookahead_assignment = ilp(augmented_scores, review_time, min_reviewer_per_paper)

        # Map back lookahead assignment to original reviewers
        current_assignment = np.zeros(num_reviewers)
        for i, reviewer in enumerate(available_reviewers):
            current_assignment[reviewer] = lookahead_assignment[0][i]

        assignment[t] = current_assignment

        # Update last assignment time for reviewers
        for reviewer in range(num_reviewers):
            if current_assignment[reviewer] == 1:
                last_assign[reviewer] = 0
            else:
                if last_assign[reviewer] < review_time:
                    last_assign[reviewer] += 1

    return assignment


def variable_threshold_assign(scores, review_time=6, min_reviewer_per_paper=3):
    (num_papers, num_reviewers) = scores.shape
    INF = 10
    last_assign = np.full(num_reviewers, INF)
    variable_threshold_assign = np.zeros((num_papers, num_reviewers))

    w_max = math.e * np.max(scores)
    for paper in range(num_papers):
        # Calculate fraction of reviewers currently occupied (x)
        x = np.sum(last_assign < review_time) / num_reviewers

        # Calculate dynamic threshold using the new potential function
        # Incorporate both reviewer load and maximum similarity score
        max_similarity_score = np.max(scores[paper])
        # print(max_similarity_score)
        # T = min((math.e * max_similarity_score)**(x-1), max_similarity_score)

        T = w_max**(x-1) 
        T = min(max_similarity_score, w_max)

        T = x*max_similarity_score

        # Get scores of available reviewers
        available_scores = np.where(last_assign >= review_time, scores[paper], 0)

        # Get the reviewers that meet the threshold and are available
        thresholded_reviewers = np.where(available_scores > T)[0]

        if len(thresholded_reviewers) >= min_reviewer_per_paper:
            # Select min_reviewer_per_paper reviewers from thresholded_reviewers
            chosen_reviewers = np.random.choice(thresholded_reviewers, min_reviewer_per_paper, replace=False)
            variable_threshold_assign[paper, chosen_reviewers] = 1
        else:
            # Fall back to the Greedy logic
            top_reviewers = np.argsort(-available_scores)[:min_reviewer_per_paper]
            variable_threshold_assign[paper, top_reviewers] = 1 

        # Update last assignment time
        last_assign += 1
        last_assign[variable_threshold_assign[paper].astype(bool)] = 0

    return variable_threshold_assign
