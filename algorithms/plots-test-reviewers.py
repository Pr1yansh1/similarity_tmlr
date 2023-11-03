import numpy as np
import greedy
import online

def obj_score(scores, assign):    
    return np.sum(scores * assign)

sim_scores = np.loadtxt('similarity_result.txt')
d, lambd = 2, 2
factor_to_compare = 0.00000001  # Factor value for Rank Greedy to compare with Greedy

aggregate_results = {
    "greedy_avg_scores": [],
    "rank_greedy_avg_scores": [],
    "same_assignments": [],
    "different_assignments": [],
    "one_reviewer_diff": [],
    "both_reviewers_diff": [],
    "avg_score_difference": [],
    "greedy_total_scores": [],
    "rank_greedy_total_scores": []
}

for trial in range(1, 27):
    scores = sim_scores[trial*15 : (trial*15) + 15, :].T

    print(f"\nFor Trial {trial}:")
    print("-" * 40)
    
    greedy_assign = greedy.assign(scores, review_time=d, min_reviewer_per_paper=lambd)
    rank_greedy_result = online.rank_greedy_assign(scores, review_time=d, min_reviewer_per_paper=lambd, factor=factor_to_compare)

    same_assignments = 0
    different_assignments = 0
    one_reviewer_diff = 0
    both_reviewers_diff = 0
    score_difference = []
    
    for paper_id in range(scores.shape[0]):
        greedy_reviewers = np.where(greedy_assign[paper_id] == 1)[0]
        rank_greedy_reviewers = np.where(rank_greedy_result[paper_id] == 1)[0]
        
        if set(greedy_reviewers) == set(rank_greedy_reviewers):
            same_assignments += 1
        else:
            different_assignments += 1
            if len(set(greedy_reviewers).intersection(rank_greedy_reviewers)) == 1:
                one_reviewer_diff += 1
            else:
                both_reviewers_diff += 1
                
            score_difference.append(np.sum(scores[paper_id, rank_greedy_reviewers]) - np.sum(scores[paper_id, greedy_reviewers]))

    avg_score_difference = np.mean(score_difference) if score_difference else 0
    greedy_scores_for_papers = [np.sum(scores[paper_id, :][greedy_assign[paper_id, :] == 1]) for paper_id in range(scores.shape[0])]
    rank_greedy_scores_for_papers = [np.sum(scores[paper_id, :][rank_greedy_result[paper_id, :] == 1]) for paper_id in range(scores.shape[0])]
    
    aggregate_results["greedy_avg_scores"].append(np.mean(greedy_scores_for_papers))
    aggregate_results["rank_greedy_avg_scores"].append(np.mean(rank_greedy_scores_for_papers))
    aggregate_results["same_assignments"].append(same_assignments)
    aggregate_results["different_assignments"].append(different_assignments)
    aggregate_results["one_reviewer_diff"].append(one_reviewer_diff)
    aggregate_results["both_reviewers_diff"].append(both_reviewers_diff)
    aggregate_results["avg_score_difference"].append(avg_score_difference)
    aggregate_results["greedy_total_scores"].append(obj_score(scores, greedy_assign))
    aggregate_results["rank_greedy_total_scores"].append(obj_score(scores, rank_greedy_result))

    # Display stats for this trial
    print(f"\nGreedy:")
    print(f"Average similarity score for each paper: {np.mean(greedy_scores_for_papers):.4f}")
    print(f"Maximum similarity score for a paper: {max(greedy_scores_for_papers):.4f}")
    print(f"75th percentile similarity score for a paper: {np.percentile(greedy_scores_for_papers, 75):.4f}")
    
    print(f"\nRank Greedy:")
    print(f"Average similarity score for each paper: {np.mean(rank_greedy_scores_for_papers):.4f}")
    print(f"Maximum similarity score for a paper: {max(rank_greedy_scores_for_papers):.4f}")
    print(f"75th percentile similarity score for a paper: {np.percentile(rank_greedy_scores_for_papers, 75):.4f}")

    print(f"\nComparison:")
    print(f"Same assignments: {same_assignments}")
    print(f"Different assignments: {different_assignments}")
    print(f"One reviewer different: {one_reviewer_diff}")
    print(f"Both reviewers different: {both_reviewers_diff}")
    print(f"Average score difference for papers with different assignments: {avg_score_difference:.4f}")

    greedy_score = obj_score(scores, greedy_assign)
    rank_greedy_score = obj_score(scores, rank_greedy_result)
    print(f"\nTotal similarity score for Greedy: {greedy_score}")
    print(f"Total similarity score for Rank Greedy: {rank_greedy_score}")

    if rank_greedy_score > greedy_score:
        print("Observation: Rank Greedy is doing better in this trial.")
    else:
        print("Observation: Greedy is performing as good or better than Rank Greedy in this trial.")

# Display aggregate results over all trials
print("\nAggregate Results Over All Trials:")
print("-" * 40)
print(f"Average similarity score for each paper (Greedy): {np.mean(aggregate_results['greedy_avg_scores']):.4f}")
print(f"Average similarity score for each paper (Rank Greedy): {np.mean(aggregate_results['rank_greedy_avg_scores']):.4f}")
print(f"Average same assignments over all trials: {np.mean(aggregate_results['same_assignments']):.2f}")
print(f"Average different assignments over all trials: {np.mean(aggregate_results['different_assignments']):.2f}")
print(f"Average one reviewer different over all trials: {np.mean(aggregate_results['one_reviewer_diff']):.2f}")
print(f"Average both reviewers different over all trials: {np.mean(aggregate_results['both_reviewers_diff']):.2f}")
print(f"Average score difference for papers with different assignments over all trials: {np.mean(aggregate_results['avg_score_difference']):.4f}")
print(f"Average total similarity score for Greedy over all trials: {np.mean(aggregate_results['greedy_total_scores']):.4f}")
print(f"Average total similarity score for Rank Greedy over all trials: {np.mean(aggregate_results['rank_greedy_total_scores']):.4f}")
