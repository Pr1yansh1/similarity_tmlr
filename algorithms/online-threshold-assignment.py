import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--DURATION', type=int, help='Duration value', default = 2)
parser.add_argument('--R_TARGET', type=int, help='R target value', default = 2)

args = parser.parse_args()
# Load similarity matrix from file
similarity_matrix = np.loadtxt('similarity_result.txt')

#define the papers dict which maps papers to timestamps 
papers = {} 

with open('paper_dict.txt', 'r') as file:
    contents = file.read()
    papers = eval(contents)

# Define constants
NUM_REVIEWERS = similarity_matrix.shape[0]
DURATION = args.DURATION
R_TARGET = args.R_TARGET
INITIAL_THRESHOLD = 10000

# Initialize reviewer availability
reviewer_availability = np.zeros(NUM_REVIEWERS, dtype=int)

# Initialize threshold
threshold = INITIAL_THRESHOLD

def assign_paper(paper_index, timestep):
    global threshold
    available_reviewers = np.where(reviewer_availability <= timestep)[0]
    if len(available_reviewers) == 0:
        return None
    similarity_scores = similarity_matrix[available_reviewers, paper_index]
    sorted_reviewers = available_reviewers
    has_assigned = False 
    for i in range(len(available_reviewers) - R_TARGET + 1):
        if np.sum(similarity_scores[available_reviewers[i:i+R_TARGET]]) >= threshold:
            print("here")
            assigned_reviewers = available_reviewers[i:i+R_TARGET]
            has_assigned = True
            break
    if not has_assigned:
        sorted_reviewers = available_reviewers[np.argsort(-similarity_scores)]
        assigned_reviewers = sorted_reviewers[:R_TARGET]
        threshold = np.sum(similarity_scores[assigned_reviewers])
    reviewer_availability[assigned_reviewers] = timestep + DURATION
    return assigned_reviewers.tolist()

import random

def assign_paper_randomized(paper_index, timestep):
    global threshold
    available_reviewers = np.where(reviewer_availability <= timestep)[0]
    qualified_reviewers = []
    for reviewer in available_reviewers:
        similarity_score = similarity_matrix[reviewer, paper_index]
        if similarity_score >= threshold:
            qualified_reviewers.append(reviewer)
    if len(qualified_reviewers) == 0:
        similarity_scores = similarity_matrix[available_reviewers, paper_index]
        sorted_reviewers = available_reviewers[np.argsort(-similarity_scores)]
        assigned_reviewers = sorted_reviewers[:R_TARGET]
        threshold = np.sum(similarity_scores[assigned_reviewers])
    random.shuffle(qualified_reviewers)
    selected_reviewers = qualified_reviewers[:R_TARGET]
    reviewer_availability[selected_reviewers] = timestep + DURATION
    return selected_reviewers


def assign_papers(papers):
    sorted_papers = sorted(papers.items(), key=lambda x: x[1])
    paper_assignments = {}
    current_timestep = 0
    current_paper_index = 0
    while current_paper_index < len(sorted_papers):
        paper_index, timestep = sorted_papers[current_paper_index]
        if timestep > current_timestep:
            current_timestep = timestep
        else:
            assigned_reviewers = assign_paper(paper_index, current_timestep)
            if assigned_reviewers is not None:
                paper_assignments[paper_index] = assigned_reviewers
                current_paper_index += 1
            else:
                print(f"No available reviewers for paper {paper_index}")
        # Increment timestep after each paper assignment
        current_timestep += 1
    return paper_assignments

def tpms_metric(assignment_dict): 
    total_similarity_sum = 0
    for paper_idx, reviewer_list in assignment_dict.items():
        for reviewer_idx in reviewer_list:
            total_similarity_sum += similarity_matrix[reviewer_idx][paper_idx]
    
    return total_similarity_sum

def fairness_metric(assignment_dict): 
    worst_off_paper = None
    worst_off_sum = float('inf') 
    for paper_idx, reviewer_list in assignment_dict.items():
        paper_sum = 0
        for reviewer_idx in reviewer_list:
            s = similarity_matrix[reviewer_idx][paper_idx]
            paper_sum += s
        if paper_sum < worst_off_sum:
            worst_off_paper = paper_idx
            worst_off_sum = paper_sum

    return (worst_off_paper, worst_off_sum)

assignments = assign_papers(papers)
print(f"Online Threshold Assignment result for duration:  {DURATION} , r_target: {R_TARGET}")
print(f"TPMS total sum similarity: {tpms_metric(assignments)}") 
print(f"Worst off paper sim sum: {fairness_metric(assignments)}")
