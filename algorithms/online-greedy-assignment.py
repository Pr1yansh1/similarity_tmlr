import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--DURATION', type=int, help='Duration value', default = 2)
parser.add_argument('--R_TARGET', type=int, help='R target value', default = 2)

args = parser.parse_args()
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

# Initialize reviewer availability
reviewer_availability = np.zeros(NUM_REVIEWERS, dtype=int)

def assign_paper(paper_index, timestep):
    # Find the reviewers that are available at the current time step
    available_reviewers = np.where(reviewer_availability <= timestep)[0]
    # If there are no available reviewers, return None
    if len(available_reviewers) == 0:
        return None
    # Find the similarity scores for the available reviewers
    similarity_scores = similarity_matrix[available_reviewers, paper_index]
    # Sort the available reviewers by their similarity scores in descending order
    sorted_reviewers = available_reviewers[np.argsort(-similarity_scores)]
    # Assign the paper to the R_target reviewers with the highest similarity scores
    assigned_reviewers = sorted_reviewers[:R_TARGET]
    # Mark the assigned reviewers as unavailable for the next DURATION time steps
    reviewer_availability[assigned_reviewers] = timestep + DURATION
    return assigned_reviewers.tolist()

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
            if len(assigned_reviewers) > 0:
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
print(f"Online Greedy Assignment result for duration:  {DURATION} , r_target: {R_TARGET}")
print(f"TPMS total sum similarity: {tpms_metric(assignments)}") 
print(f"Worst off paper sim sum: {fairness_metric(assignments)}")
