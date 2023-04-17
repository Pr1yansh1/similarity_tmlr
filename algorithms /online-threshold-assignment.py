import numpy as np

# Load similarity matrix from file
similarity_matrix = np.loadtxt('similarity_result.txt')

#define the papers dict which maps papers to timestamps 

papers = {}
# Define constants
NUM_REVIEWERS = similarity_matrix.shape[0]
DURATION = 2
R_TARGET = 2
INITIAL_THRESHOLD = 0.5

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

    sorted_reviewers = available_reviewers[np.argsort(-similarity_scores)]
    for i in range(len(sorted_reviewers) - R_TARGET + 1):
        if np.sum(similarity_scores[sorted_reviewers[i:i+R_TARGET]]) >= threshold:
            assigned_reviewers = sorted_reviewers[i:i+R_TARGET]
            break
    else: 
        assigned_reviewers = sorted_reviewers[:R_TARGET]
        threshold = np.sum(similarity_scores[assigned_reviewers])
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
            if assigned_reviewers is not None:
                paper_assignments[paper_index] = assigned_reviewers
                current_paper_index += 1
            else:
                print(f"No available reviewers for paper {paper_index}")
        # Increment timestep after each paper assignment
        current_timestep += 1
    return paper_assignments



assignments = assign_papers(papers)


