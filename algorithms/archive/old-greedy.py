# parameters
min_reviewer_per_paper = 3
max_paper_per_reviewer = 6 


import heapq

def online_greedy(stream, similarity_matrix, conflicts, k):
    """
    Match arriving papers to the set of reviewers with the maximum similarity
    according to a similarity matrix, using an online greedy algorithm.
    Conflicting reviewers will not be assigned to their respective papers.

    :param stream: an iterator that yields the arriving papers.
    :param similarity_matrix: a matrix that represents the similarity between buyers and goods, 
    similarity_matrix[i][j] represents the similarity score between paper i and reviewer j
    :param conflicts: a dictionary that maps each paper to a list of conflicting reviewers
    :param k: the maximum number of reviewers to match each buyer with
    :return: a dictionary that maps each paper to a list of k non-conflicting reviewers
    """

    papers = []
    reviewers = set(similarity_matrix.keys())
    matches = {}

    for paper in stream: 

        # If there are no more reviewers left 
        if not reviewers:
            break

        # Remove the conflicting reviewers from the set of available reviewers
        available_reviewers = reviewers.difference(conflicts[paper])

        # Compute the similarity between the buyer and each available good
        similarities = [(similarity_matrix[paper][r], r) for r in available_reviewers]

        # Select the k reviewers with the maximum similarity
        max_similarities = heapq.nlargest(k, similarities)

        # Update the matches dictionary with the selected goods
        matches[paper] = [good for (similarity, good) in max_similarities]

        # Remove the selected goods from the list of available goods
        reviewers.difference_update(matches[paper])

        # Add the current buyer to the list of processed buyers
        papers.append(paper)

    for _ in reviewers:
        matches[None] = []

    return matches
