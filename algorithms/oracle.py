import time
import numpy as np
from scipy.sparse import csr_matrix
from cvxopt import matrix, spmatrix, solvers, glpk


def constraint_matrices(scores, review_time, min_reviewer_per_paper):
    """
    Solve the corresponding linear program to compute the paper-reviewer assignments.
    :param scores: np.array, 2d matrix of shape n_papers x n_reviewers, the similarity matrix.
    :param review_time: time window in which reviewer is occupied after assignment.
    :param min_reviewer_per_paper: minimum and maximum reviewers that each paper should be reviewed by.
    :return: Matching. Solve the standard minimization problem using LP formulation.
    """
    num_papers, num_reviewers = scores.shape
    print(f"# papers = {num_papers}, # reviewers = {num_reviewers}")
    d = review_time
    lambd = min_reviewer_per_paper
    c = -np.ravel(scores)

    print("Constructing the sparse constraint matrix:")
    # Adjust the number of constraints to include max reviewers per paper
    num_cons = 2 * num_papers + 3 * num_papers * num_reviewers
    num_vars = num_papers * num_reviewers
    print(f"# Optimization variables: {num_vars}, # Optimization constraints: {num_cons}")

    # Adjust the size for the arrays to accommodate the new constraints
    i_idx = np.zeros((4 + d) * num_papers * num_reviewers, dtype=np.int64)
    j_idx = np.zeros((4 + d) * num_papers * num_reviewers, dtype=np.int64)
    dvals = np.zeros((4 + d) * num_papers * num_reviewers, dtype=np.int8)
    bvals = np.zeros(num_cons, dtype=np.double)

    for k in range((4 + d) * num_papers * num_reviewers):
        if k < num_papers * num_reviewers:
            # Existing constraint: Min reviewers per paper
            p = k // num_reviewers
            r = k % num_reviewers
            i_idx[k], j_idx[k] = p, p * num_reviewers + r
            dvals[k] = -1
            bvals[i_idx[k]] = -lambd
        elif k < 2 * num_papers * num_reviewers:
            # New constraint: Max reviewers per paper
            kprime = k - num_papers * num_reviewers
            p = kprime // num_reviewers
            r = kprime % num_reviewers
            i_idx[k], j_idx[k] = num_papers + p, p * num_reviewers + r
            dvals[k] = 1
            bvals[i_idx[k]] = lambd
        elif k < 3 * num_papers * num_reviewers:
            # Existing constraint: Assignment >= 0
            kprime = k - 2 * num_papers * num_reviewers
            base = 2 * num_papers
            i_idx[k], j_idx[k] = base + kprime, kprime
            dvals[k] = -1
            bvals[i_idx[k]] = 0
        elif k < 4 * num_papers * num_reviewers:
            # Existing constraint: Assignment <= 1
            kprime = k - 3 * num_papers * num_reviewers
            base = 2 * num_papers + num_papers * num_reviewers
            i_idx[k], j_idx[k] = base + kprime, kprime
            dvals[k] = 1
            bvals[i_idx[k]] = 1
        else:
            # Existing constraint: Reviewer workload in time window
            kprime = k - 4 * num_papers * num_reviewers
            t = kprime // (num_reviewers * num_papers)
            p = (kprime % (num_reviewers * num_papers)) // num_reviewers
            r = kprime % num_reviewers
            base = 2 * num_papers + 2 * num_papers * num_reviewers
            
            i_idx[k] = base + p * num_reviewers + r
            j_idx[k] = min(p + t, num_papers-1) * num_reviewers + r 
            dvals[k] = 1 if p+t < num_papers else 0
            bvals[i_idx[k]] = 1

    A = csr_matrix((dvals, (i_idx, j_idx)), shape=(num_cons, num_vars)).tocoo()
    G = spmatrix(A.data.tolist(), A.row.tolist(), A.col.tolist(), size=A.shape)
    obj = matrix(c.reshape(-1, 1))
    b = matrix(bvals.reshape(-1, 1))

    return obj, G, b

def lp(scores, review_time=6, min_reviewer_per_paper=3):
    obj, G, b = constraint_matrices(scores, review_time, min_reviewer_per_paper)
    print("Start solving the LP:")
    start_time = time.time()
    sol = solvers.lp(obj, G, b, solver="glpk")
    #sol = solvers.lp(obj, G, b, solver="mosek")
    end_time = time.time()
    print(f"Time used to solve the LP: {end_time - start_time} seconds.")
    opt_x = np.array(sol["x"]).reshape(scores.shape)
    return opt_x


def ilp(scores, review_time=6, min_reviewer_per_paper=3):
    obj, G, b = constraint_matrices(scores, review_time, min_reviewer_per_paper)
    print("Start solving the ILP:")
    start_time = time.time()
    sol2 = glpk.ilp(obj, G, b, B=set(range(len(obj))))
    end_time = time.time()
    print(f"Time used to solve the ILP: {end_time - start_time} seconds.")
    opt_x2 = np.array(sol2[1]).reshape(scores.shape)
    return opt_x2
