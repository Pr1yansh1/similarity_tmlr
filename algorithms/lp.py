import time
import numpy as np
from scipy.sparse import csr_matrix
from cvxopt import matrix, spmatrix, solvers

def find_match(S, review_time=6, min_reviewer_per_paper=3):
    """
    Solve the corresponding linear program to compute the paper-reviewer assignments.
    :param S:   np.array, 2d matrix of shape n_papers x n_reviewers, the similarity matrix.
    :param review_time:  # time window in which reviewer is occupied after assingnment
    :param min_reviewer_per_paper:  # reviewers that each paper should be reviewed.
    :return:    Matching. Solve the standard minimization problem using LP formulation.
    """
    (num_papers, num_reviewers) = S.shape
    print(f"# papers = {num_papers}, # reviewers = {num_reviewers}")
    d = review_time
    lambd = min_reviewer_per_paper

    c = np.zeros(num_papers * num_reviewers, dtype=np.double)
    for i in range(num_papers):
        for j in range(num_reviewers):
            c[i * num_reviewers + j] = S[i][j]
    print("Constructing the sparse constraint matrix:")
    num_cons = num_papers + 3 * num_papers * num_reviewers
    num_vars = num_papers * num_reviewers
    # vars: p1r1, p1r2, .., p1rn, ..., pmr1, .. pmrn
    print(f"# Optimization variables: {num_vars}, # Optimization constraints: {num_cons}")
    # Number of non-zero values in the matrix: n * m + 2 * n * m + d * n * m =  (3 + d) * n * m.
    i_idx = np.arange((3 + d) * num_papers * num_reviewers, dtype=np.int64)
    j_idx = np.zeros((3 + d) * num_papers * num_reviewers, dtype=np.int64)
    dvals = np.zeros((3 + d) * num_papers * num_reviewers, dtype=np.int8)
    bvals = np.zeros(num_cons, dtype=np.double)
    for k in range((3 + d) * num_papers * num_reviewers):
        if k < num_papers * num_reviewers:
            # Constraints to ensure that num_reviewers per paper at least lambd.
            # For every p, sum_r -1 * x_{p * num_reviewers + r} <= -lambd
            p = k // num_reviewers
            r = k % num_reviewers
            i_idx[k], j_idx[k] = p, p * num_reviewers + r
            
            dvals[k] = -1
            bvals[i_idx[k]] = -lambd            
        elif k < 2 * num_papers * num_reviewers:
            # Constraints to ensure that >= 0.
            kprime = k - num_papers * num_reviewers
            i_idx[k], j_idx[k] = num_papers + kprime, kprime
            dvals[k] = -1
            bvals[i_idx[k]] = 0
        elif k < 3 * num_papers * num_reviewers:
            # Constraints to ensure that <= 1.
            kprime = k - 2 * num_papers * num_reviewers
            base = num_papers + num_papers * num_reviewers
            i_idx[k], j_idx[k] = kprime + base, kprime
            dvals[k] = 1
            bvals[i_idx[k]] = 1
        else:
            # Constraints to ensure that num_papers per reviewer at most one paper in [t, t+d]
            kprime = k - 3 * num_papers * num_reviewers
            # kprime = t * num_papers * num_reviewers + p * num_reviewers + r
            t = kprime // (num_reviewers * num_papers)
            p = (kprime % (num_reviewers * num_papers)) // num_reviewers
            r = kprime % num_reviewers
            base = num_papers + 2 * num_papers * num_reviewers
            
            # For every r, p, sum_{t = 0}^{d} x_{(p + t) * num_reviewers + r} <= 1
            i_idx[k] = base + p * num_reviewers + r # max(p + t, num_papers -1)
            j_idx[k] = ((p + t) % num_papers) * num_reviewers + r
            dvals[k] = 1
            bvals[i_idx[k]] = 1


    A = csr_matrix((dvals, (i_idx, j_idx)), shape=(num_cons, num_vars)).tocoo()
    G = spmatrix(A.data.tolist(), A.row.tolist(), A.col.tolist(), size=A.shape)
    obj = matrix(c.reshape(-1, 1))
    b = matrix(bvals.reshape(-1, 1))
    print(f"Shape of the constraint matrix: {A.shape}")
    print("Start solving the LP:")
    start_time = time.time()
    sol = solvers.lp(obj, G, b, solver="glpk")
    #sol = solvers.lp(obj, G, b, solver="mosek")
    end_time = time.time()
    print(f"Time used to solve the LP: {end_time - start_time} seconds.")
    opt_x = np.array(sol["x"]).reshape(num_papers, num_reviewers)
    return opt_x

# scores = np.loadtxt("similarity_result.txt")
scores = np.random.rand(16, 8)
print(find_match(scores, review_time = 2))

from scipy.stats import rankdata
print(rankdata(scores, axis=1))

