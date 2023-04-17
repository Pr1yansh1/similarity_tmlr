import numpy as np
from scipy.sparse import csr_matrix
from cvxopt import matrix, solvers
INF = 1e5

scores = np.loadtxt("similarity_result.txt")
similarity_matrix = scores #["similarity_matrix"]
# mask_matrix = scores["mask_matrix"]

review_time = 5
reviewers, papers = similarity_matrix.shape
last_assigned = np.full(reviewers, INF) # time since last assigned
assign = np.zeros(papers)

print("Computing greedy assignments..")

for paper in range(papers):

    max_simscore = 0
    max_reviewer = None
    for reviewer in range(reviewers):
        if last_assigned[reviewer] >= review_time and \
                similarity_matrix[reviewer][paper] >= max_simscore:
            max_simscore = similarity_matrix[reviewer][paper]
            max_reviewer = reviewer

    if max_reviewer == None:
        print("Didn't assign ", paper, assign)
    else:    
        assign[paper] = max_reviewer
    last_assigned += 1
    last_assigned[reviewer] = 0

print(assign.shape, assign[:10])
assignment_scores = [similarity_matrix[int(assign[paper]), paper] for paper in range(papers)]

print("Greedy assignment scores", min(assignment_scores), sum(assignment_scores))


print("Computing offline oracle assignments..")

# For every r, t, \sum_{p=t}^{t+d} x_{p,r} \leq 1
A1_shape = (reviewers * (papers-review_time+1), reviewers * papers)
A1_rows, A1_cols = [], []

for r in range(reviewers):
    for t in range(papers-review_time +1):
        A1_rows.append([r * t] * review_time)
        A1_cols.append(list(range(r * t, r * (t + review_time))))

data = np.ones(len(A1_rows))
A1 = csr_matrix((data, (A1_rows, A1_cols)), A1_shape)

#A1 = np.zeros((reviewers * (papers-review_time+1), reviewers * papers))
#for r in range(reviewers):
    #for t in range(papers-review_time+1):
        #A1[r * t][r * t: r * (t+review_time)] = 1
b1 = np.ones((reviewers * (papers-review_time)))

# For every p, \sum_r x_{p, r} = 3
A2 = np.zeros((papers, reviewers * papers))
for p in range(papers):
    A2[p][::papers] = 1

# For every p, r, 0 \leq x_{p,r} \leq 1
A3 = np.eye(papers * reviewers)
b3 = np.ones(papers * reviewers)
A4 = -A3
b4 = np.zeros(papers * reviewers)

A = 0

print("Plotting something..")
import matplotlib.pyplot as plt
    
plt.hist(assignment_scores, bins=20)
plt.show()
