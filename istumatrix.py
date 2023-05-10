# Check if the Oracle LP's constraint matrix is totally unimodular

import numpy as np
from itertools import combinations

def is_totally_unimodular(A):
    '''
        Check if A is TU. i.e. if any square submatrix has determinant not in {-1, 0, 1}
    '''
    m, n = A.shape
    if m > n:
        return is_totally_unimodular(A.T)
    
    for i in range(1, m+1):
        for subrows in combinations(A, i):
            for subcols in combinations(np.array(subrows).T, i):
                subsqmatrix = np.array(subcols)
                if np.linalg.det(subsqmatrix) not in {-1, 0, 1}:
                    print(subsqmatrix.T)
                    return False

    return True
                

def makeA(p, r, d):
    '''
        Makes LP constraint matrix whose TUness is sufficient to check
        for parameters p = # papers, r = # reviewers, d = review time
        for variables ordered [x11 .. x1p x21 .. x2p ... xr1 .. xrp]
    '''
    Ap = np.zeros((p-d+1, p))
    for t in range(p-d+1):
        Ap[t, t:t+d] = 1

    I_rowblocks = np.block([np.eye(p)]*r)
    A_diagblocks = np.kron(np.eye(r, dtype=int), Ap)
    return np.block([[I_rowblocks], [A_diagblocks]])

if __name__ == '__main__':
    A = makeA(4, 3, 3)
    #print(A)
    print(is_totally_unimodular(A))
