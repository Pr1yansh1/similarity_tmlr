#!/usr/bin/env python
#Alberto Rodriguez Sanchez, 2161801914
#2016
#
# This program decide if one matrix A is totally unimodular iff it follows the next rules
#
# Rule 1: Has only elements 1,0 or -1
# Rule 2: Don't have the next four sub matrices
#      | 1  1 | or |-1  1 | or  | 1 -1 | or | 1 1 |
#      | 1 -1 |    | 1  1 |     | 1  1 |    |-1 1 |
# Rule 3:A will be reduced to one easy to check TU matrix without break previous rules
# this rule is not used now in this program, but if used, reduce the algorithm complexity


import numpy as np
import sys
from itertools import permutations

def checkRule1(A):
    '''check if every element in A is 0,1 or -1'''
    return np.all(np.logical_or.reduce((A==0, A==1, A==-1)))

def checkRule2(A,r,c):
    '''Check if one sub matriz have determinant 0,1 or -1'''
    det=A[r[0]][c[0]]*A[r[1]][c[1]]- A[r[1]][c[0]]*A[r[0]][c[1]]
    return det == 1 or det == -1 or det==0

def theoremNo2det(A):
    '''
       Check Rule 1 for matrix A 
       Generate all 2x2 submatrices of A and check rule 2
    '''
    if checkRule1(A):
        m,n=A.shape
        for r in permutations(range(m), 2):
            for c in permutations(range(n), 2):
                if not checkRule2(A,r,c):
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
    #assert len(sys.argv) > 1, "Usage: " + sys.argv[0] + " matrixFile"
    #read A from file
    #A=np.loadtxt(sys.argv[1])

    A = makeA(7, 3, 4)
    print(A)
    
    if theoremNo2det(A):
        print ('A is a TU Matrix')
