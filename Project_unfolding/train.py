# reference: arXiv:hep-ph/9509307
import numpy as np 
from vector_creation import *

def train(location, xi, first_bin, last_bin, bin_size):
    x, b, A = bin_info(location, first_bin, last_bin, bin_size)

    # size of C depends on x
    nC = len(x)
    C = -np.identity(nC) 
    for i in range(nC):
        if (0 < i < nC-1):
            C[i,i] = -2
        if (i<nC-1):
            C[i,i+1] = 1
        if (i>0):
            C[i,i-1] = 1
    C += xi * np.identity(nC)
    inv_C = np.linalg.inv(C)

    # covariant matrix B for b 
    ### NOT USED
#     cov_B = np.diag(b)
#     Q, r_sqr, QT = np.linalg.svd(cov_B)

    ###    # rescaling A and b  ###
#     ### OLD WAY 
#     ### problematic because SVD yields r in decreasing order
#     Q_rescaled = np.zeros(Q.shape)
#     r = np.sqrt(r_sqr)
#     for i in range(len(Q_rescaled)):
#         Q_rescaled[i,:] = Q[i,:]/r[i]
#     Atilde = Q_rescaled @ A
#     btilde = Q_rescaled @ b

#     ### NEW WAY
#     ### r^2 should be the eigenvalues of the (diagonal) matrix B, so r = np.sqrt(b)
    r = np.sqrt(b)
    Atilde = np.empty(A.shape)
    btilde = np.empty(b.shape)
    for i in range(len(b)):
        Atilde[i,:] = A[i,:]/r[i]
        btilde[i] = b[i]/r[i]
         
    U, s, VT = np.linalg.svd(Atilde @ inv_C)
    S = np.diag(s)

    d = U.T @ btilde
    bin_values = [first_bin, last_bin, bin_size]
    return U, s, VT, bin_values, x, C, d  