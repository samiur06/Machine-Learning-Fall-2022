# reference: arXiv:hep-ph/9509307
import numpy as np 
from vector_creation import *

def test(location, fulloutput_train, tau):
    U = fulloutput_train[0] 
    s = fulloutput_train[1]
    VT = fulloutput_train[2]
    bin_values = fulloutput_train[3] 
    xini = fulloutput_train[4] 
    C = fulloutput_train[5]
    first_bin = bin_values[0]
    last_bin = bin_values[1]
    bin_size = bin_values[2]
    inv_C = np.linalg.inv(C)

    x, b, A = bin_info(location, first_bin, last_bin, bin_size)

    cov_B = np.diag(b)
    Q, r_sqr, QT = np.linalg.svd(cov_B)

    Q_rescaled = np.zeros(Q.shape)

    r = np.sqrt(r_sqr)
    for i in range(len(Q_rescaled)):
        Q_rescaled[i,:] = Q[i,:]/r[i]
    
    # Q_rescaled
    btilde = Q_rescaled @ b
    d = U.T @ btilde

    z = np.zeros(len(s))
    for i in range(len(s)):
        z[i] = d[i] * s[i] /(s[i]**2 + tau)

    V = VT.T
    w = inv_C @ V @ z

    bigZ_elements = np.zeros(len(s))
    for i in range(len(s)):
        bigZ_elements[i] = s[i]**2 / (s[i]**2 + tau)**2

    bigZ = np.diag(bigZ_elements)
    bigW = inv_C @ V @ bigZ @ VT @ np.linalg.inv(C.T)

    x_unfolded = np.zeros(len(x))
    for i in range(len(x)):
        x_unfolded[i] = xini[i] * w[i]
    cov_unf_x_theo = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for k in range(len(x)):
            cov_unf_x_theo[i,k] = xini[i] * bigW[i,k] * xini[k]

    return x_unfolded, x 