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

    
    r = np.sqrt(b)
    Atilde = np.empty(A.shape)
    btilde = np.empty(b.shape)
    for i in range(len(b)):
        Atilde[i,:] = A[i,:]/r[i]
        btilde[i] = b[i]/r[i]

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

    return x_unfolded, x, b 