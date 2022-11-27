# reference: arXiv:hep-ph/9509307
# Vector x, b, and matrix A are created from tabular data (txt file) in this function

# Enter the location for a txt file of 2 column table 
# true and measured values of a quantity, respectively

import numpy as np

def bin_info(location, 
             first_bin, 
             last_bin,
             bin_size,
            ):
    
    #true and measured
    true_train = np.loadtxt(location)[:,0]
    meas_train = np.loadtxt(location)[:,1]
    
#     The size of the table should remain the same, 
#     if not the function will give an error 

    bin_width = (last_bin - first_bin)/(bin_size - 1)
    bin_full = np.arange(first_bin, last_bin + bin_width, bin_width)
    
    x = np.zeros(bin_size)
    b = np.zeros(bin_size)
    A = np.zeros((bin_size, bin_size))

    for k in range(len(true_train)): 
        true_val = true_train[k]
        meas_val = meas_train[k]
#         if greater than max(bin_full) value (BEYOND the range), 
#         put it in the last bin
        if (true_val>=max(bin_full)):
            true_val = (bin_full[-1] + bin_full[-2])/2
        if (meas_val>=max(bin_full)):
            meas_val = (bin_full[-1] + bin_full[-2])/2
#         if falls behind the first edge of the first bin (BELOW the range), 
#         put it in the first bin
        if (true_val<=min(bin_full)-bin_width):
            true_val = (bin_full[-1] + bin_full[-2])/2
        if (meas_val<=min(bin_full)-bin_width):
            meas_val = (bin_full[-1] + bin_full[-2])/2
            
#         find the bin number and add counts to vectors x, b and matrix A
        j = np.digitize(true_val, bin_full)
        x[j] +=1
        i = np.digitize(meas_val, bin_full)
        b[i] +=1
        A[i,j] +=1
    
    return x, b, A 
