import numpy as np
from scipy.fft import fft, ifft

def ch2(x,y,Lhat):
    # length of x
    Nx = len(x)

    ## Efficient implementation using a convolution
    hhat = np.convolve(x[::-1], y)
    hhat = hhat[Nx-1:]  # skip first Nx-1 entries
    hhat = hhat[:Lhat] # truncate to length Lhat
    # scaling: hhat to h
    alpha = sum([a**2 for a in x])  # scaling
    h = 1/alpha * hhat
    return h

def ch3(x,y,Lhat,epsi):
    # Zero padding
    x = np.pad(x, (0, len(y) - len(x)))
    # FFTs to find X[k] and Y[k]
    Xk = fft(x, len(x)+len(y)-1)
    Yk = fft(y, len(x)+len(y)-1)
    # Computation of H[k]
    # Set H[k]=0 where |X[k]| is smaller than a threshold
    # Fix any runtime warnings with a placeholder 1
    mask = np.abs(Xk) > epsi
    denom = np.where(mask, Xk, 1)
    Hk = np.where(mask, Yk / denom, 0)
    # IFFT to find h[n]
    h = ifft(Hk)
    # Truncation to length Lhat (optional and actually not recommended before you inspect the entire h)
    h = h[:Lhat]
    return np.array(h)