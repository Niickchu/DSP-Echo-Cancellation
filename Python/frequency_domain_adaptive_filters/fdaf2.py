import numpy as np
from scipy.signal import cheby2, lfilter

# Frequency Domain Adaptive Filter (FDAF) implementation
def fdaf2(x, d, mu, filter_length, del_param, lam):
    W = np.zeros(filter_length)
    P = del_param * np.ones(filter_length)
    y = np.zeros(len(x))
    e = np.zeros(len(x))
    
    for i in range(0, len(x), filter_length):
        X = np.fft.fft(x[i:i + filter_length])
        D = np.fft.fft(d[i:i + filter_length])
        Y = np.fft.fft(W) * X
        y[i:i + filter_length] = np.fft.ifft(Y).real
        e[i:i + filter_length] = d[i:i + filter_length] - y[i:i + filter_length]
        E = np.fft.fft(e[i:i + filter_length])
        P = lam * P + (1 - lam) * np.abs(X)**2
        W += mu * np.fft.ifft(E * np.conj(X) / P).real
    
    return y, e