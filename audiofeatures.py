# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:10:08 2026

@author: tubi2990
"""

import numpy as np

def computeTDOA(epochs, fs, max_tau=0.001):
    """
    epochs: shape (2, samples, trials)
    fs: sampling rate
    max_tau: maximum expected delay in seconds
    """
    tdoaFeat = []

    n_samples = epochs.shape[1]
    max_shift = int(fs * max_tau)

    for i in range(epochs.shape[2]):
        L = epochs[0, :, i]
        R = epochs[1, :, i]

        # FFT
        SIG_L = np.fft.rfft(L, n=2*n_samples)
        SIG_R = np.fft.rfft(R, n=2*n_samples)

        # GCC-PHAT
        R_cross = SIG_L * np.conj(SIG_R)
        R_cross /= np.abs(R_cross) + 1e-15

        cc = np.fft.irfft(R_cross)

        # keep plausible lags
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

        shift = np.argmax(cc) - max_shift
        tau = shift / fs

        tdoaFeat.append(tau)

    return np.array(tdoaFeat)
        
        
def computeILD(epochs, eps=1e-12):
    """
    epochs: shape (2, samples, trials)
    returns: ILD per trial in dB
    """
    ildFeat = []

    for i in range(epochs.shape[2]):
        L = epochs[0, :, i]
        R = epochs[1, :, i]

        rms_L = np.sqrt(np.mean(L**2))
        rms_R = np.sqrt(np.mean(R**2))

        ild = 20 * np.log10((rms_L + eps) / (rms_R + eps))
        ildFeat.append(ild)

    return np.array(ildFeat)