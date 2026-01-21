# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:55:29 2026

@author: tubi2990
"""

# libraries 
import numpy as np
from scipy.signal import butter, filtfilt


# --- FILTERING 

# high-pass filter 
def HPfilter(data, fs, cutoff=50, order=6):
    """
    HPFILTER  Apply zero-phase high-pass filtering to continuous data

    Parameters
    ----------
    data : ndarray
        Input signal, shape (samples,) or (channels, samples)
    fs : float
        Sampling rate in Hz
    cutoff : float, optional
        High-pass cutoff frequency in Hz (default = 50 Hz)
    order : int, optional
        Filter order of the Butterworth filter (default = 6)

    Returns
    -------
    data_filt : ndarray
        High-pass filtered signal with the same shape as input
    """
    
    b, a = butter(order, cutoff/(fs/2), btype='high')
    return filtfilt(b, a, data)



# low-pass filter
def LPfilter(data, fs, cutoff=10000, order=6):
    """
    LPFILTER  Apply zero-phase low-pass filtering to continuous data

    Parameters
    ----------
    data : ndarray
        Input signal, shape (samples,) or (channels, samples)
    fs : float
        Sampling rate in Hz
    cutoff : float, optional
        Low-pass cutoff frequency in Hz (default = 10000 Hz)
    order : int, optional
        Filter order of the Butterworth filter (default = 6)

    Returns
    -------
    data_filt : ndarray
        Low-pass filtered signal with the same shape as input

    """
    b, a = butter(order, cutoff/(fs/2), btype='low')
    return filtfilt(b, a, data)


# --- DATA EPOCHING 
def epochaudio(data, t_data, t_markers, fs, epoch_window):
    """
    Epoch continuous audio data based on marker timestamps

    Parameters
    ----------
    data : ndarray
        Shape (channels, samples) or (samples,)
    t_data : ndarray
        Time stamps of data (seconds), shape (samples,)
    t_markers : ndarray
        Marker time stamps (seconds), shape (trials,)
    fs : float
        Sampling rate (Hz)
    epoch_window : tuple or list
        (start, end) in seconds relative to marker, e.g. (0, 0.5)

    Returns
    -------
    epochs : ndarray
        Shape (channels, time, trials)
    t_epoch : ndarray
        Time vector for one epoch (seconds)
    """

    # ensure 1D arrays
    t_data = np.asarray(t_data).ravel()
    t_markers = np.asarray(t_markers).ravel()

    # samples per epoch
    nsamples = int(round((epoch_window[1] - epoch_window[0]) * fs))
    t_epoch = np.linspace(epoch_window[0], epoch_window[1], nsamples)

    # ensure data is 2D: channels × samples
    data = np.asarray(data)
    if data.ndim == 1:
        data = data[np.newaxis, :]  # force row

    nchannels, nsamp_data = data.shape
    ntrials = len(t_markers)

    # preallocate epochs with NaNs (MATLAB-like)
    epochs = np.full((nchannels, nsamples, ntrials), np.nan)

    for tr in range(ntrials):
        # epoch start & end time
        t_start = t_markers[tr] + epoch_window[0]

        # find first index where t_data >= t_start
        idx_start = np.searchsorted(t_data, t_start, side='left')
        idx_end = idx_start + nsamples

        # boundary check
        if idx_start >= nsamp_data or idx_end > nsamp_data:
            continue

        # extract epoch
        epochs[:, :, tr] = data[:, idx_start:idx_end]

    return epochs, t_epoch

