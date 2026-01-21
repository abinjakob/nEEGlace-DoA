# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:42:17 2026

nEEGlace Audio DoA Offline Classifier 
-------------------------------------



@author: Abin Jacob
         Translational Psychology Lab
         Carl von Ossietzky University Oldenburg
         abin.jacob@uni-oldenburg.de
         
"""

#%% libraries

import os
import numpy as np
import pyxdf
import matplotlib.pyplot as plt
import sounddevice as sd

from audioprocessing import HPfilter, LPfilter, epochaudio

#%% load audio and marker data 

# data path
rootpath = r"L:\Cloud\T-PsyOL\PhD Project\nEEGlace\sourceDoA\Data"
foldername = "Piloting-16012026"
filename = "sub-P001_ses-S001_task-far180_run-001_eeg.xdf"

# load all available streams
streams, header = pyxdf.load_xdf(os.path.join(rootpath, foldername, filename))

# audio stream
audio = streams[1]["time_series"].T    
t_audio = streams[1]["time_stamps"]
fs = float(streams[1]["info"]["nominal_srate"][0])

# marker stream
markers = streams[0]["time_series"]
t_markers = streams[0]["time_stamps"]


#%% audio data pre-processing and epoching 

# remove NaNs and Infs
audio[~np.isfinite(audio)] = 0
# normalize 
audio /= np.max(np.abs(audio))

# high-pass filtering 
for ch in range(audio.shape[0]):
    audio[ch, :] = HPfilter(audio[ch, :], fs)
# low-pass filtering 
for ch in range(audio.shape[0]):
    audio[ch, :] = LPfilter(audio[ch, :], fs)
    
# epocing 
epoch_window = (0, 0.85)
epochs, t_epoch = epochaudio(
    data=audio,
    t_data=t_audio,
    t_markers=t_markers,
    fs=fs,
    epoch_window=epoch_window
)

#%% plotting some raw audio with markers 

# duration to plot (sec)
duration2plot = 100 

t0 = t_audio[0]
idx = (t_audio - t0) <= duration2plot

plt.figure()
plt.plot(t_audio[idx] - t0, audio[0, idx], 'r', label='Ch 1')
plt.plot(t_audio[idx] - t0, audio[1, idx], 'b', label='Ch 2')

for m, tm in zip(markers, t_markers):
    if tm - t0 <= duration2plot:
        if m[0] == '1':
            plt.axvline(tm - t0, color='g', linestyle='--')
        elif m[0] == '2':
            plt.axvline(tm - t0, color='c', linestyle='--')

plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")
plt.title(f'{duration2plot} sec Raw Audio with Trial Markers')
plt.legend()
plt.show()

#%% play audio file

sd.play(audio.T, fs)
# sd.stop()

#%% seperate left and right audio trials 

trialsL, trialsR = [],[]
for i, m in enumerate(markers):
    if m[0] == '1':
        trialsL.append(epochs[:, :, i])
    elif m[0] == '2':
        trialsR.append(epochs[:, :, i])

trialsL = np.stack(trialsL, axis=2)
trialsR = np.stack(trialsR, axis=2)

#%% plotting average trails for each condition

plt.figure()

# left trials
plt.subplot(2, 1, 1)
plt.plot(t_epoch, np.mean(trialsL[0, :, :], axis=1), 'r')
plt.plot(t_epoch, np.mean(trialsL[1, :, :], axis=1), 'b')
plt.title("Left Trials")
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")

# right trials
plt.subplot(2, 1, 2)
plt.plot(t_epoch, np.mean(trialsR[1, :, :], axis=1), 'b')
plt.plot(t_epoch, np.mean(trialsR[0, :, :], axis=1), 'r')
plt.title("Right Trials")
plt.xlabel("Time [sec]")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

