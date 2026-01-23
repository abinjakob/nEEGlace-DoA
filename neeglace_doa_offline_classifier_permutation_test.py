# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 13:37:06 2026

nEEGlace Audio DoA Offline Classifier - Permutation Test
--------------------------------------------------------

A permutation test is carried to check the validity of the DoA Model 
The model is run 1000 times with shuffled data and a histogram is plotted

Note: Only considered the ILD features for the permutation test as the model
      remains the same for all the features. 

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

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audioprocessing import HPfilter, LPfilter, epochaudio
from audiofeatures import computeILD, computeTDOA

from sklearn.utils import shuffle

#%% load audio and marker data 

# data path
rootpath = r"L:\Cloud\T-PsyOL\PhD Project\nEEGlace\sourceDoA\Data"
foldername = "Piloting-22012026"
filename = "sub-P001_ses-S001_task-task-nearneeglace90sourceclose_run-001_eeg.xdf"

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

# clean data points
audio[~np.isfinite(audio)] = 0
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

# seperate left and right audio trials 
trialsL, trialsR = [], []
for i, m in enumerate(markers):
    if m[0] == '1':
        trialsL.append(epochs[:, :, i])
    elif m[0] == '2':
        trialsR.append(epochs[:, :, i])

trialsL = np.stack(trialsL, axis=2)
trialsR = np.stack(trialsR, axis=2)


#%% extracting audio featrures for DoA classification 

# ILD features (broad band)
featILD = computeILD(epochs)

# creating labels     
labels = np.array([0 if m[0]=='1' else 1 for m in markers])


#%% prepare data for classification

# feature vector for ILD only features
X_ILD = featILD.reshape(-1, 1)

# label vector
y = labels 

#%% PERMUTATION TEST

# number of test runs 
testTrials = 1000
PTacc = []

# model performance observed (with real data) 
modelAcc = 1

for iRun in range(testTrials):
    
    # shuffling labels 
    y = shuffle(y)

    # split the dataset into trainning and testing set
    X_train_ILD, X_test_ILD, y_train, y_test = train_test_split(X_ILD, y, test_size=0.3, random_state=42)
    
    
    # define a pipeline with preprocessing (scaling) and SVM classifier
    pipeline = make_pipeline(StandardScaler(), SVC(probability=True))
    
    # parameter grid for SVM
    param_grid = {
        'svc__C': [0.1, 1, 10, 100],  # SVM regularization parameter
        'svc__gamma': [0.001, 0.01, 0.1, 1],  # Kernel coefficient for 'rbf'
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid']  # Kernel type
    }
    
    # apply cros-validaion on training set to find best SVM parameters
    clf_ILD = GridSearchCV(pipeline, param_grid, cv=5)
    
    # train the models
    clf_ILD.fit(X_train_ILD, y_train)
    
    # make predictions 
    y_pred_ILD = clf_ILD.predict(X_test_ILD)
    
    # calculate model performance
    # accuracy
    accuracy = accuracy_score(y_test, y_pred_ILD)
    PTacc.append(accuracy)
    
    # reporting after every 50 trials
    if iRun % 50 == 0:
        print(f'Running the permutation trial: {iRun} to {iRun+50}')


#%% plot histogram

# calculating the emperical chance levels 
totalSamples = len(labels)
classCount = np.unique(labels, return_counts=True)[1]
chanceLevel = np.max(classCount) / totalSamples

plt.figure(figsize=(10,8))
# plot accuracy
plt.hist(PTacc, bins=40, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
# Add a vertical line for the observed accuracy
plt.axvline(x=modelAcc, color='red', linestyle='dashed', linewidth=1, label='model accuracy')
plt.axvline(x=chanceLevel, color='magenta', linestyle='dashed', linewidth=1, label='emp chance level')  
plt.xlim([0,1])  
plt.title('Accuracy')
plt.legend()





