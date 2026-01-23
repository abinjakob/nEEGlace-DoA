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

from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audioprocessing import HPfilter, LPfilter, epochaudio
from audiofeatures import computeILD, computeTDOA

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

#%% play audio file

sd.play(audio.T, fs)
# sd.stop()

#%% plotting some audio with markers 

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

#%% extracting audio featrures for DoA classification 

# ILD features (broad band)
featILD = computeILD(epochs)

# TDOA GCC-PHAT features 
featTDOA = computeTDOA(epochs, fs)

# creating labels     
labels = np.array([0 if m[0]=='1' else 1 for m in markers])

#%% plotting the features

plt.figure()
# plotting ILD features
plt.subplot(2,1,1)
plt.hist(featILD[labels == 0], bins=30, alpha=0.6, label='Left')
plt.hist(featILD[labels == 1], bins=30, alpha=0.6, label='Right')
plt.legend()
plt.title('ILD Features')

# plotting TDOA features
plt.subplot(2,1,2)
plt.hist(featTDOA[labels == 0], bins=30, alpha=0.6, label='Left')
plt.hist(featTDOA[labels == 1], bins=30, alpha=0.6, label='Right')
plt.legend()
plt.title('TDOA Features')
plt.tight_layout()

plt.figure()
plt.scatter(featILD[labels == 0], featTDOA[labels == 0], label='Left')
plt.scatter(featILD[labels == 1], featTDOA[labels == 1], label='Right')
plt.legend()
plt.title('Feature Space')

#%% prepare data for classification

# feature vector for ILD only features
X_ILD = featILD.reshape(-1, 1)

# feature vector for TDOA only features
X_TDOA = featTDOA.reshape(-1, 1) 

# feature vector for both 
X_both = np.column_stack([featILD, featTDOA])

# label vector
y = labels 

#%% classification using SVM Classifier

# split the dataset into trainning and testing set
X_dummy, X_dummy_test, y_train, y_test, train_idx, test_idx = train_test_split(X_ILD, y, np.arange(len(y)), test_size=0.3, random_state=42, stratify=y)

X_train_ILD = X_ILD[train_idx]
X_test_ILD  = X_ILD[test_idx]
X_train_TDOA = X_TDOA[train_idx]
X_test_TDOA  = X_TDOA[test_idx]
X_train_both = X_both[train_idx]
X_test_both  = X_both[test_idx]

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
clf_TDOA = GridSearchCV(pipeline, param_grid, cv=5)
clf_both = GridSearchCV(pipeline, param_grid, cv=5)

# train the models
clf_ILD.fit(X_train_ILD, y_train)
clf_TDOA.fit(X_train_TDOA, y_train)
clf_both.fit(X_train_both, y_train)

# make predictions 
y_pred_ILD = clf_ILD.predict(X_test_ILD)
y_pred_TDOA = clf_TDOA.predict(X_test_TDOA)
y_pred_both = clf_both.predict(X_test_both)

# calculate model performance
# accuracy
accuracy_ILD = accuracy_score(y_test, y_pred_ILD)
accuracy_TDOA = accuracy_score(y_test, y_pred_TDOA)
accuracy_both = accuracy_score(y_test, y_pred_both)

print('Model Performance Metrics')
print(f'ILD Only Accuracy: {accuracy_ILD*100:.2f}%')
print(f'TDOA Only Accuracy: {accuracy_TDOA*100:.2f}%')
print(f'ILD + TDOA Only Accuracy: {accuracy_both*100:.2f}%')

#%% plotting ROC curve

# predict probabilities for ROC curves
y_prob_ILD  = clf_ILD.predict_proba(X_test_ILD)[:,1]   # probability of class 1
y_prob_TDOA = clf_TDOA.predict_proba(X_test_TDOA)[:,1]
y_prob_both = clf_both.predict_proba(X_test_both)[:,1]

# compute ROC curve and AUC
fpr_ILD, tpr_ILD, _ = roc_curve(y_test, y_prob_ILD)
fpr_TDOA, tpr_TDOA, _ = roc_curve(y_test, y_prob_TDOA)
fpr_both, tpr_both, _ = roc_curve(y_test, y_prob_both)

auc_ILD = auc(fpr_ILD, tpr_ILD)
auc_TDOA = auc(fpr_TDOA, tpr_TDOA)
auc_both = auc(fpr_both, tpr_both)

# plot ROC curves
plt.figure(figsize=(8,6))
plt.plot(fpr_ILD, tpr_ILD, label=f'ILD (AUC = {auc_ILD:.2f})', linewidth=2)
plt.plot(fpr_TDOA, tpr_TDOA, label=f'TDOA (AUC = {auc_TDOA:.2f})', linewidth=2)
plt.plot(fpr_both, tpr_both, label=f'ILD+TDOA (AUC = {auc_both:.2f})', linewidth=2)

plt.plot([0,1], [0,1], 'k--', linewidth=1, label='Chance line') 
plt.xlim([-0.02,1.02])
plt.ylim([-0.02,1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Left vs Right Classification')
plt.legend()
plt.show()