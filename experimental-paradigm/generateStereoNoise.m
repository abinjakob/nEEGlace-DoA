
clear; clc; close all; 
rng('shuffle');

% ------------------------------------------------------------------------
% ----------------------------- SCRIPT SETUP -----------------------------

% parameters
fs = 44100;                  % sampling rate (hz)
stimDur = 0.5;               % duration of stimulus (sec)
iti = 1;                     % inter-trial interval (sec)
trialsPerSide = 200;         % number of trials per side
freqRange = [300 6000];      % band-limited noise (hz)
startwait = 20;              % time to wait before experiment start (sec)

% ------------------------------------------------------------------------

% generate sitm
nSamples = round(stimDur * fs);
t = (0:nSamples-1)/fs;
% pre-generate band-limited noise
noiseFull = randn(nSamples,1);
[b,a] = butter(4,freqRange/(fs/2));   
noiseFiltered = filter(b,a,noiseFull);

% setup trial
trials = [ones(trialsPerSide,1); ones(trialsPerSide,1)*2]; 
trials = trials(randperm(length(trials))); 

% initialize LSL 
disp('Loading LSL library...');
lib = lsl_loadlib();
% create a new LSL marker stream 
disp('Creating a new marker stream info...');
info = lsl_streaminfo(lib,'MarkerStream','Markers',1,0,'cf_string','myuniquesourceid23443');
outlet = lsl_outlet(info);

% setup PsychToolBox
AssertOpenGL;
InitializePsychSound(1);
nrchannels = 2; 
pahandle = PsychPortAudio('Open', [], 1, 1, fs, nrchannels);

% pause before experiment start
disp(['Experiment will begin in ', num2str(startwait), ' sec...'])
WaitSecs(startwait);

% main trial loop
for i = 1:length(trials)
    
    side = trials(i);
    
    % create stereo signal
    stereoSig = zeros(nSamples,2);
    if side == 1 
        stereoSig(:,1) = noiseFiltered;
        stereoSig(:,2) = 0;    % silent right
        marker = '1'; 
    else % right speaker
        stereoSig(:,1) = 0;    % silent left
        stereoSig(:,2) = noiseFiltered;
        marker = '2'; 
    end
    
    % send LSL marker 
    outlet.push_sample({marker});
    
    % fill buffer
    PsychPortAudio('FillBuffer', pahandle, stereoSig');
    % play stimulus
    PsychPortAudio('Start', pahandle, 1, 0, 1);
    
    % Wait for duration + ITI
    WaitSecs(stimDur + iti);
    
end

% cleanup
PsychPortAudio('Close', pahandle);
disp('Pilot complete!');
