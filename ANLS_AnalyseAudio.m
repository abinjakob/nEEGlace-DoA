clear; clc; close all;

% data path
rootpath   = 'L:\Cloud\T-PsyOL\PhD Project\nEEGlace\sourceDoA\Data'; 
foldername = 'Piloting-16012026';  
filename   = 'sub-P001_ses-S001_task-far180_run-001_eeg.xdf'; 

cd('L:\Cloud\T-PsyOL\PhD Project\nEEGlace\sourceDoA')

% load streams from XDF data
streams = load_xdf(fullfile(rootpath, foldername, filename)); 

% audio data
audio = streams{2}.time_series;
t_audio = streams{2}.time_stamps; 
fs = str2num(streams{2}.info.nominal_srate); 

% markers 
markers = streams{1}.time_series; 
t_markers = streams{1}.time_stamps; 

% epoching data 
epoch_window = [0 0.5];   
[epochs, t_epoch] = epochaudio(audio, t_audio, t_markers, fs, epoch_window);


%% plot some raw data

% duration of audio to plot
duratio2plot = 100; 

t0 = t_audio(1);
idx = (t_audio - t0) <= duratio2plot;

figure
hold on 
% plot audio
plot(t_audio(idx) - t0, audio(1, idx), 'r')
plot(t_audio(idx) - t0, audio(2, idx), 'b')
% plot markers 
for i = 1:length(t_markers)
    if t_markers(i) - t0 <= duratio2plot
        if markers{i} == '1'
            xline(t_markers(i) - t0, 'g--', markers{i});
        elseif markers{i} == '2'
            xline(t_markers(i) - t0, 'c--', markers{i});
        end
    end
end

xlabel('Time [sec]'); ylabel('Amplitude'); 


%% play audio

audio(isnan(audio)) = 0;
fs = str2double(streams{2}.info.nominal_srate);
player = audioplayer(audio, fs);
play(player)

% stop(player)

%% seperate trials 
trialsL = []; trialsR = [];

for i=1:size(markers,2)
    if markers{i} == '1'
        trialsL = cat(3, trialsL, epochs(:,:,i));
    elseif markers{i} == '2'
        trialsR = cat(3, trialsR, epochs(:,:,i));
    end
end

%% plot trials

figure;

% plot left trials 
subplot(2,2,1)
plot(t_epoch, mean( trialsL(1,:,:) ,3), 'r')
subplot(2,2,2)
plot(t_epoch, mean( trialsL(2,:,:) ,3), 'b')
xlabel('Time [sec]'); ylabel('Amplitude'); 

% plot right trials 
subplot(2,2,3)
plot(t_epoch, mean( trialsR(1,:,:) ,3), 'r')
subplot(2,2,4)
plot(t_epoch, mean( trialsR(2,:,:) ,3), 'b')
xlabel('Time [sec]'); ylabel('Amplitude'); 