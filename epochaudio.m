function [epochs, t_epoch] = epochaudio(data, t_data, t_markers, fs, epoch_window)
% EPOCH_DATA  Epoch continuous data based on marker timestamps
%
% INPUTS
%   data          : [channels x samples] or [1 x samples]
%   t_data        : time stamps of data (seconds)
%   t_markers     : marker time stamps (seconds)
%   fs            : sampling rate (Hz)
%   epoch_window  : [start end] in seconds relative to marker (e.g. [0 0.5])
%
% OUTPUTS
%   epochs        : [channels x time x trials]
%   t_epoch       : time vector for one epoch (seconds)

% ensure row-wise time
t_data = t_data(:)';
t_markers = t_markers(:)';

% samples per epoch
nsamples = round((epoch_window(2) - epoch_window(1)) * fs);
t_epoch = linspace(epoch_window(1), epoch_window(2), nsamples);

% number of channels
if isvector(data)
    data = data(:)';          % force row
end
nchannels = size(data,1);
ntrials   = length(t_markers);

% preallocate
epochs = nan(nchannels, nsamples, ntrials);

for tr = 1:ntrials
    % epoch start & end time
    t_start = t_markers(tr) + epoch_window(1);
    t_end   = t_markers(tr) + epoch_window(2);

    % find closest indices
    idx_start = find(t_data >= t_start, 1, 'first');
    idx_end   = idx_start + nsamples - 1;

    % boundary check
    if isempty(idx_start) || idx_end > length(t_data)
        continue
    end

    % extract epoch
    epochs(:,:,tr) = data(:, idx_start:idx_end);
end
end
