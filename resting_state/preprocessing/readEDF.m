% Title: Read EEG Data (readEDF.m)
% Author: Alberto Peces Contreras
% Date: 05/10/2024
% Working Time: 1 hour

%% FUNCTIONS
function struct_channels = readEDF(path)
    % This function allows to save all samples from every EEG channel in a
    % specific structure where all data is easily available.
    % arg_1 = specific path to the file.edf
    % Format: "C:\Users\...\...\...\Folder\file.edf"
    % struct_channels -> channel names as keys and values as (1xn) double

    % Load file.edf 
    data = edfread(path);
    
    % Labels from different channels in EEG recording
    channel_labels = data.Properties.VariableNames;
    % Get number of channels
    n_channels = length(channel_labels);
    % Number of seconds for EEG recording
    n_seconds = height(data); 
    
    % Structure for saving all channel recordings
    
    struct_channels = {};
    
    % For every available channel 
    for j = 1:n_channels
        % Save data points for each channel in channel_j
        channel_j = [];
        for i = 1:n_seconds
            % 500-sample window for every recording second
            window = data(i,j).(1);
            % Select samples form each window
            samples = window{1,1};
            % Save samples in each respective channel
            channel_j = [channel_j; samples];
        end
        % Save all channels in a specific structure
        struct_channels.(channel_labels{j}) = channel_j';
    end
end