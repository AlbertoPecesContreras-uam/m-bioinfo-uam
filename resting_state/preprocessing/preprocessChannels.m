%% FUNCTIONS

function out_S = preprocessChannels(S, option)
    % This function is able to apply all steps for preprocessing:
    % Pass-Band, ICA (option="all"), and Gaussian-Smooth filters.
    % arg_1 = structure with all channels as fieldnames and EEG datapoints 
    % as values.
    % arg_2 = string -> if "all", ICA is applied. Otherwise, not. 

    % Save all fieldnames in keys variable
    keys = fieldnames(S);
    
    fs = 500; % Frequency rate in [Hz]
    f_cutoff = {0.5, 100}; % Frequency rate window for PB-Filter
    sigma = 2; % Variance for GS-filter 
    
    % Z-Score normalized and Pass-Band Filter
    for i = 1:length(keys)
        % Select current key (CHANNEL)
        current_key = keys{i};
        
        % Select values from current key (CHANNEL values)
        signal = S.(current_key);
        
        % Extreme initial picks -> IGNORE THEM
        signal = signal(100:end);
        
        % Z-score
        signal = standardScaler(signal);
        
        % Update signal values after applying Pass-Band Filter
        S.(current_key) = applyPassBand(signal, fs, f_cutoff);
    end

    % Independent Component Analysis on OA files
    if option == "all"
        % Apply ICA
        out_ICA = applyICA(S);

        for j = 1:length(keys)
            % Select current key (channel)
            current_key = keys{j};
            % Update signal values after ICA preprocessing
            S.(current_key) = out_ICA(j,:);
        end
    end
    
    % Gaussian-Smooth Filter
    for k = 1:length(keys)

        % Select current key (channel)
        current_key = keys{k};

        % Select values from current key (signal values)
        signal = S.(current_key);
        
        % Apply Gaussian-Smooth Filter
        out_GS = applyGaussianSmooth(signal, sigma);

        % Update signal values after Gaussian-Smooth preprocessing
        S.(current_key) = out_GS;

    end
    
    out_S = S;

end