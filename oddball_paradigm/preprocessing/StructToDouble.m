%% Turn Structure to Double format -> ¡ keys as columns !
function out_double = StructToDouble(S)

    % This function is able to turn structure S with fieldnames as EEG
    % channels and values as their respective signals into double array.
    % arg_1 = structure with keys as EEG channels and values as signals.

    % size(out_double) = [values, keys] 
    % Channels and values are columns and rows, respectively.

    out_double = [];  % Concatenate by columns
    
    % Names from each EEG channel
    keys = fieldnames(S);
    
    % For each channel, we extract and concatenate signals
    for i = 1:length(keys)
        signal = S.(keys{i});
        
        % Concatenate by rows -> (1xn) signal in each row
        out_double = [out_double; signal];  
    end
end