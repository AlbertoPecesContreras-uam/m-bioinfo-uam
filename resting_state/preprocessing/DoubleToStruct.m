function S_out = DoubleToStruct(double_in, f_names)
    % This function is able to turn double matrix into structure S with 
    % specific fieldnames as EEG channels and values.
    % arg_1 = double matrix where rows are channels and cols values.
    % arg_2 = keys for new structure generated

    S_out = struct();  
    
    % For each channel, we save its respective signal
    for i = 1:length(f_names)
        S_out.(f_names{i}) = double_in(i,:);
    end
end