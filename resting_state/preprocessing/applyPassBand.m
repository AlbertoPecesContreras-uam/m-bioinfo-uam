% Title: Butterworth Pass-Band Filter (applyPassBand.m)
% Author: Alberto Peces Contreras
% Date: 05/10/2024
% Working Time: 30 min

%% FUNCTIONS
function out = applyPassBand(signal, fs, f_cutoff)
    
    % This function is able to apply a Pass-Band Filter using as 
    % f_cutoff = {f_low, f_high} frequency window.
    % arg_1 = signal (1 x n) double, filter is applied on columns
    % arg_2 = sample rate in [Hz]
    % arg_3 = {f_low, f_high} in [Hz]

    f_low = f_cutoff{1};% Low cut-off [Hz]
    f_high = f_cutoff{2};% High cut-off [Hz]
    
    % Scale cut-off frequencies for filter design
    Wn = [f_low f_high] / (fs / 2);
    
    % Pass-Band filter -> Butterworth (4º order)
    [b, a] = butter(4, Wn, 'bandpass');
    
    % Signal filtered by Pass-Band Filter
    out = filtfilt(b, a, signal);

end
