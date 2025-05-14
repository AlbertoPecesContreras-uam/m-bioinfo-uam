%% FUNCTIONS

function [matches] = processICA(Z_ica, X)

    % This function performs the multiplication between a weight matrix 'W'
    % from a previous ICA process and a set of signals 's'. 

    % Every signal from that set is normalized following the same scaling 
    % parameters used for building weight matrix 'W'.
    
    % double_S: double (m x n), where m is nChannels and n are values.
    % params: {W, mu, std} {23x23,1x23, 1x23}
    
    % Define FRONT CHANNELS from LEFT AND RIGHT EYE with eye-blinks
    %ICA_components = 23;
    %front_channels = [S.Fp1; S.Fp2] = [s(7, :); s(8, :)];
    

    front_channels = [X(7,:); X(8,:)];

    % Force to find one or two independent components with information of
    % eye-blinks from left and right eye.
    matches = findComponents(front_channels, Z_ica);

end

%{

    fig_5 = figure("Visible","on");
    
    
    set(fig_5, 'Name', 'ICA COMPONENTS', 'NumberTitle', 'off');
    set(fig_5, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);  
    
    for i = 1:23
    
        signal = out_ICA(i, :);
    
        subplot(23, 1, i);  % Crear subgráficas para cada componente
        plot(signal);  % Graficar cada componente
        title("C"+num2str(i));
        grid on;
        axis tight;
    end

figure(1);
for i = 1:23
    subplot(23, 1, i);  % Crear subgráficas para cada componente
    plot(signals(:, i));  % Graficar cada componente
    title(['Pre ICA', num2str(i)]);
end

figure(2);
for i = 1:23
    subplot(23, 1, i);  % Crear subgráficas para cada componente
    plot(out(i, :));  % Graficar cada componente
    title(['Post ICA', num2str(i)]);
end
figure(3);
for i = 1:23
    subplot(23, 1, i);  % Crear subgráficas para cada componente
    plot(out_ICA(i, :));  % Graficar cada componente
    title(['Comp ICA ', num2str(i)]);
end
%}

%% FUNCTIONS


function out_matches = findComponents(ref_channels, out_ICA)
    % This function is able to compute an exhaustive comparison between
    % reference channels and every ICA component. The aim of this function
    % is to find those more similar components to reference channels.
    % As many ICA components as reference channels. This is performed by
    % computeSimilarity() function using the value which maximizes 
    % cross-correlation as shift between signals.

    [ICA_comps, ~] = size(out_ICA);
    [nChannels, ~] = size(ref_channels);

    out_matches = [];

    % For each channel with eye-blinks [Fp1, Fp2]
    for j = 1:nChannels
        % Double arrays for saving similarity metrics
        R2 = [];
        RMSE = [];
        % Select and transpose values from front channel
        signal1 = ref_channels(j,:);% Channel for contrast
        
        % For each component from ICA [C1, C2, ..., C23]
        for i = 1:ICA_comps
            % Select values from component(i)
            signal2 = out_ICA(i, :);

            % Compute RMSE and R2 as similarity metrics for both signal1 
            % and signal2 in order to decide how much identity they share.
            % Both signals are scaled following Z-score and corrected in 
            % terms of shift.
            [R2_value, RMSE_value] = computeSimilarityMetrics(signal1, ...
                                                              signal2);
           
            % R2 and RMSE values from each comparison
            R2 = [R2, R2_value];
            RMSE = [RMSE, RMSE_value];
    
        end
        
        
        % Find the best values for R2 with a tolerance equal to 0.05
        threshold_R2 = max(R2)- 0.05 * max(R2);
        best_idx_R2 = find(R2 >= threshold_R2);
        
        % Find the best values for RMSE with a tolerance equal to 0.05
        threshold_RMSE = min(RMSE)+0.05*min(RMSE);
        best_idx_RMSE = find(RMSE <= threshold_RMSE);

        % Select identical idx (ICA component position) from both metrics
        identical = intersect(best_idx_R2, best_idx_RMSE);
        
        % Unique values in best_idx_R2 that do not appear in best_idx_RMSE
        unique_R2 = setdiff(best_idx_R2, best_idx_RMSE);
    
        % Unique values in best_idx_RMSE that do not appear in best_idx_R2 
        unique_RMSE = setdiff(best_idx_RMSE, best_idx_R2);
    
        % Select non-identical idx between both metrics. Non-identical 
        % values are the union between unique values in both lists
        non_identical = union(unique_R2, unique_RMSE);
        
        % Save matches of the more similar ICA components to front channels
        out_matches = [union(identical, non_identical), out_matches];
    
    end

    out_matches = unique(out_matches);

end

%% COMPUTE SIMILARITY METRICS: RMSE and R2

function [R2, RMSE] = computeSimilarityMetrics(signal1, signal2)
    
    % Compute RMSE and R2 as similarity metrics for both signal1 and
    % signal2 in order to decide how much identity they share.
    % Both signals are scaled following Z-score and corrected in terms of
    % shift.

    % arg1 = double array -> 1 row and values as columns -> (1, n)
    % arg2 = double array -> 1 row and values as columns -> (1, n)

    % Scale signals following Z-score
    signal1_norm = (signal1 - mean(signal1)) / std(signal1);
    signal2_norm = (signal2 - mean(signal2)) / std(signal2);

    % Compute shift between two signals by using cross-correlation.
    % The shift will be that value which maximize cross-correlation.
    lag = computeShift(signal1_norm, signal2_norm);
    
    % Correct by shift (if lag is positive, shift signal1; otherwise, signal2)
    if lag >= 0
        % Shift signal2 to the right -> forwards
        signal2_aligned = circshift(signal2_norm, lag);  
    
        % Compute similarity in terms of R2 and RMSE for both signals
        r_matrix = corrcoef(signal1_norm, signal2_aligned);
        
        % R2 and RMSE values
        R2 = r_matrix(1, 2)^2;
        RMSE = sqrt(mean((signal1_norm - signal2_aligned).^2));
        
        % Visualizar las señales alineadas

        %{
        figure;
        plot(signal1_norm, 'b', 'DisplayName', 'Signal 1 (Aligned)');
        hold on;
        plot(signal2_aligned, 'r', 'DisplayName', 'Signal 2 (Aligned)');
        legend;
        title(['R2: ', num2str(R2), ', RMSE: ', num2str(RMSE), ', Desfase: ', num2str(lag)]);
        xlabel('Tiempo');
        ylabel('Amplitud Normalizada');
        grid on;
        %}
        
    else
        % Shift signal1 to the right -> forwards
        signal1_aligned = circshift(signal1_norm, -lag);
    
        % Compute similarity in terms of R2 and RMSE for both signals
        r_matrix = corrcoef(signal1_aligned, signal2_norm);

        % R2 and RMSE values
        R2 = r_matrix(1, 2)^2;
        RMSE = sqrt(mean((signal1_aligned - signal2_norm).^2));
        
        % Visualizar las señales alineadas

        %{
        figure;
        plot(signal1_aligned, 'b', 'DisplayName', 'Signal 1 (Aligned)');
        hold on;
        plot(signal2_norm, 'r', 'DisplayName', 'Signal 2 (Aligned)');
        legend;
        %title(['Correlación: ', num2str(correlation), ', RMSE: ', num2str(rmse_value), ', Desfase: ', num2str(lag)]);
        title(['R2: ', num2str(R2), ', RMSE: ', num2str(RMSE), ', Desfase: ', num2str(lag)]);
        xlabel('Tiempo');
        ylabel('Amplitud Normalizada');
        grid on;
        %}
        
    end

    
end

%% COMPUTE SHIFT BETWEEN SIGNALS

function lag = computeShift(s1_norm, s2_norm)
    % Compute shift between two signals by using cross-correlation.
    % The shift will be that value which maximize cross-correlation.
    % arg1 = double array with values along 1 row -> (1, n)
    % arg2 = double array with values along 1 row -> (1, n)

    % Compute cross-correlation in order to find shift
    [cross_corr, lags] = xcorr(s1_norm, s2_norm);
    
    % Find that shift which maximizes cross-correlation
    [~, idx] = max(cross_corr);
    
    lag = lags(idx);  % This is the shift between both signals
end

%% BACK-UP -> ORIGINAL

%{

function out_matches = findComponents(ref_channels, out_ICA)
    % This function is able to compute an exhaustive comparison between
    % reference channels and every ICA component. The aim of this function
    % is to find those more similar components to reference channels.
    % As many ICA components as reference channels. This is performed by
    % computeSimilarity() function using the value which maximizes 
    % cross-correlation as shift between signals.

    [ICA_comps, ~] = size(out_ICA);
    [nChannels, ~] = size(ref_channels);

    out_matches = [];

    % For each channel with eye-blinks [Fp1, Fp2]
    for j = 1:nChannels
        % Double arrays for saving similarity metrics
        R2 = [];
        RMSE = [];
        % Select and transpose values from front channel
        signal1 = ref_channels(j,:);% Channel for contrast

        % For each component from ICA [C1, C2, ..., C23]
        for i = 1:ICA_comps
            % Select values from component(i)
            signal2 = out_ICA(i, :);

            % Compute RMSE and R2 as similarity metrics for both signal1 
            % and signal2 in order to decide how much identity they share.
            % Both signals are scaled following Z-score and corrected in 
            % terms of shift.
            [R2_value, RMSE_value] = computeSimilarityMetrics(signal1, ...
                                                              signal2);
            
            % R2 and RMSE values from each comparison
            R2 = [R2, R2_value];
            RMSE = [RMSE, RMSE_value];
    
        end
        
        % Find the best values for R2 with a tolerance equal to 0.05
        threshold_R2 = max(R2) - 0.05 * max(R2);
        best_idx_R2 = find(R2 >= threshold_R2);

        % Find the best values for RMSE with a tolerance equal to 0.05
        threshold_RMSE = min(RMSE)+0.05*min(RMSE);
        best_idx_RMSE = find(RMSE <= threshold_RMSE);

        % Select identical idx (ICA component position) from both metrics
        identical = intersect(best_idx_R2, best_idx_RMSE);
        
        % Unique values in best_idx_R2 that do not appear in best_idx_RMSE
        unique_R2 = setdiff(best_idx_R2, best_idx_RMSE);
    
        % Unique values in best_idx_RMSE that do not appear in best_idx_R2 
        unique_RMSE = setdiff(best_idx_RMSE, best_idx_R2);
    
        % Select non-identical idx between both metrics. Non-identical 
        % values are the union between unique values in both lists
        non_identical = union(unique_R2, unique_RMSE);
        
        % Save matches of the more similar ICA components to front channels
        out_matches = [union(identical, non_identical), out_matches];
    
    end

    out_matches = unique(out_matches);

end



function out_matches = findComponents(ref_channels, out_ICA)
    % This function is able to compute an exhaustive comparison between
    % reference channels and every ICA component. The aim of this function
    % is to find those more similar components to reference channels.
    % As many ICA components as reference channels. This is performed by
    % computeSimilarity() function using the value which maximizes 
    % cross-correlation as shift between signals.

    [ICA_comps, ~] = size(out_ICA);
    [nChannels, ~] = size(ref_channels);

    out_matches = [];

    % For each channel with eye-blinks [Fp1, Fp2]
    for j = 1:nChannels
        % Double arrays for saving similarity metrics
        R2 = [];
        RMSE = [];
        % Select and transpose values from front channel
        signal1 = ref_channels(j,:);% Channel for contrast

        % For each component from ICA [C1, C2, ..., C23]
        for i = 1:ICA_comps
            % Select values from component(i)
            signal2 = out_ICA(i, :);

            % Compute RMSE and R2 as similarity metrics for both signal1 
            % and signal2 in order to decide how much identity they share.
            % Both signals are scaled following Z-score and corrected in 
            % terms of shift.
            [R2_value, RMSE_value] = computeSimilarityMetrics(signal1, ...
                                                              signal2);
            
            % R2 and RMSE values from each comparison
            R2 = [R2, R2_value];
            RMSE = [RMSE, RMSE_value];
    
        end
        
        % Find the best values for R2 with a tolerance equal to 0.05
        threshold_R2 = max(R2) - 0.05 * max(R2);
        best_idx_R2 = find(R2 >= threshold_R2);

        % Find the best values for RMSE with a tolerance equal to 0.05
        threshold_RMSE = min(RMSE)+0.05*min(RMSE);
        best_idx_RMSE = find(RMSE <= threshold_RMSE);

        % Select identical idx (ICA component position) from both metrics
        identical = intersect(best_idx_R2, best_idx_RMSE);
        
        % Unique values in best_idx_R2 that do not appear in best_idx_RMSE
        unique_R2 = setdiff(best_idx_R2, best_idx_RMSE);
    
        % Unique values in best_idx_RMSE that do not appear in best_idx_R2 
        unique_RMSE = setdiff(best_idx_RMSE, best_idx_R2);
    
        % Select non-identical idx between both metrics. Non-identical 
        % values are the union between unique values in both lists
        non_identical = union(unique_R2, unique_RMSE);
        
        % Save matches of the more similar ICA components to front channels
        out_matches = [union(identical, non_identical), out_matches];
    
    end

    out_matches = unique(out_matches);

end
%}