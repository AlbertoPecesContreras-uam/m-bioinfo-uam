%% READING

% Get data directory
data_path = "E:\TFM\DATOS";

% Display data directory 
disp("# ---- Reading Directory ---- #")
disp("Data path:")
disp(data_path)
disp(' ');

% Set specific data subpaths
EEG_path = data_path+'\EEG';
MAT_path = data_path+'\MATLAB';

EEG = readDirectory(EEG_path);% CHECK --> OK
MAT = readDirectory(MAT_path);% CHECK --> OK

% Display subfolders in subpaths
disp('PROCESS: ');
disp("-) DATA .\EEG:")
disp(' ');
disp("-) DATA .\MATLAB:")
disp(' ');

%% BUILDING DIRECTORIES

% controls = EEG.datos{1}{1} --> POST
% experiments = EEG.datos{1}{2} --> POST
% controls = EEG.datos{2}{1} --> PRE
% experiments = EEG.datos{2}{2} --> PRE
% controls = EEG.datos{3}{1} --> SEG
% experiments = EEG.datos{3}{2} --> SEG

%disp(EEG.instante{1}+"\"+EEG.grupo{1}{1}+"\"+EEG.datos{1}{1}{1});
%disp(EEG.instante{1}+"\"+EEG.grupo{1}{1}+"\"+EEG.datos{1}{1}{1});
%disp(EEG.datos{1}{1}{1}(1:l-4)+".csv");

% Set MAIN path and FOLDER for saving data
save_folder = "E:\TFM\PREPROCESADO";

% Set subpaths for saving data inside main folder
build_EEG_path = save_folder+"\EEG";
build_MAT_path = save_folder+"\MAT";

% Build directory with original arrangement for saving output

% -> .\PREPROCESADO\EEG
% -> .\PREPROCESADO\MAT

buildDirectory(save_folder, build_EEG_path, EEG);% CHECK --> OK
buildDirectory(save_folder, build_MAT_path, MAT);% CHECK --> OK

%% STARTING POINT: PREPROCESSING

% Add the folder to the path where fastICA is found (...)
addpath('pca_ica\')

disp('0) ONE subject.');
disp('1) ALL subjects.');
option = input('ICA Weight Matrix Type:');

if option == 0

    subject_ref_OA_path = data_path+"\EYE-BLINKS-REFERENCIA" + ...
               "\20240227104618_POST_Control_89_MAP_OA.edf";

    subject_ref_OC_path = data_path+"\EYE-BLINKS-REFERENCIA" + ...
               "\20240227105021_POST_Control_89_MAP_OC.edf";

    % Read .edf file and gather all channels with their data points
    S_ref_OA = readEDF(subject_ref_OA_path);
    S_ref_OC = readEDF(subject_ref_OC_path);

    % EEG matrix (m x n), where m is nChannel and n is values
    one_OA_EEG = StructToDouble(S_ref_OA);
    one_OC_EEG = StructToDouble(S_ref_OC);

    % Extreme initial/final picks -> IGNORE THEM
    one_OA_EEG = one_OA_EEG(:,2000:74901);
    one_OC_EEG = one_OC_EEG(:,2000:74901);

    data_OA_EEG = one_OA_EEG;
    data_OC_EEG = one_OC_EEG;

else
    % Concatenate all signals from OA and OC tests building a matrix (23xn)
    all_OA_EEG = concatenateSignals(EEG_path, EEG, 1, "OA");% CHECK --> OK
    all_OC_EEG = concatenateSignals(EEG_path, EEG, 1, "OC");% CHECK --> OK

    data_OA_EEG = all_OA_EEG;
    data_OC_EEG = all_OC_EEG;
end

% Remove X, Y, and Z channels
data_OA_EEG = data_OA_EEG(1:end-3,:);
data_OC_EEG = data_OC_EEG(1:end-3,:);

f_cut_off = {1,30};% Frequency Window of Interest

[m,~] = size(data_OA_EEG); 

% Aplicamos Pass-Band Filter (original fs = 500 Hz)
for i = 1:m
    data_OA_EEG(i,:) = applyPassBand(data_OA_EEG(i,:), 500, f_cut_off);
    data_OC_EEG(i,:) = applyPassBand(data_OC_EEG(i,:), 500, f_cut_off);
end

% DIMENSIONS
% data_OA_EEG = (20 x K), K = m·N, (m = Nº subjects, N = signal samples) 
% data_OC_EEG = (20 x K), K = m·N, (m = Nº subjects, N = signal samples) 

% CENTER ROWS (mean by rows)
[Zc_OA, mu_OA] = centerRows(data_OA_EEG);% mu_OA = (20 x 1)
[Zc_OC, mu_OC] = centerRows(data_OC_EEG);% mu_OC = (20 x 1)

% STANDARD DEVIATION (std by rows)
std_OA = std(data_OA_EEG,1,2);% std_OA = (20 x 1)
std_OC = std(data_OC_EEG,1,2);% std_OC = (20 x 1)

% WHITEN ROWS
% Operation (X./v): divide each row of 'X' by corresponding value in 'v'
% X_1_1/std_1, X_1_2/std_1, X_1_3/std_1, ..., X1_K/std_1
% X_2_1/std_2, X_2_2/std_2, X_2_3/std_2, ..., X2_K/std_2
% ...
% X_20_1/std_20, X_20_2/std_20, X_20_3/std_20, ..., X20_K/std_20
[Zw_OA, T_OA] = whitenRows(Zc_OA./std_OA);
[Zw_OC, T_OC] = whitenRows(Zc_OC./std_OC);

disp('Media por fila (OA):');
disp(mean(Zw_OA, 2));
disp('Desviación estándar por fila (OA):');
disp(std(Zw_OA, 0, 2));

disp('Media por fila (OC):');
disp(mean(Zw_OC, 2));
disp('Desviación estándar por fila (OC):');
disp(std(Zw_OC, 0, 2));

[~, W_OA, ~, ~] = fastICA(Zw_OA, m, 'kurtosis', 1);
[~, W_OC, ~, ~] = fastICA(Zw_OC, m, 'kurtosis', 1);

paramsICA_OA = {mu_OA, std_OA, T_OA, W_OA};
paramsICA_OC = {mu_OC,std_OC, T_OC, W_OC};

params = {paramsICA_OA, paramsICA_OC};

startPreprocessing(EEG_path, build_EEG_path, EEG, params, 1);

%% FUNCTIONS

function r_comps = getCompsICA(path, S_Dir, params, flag, type)
    
    % For OA files:
    % It process OA.edf files and gather (for each one of them) all ICA 
    % components detected as eye-blink artifacts.
    
    % As output, we will have 1 component for removal.
    
    eye_blink_comps = [];

    params_OA = params{1};

    % For each directory with time instant
    for i = 1:length(S_Dir.instante)
        % For every group inside each time instant
        for j = 1:length(S_Dir.grupo{i})
            % For every subject inside each group
            for k = 1:length(S_Dir.datos{i}{j})

                % Build path for specific search
                search_path = path+"\"+S_Dir.instante{i}+"\"+...
                                       S_Dir.grupo{i}{j}+"\"+...
                                       S_Dir.datos{i}{j}{k};


                
                % Check if the substring _OA.edf exists in the string
                isOA = contains(S_Dir.datos{i}{j}{k}, '_OA.edf');

                if isOA && type == "OA" 
                    % EEGmatrix(m x n), where m is nChannel and n is values
                    double_EEG = StructToDouble(readEDF(search_path));
                    
                    % Extreme initial/final picks -> IGNORE THEM
                    % Ignore X, Y and Z channels
                    X = double_EEG(1:end-3,2000:74901);
                    
                    % Get dimension of EEG matrix
                    [m_EEG,~] = size(X);

                    % Read OA.edf file and gather all channels with their 
                    % respective data points
                    
                    disp('ANALIZANDO: ');
                    disp(search_path);
    
                    % Set parameters for preprocessing filters
                    %fs = 500;
                    %f_cut_off = {0.5,40};

                    % Apply Pass-Band Filter
                    for m = 1:m_EEG
                        X(m,:) = applyPassBand(X(m,:), 500, {1,30});
                    end

                    %k = round(500 / 256);  % Factor de reducción

                    % Downsampling a 256 Hz
                    %X = downsample(X', round(500 / 256))';  

                    mu_OA = params_OA{1};
                    std_OA = params_OA{2};
                    T_OA = params_OA{3};
                    W_OA = params_OA{4};
                    
                    % Center
                    Xc = bsxfun(@minus,X,mu_OA);
                    % Apply ICA
                    Z_ica = W_OA*(T_OA*(Xc./std_OA));

                    e_b_comps = processICA(Z_ica, Xc./std_OA);

                    % Save components for removal decision
                    eye_blink_comps = [eye_blink_comps, e_b_comps];

                end

                if flag == 1
                    p = k/length(S_Dir.datos{i}{j})*100;
                    fprintf('Progress: %.2f%%\r',round(p,2));
                    if p == 100.00
                        clc;
                    end
                end
            end
        end
    end


    % Inicializamos con 'true'
    rows_to_select = true(m_EEG, 1); 
    if type == "OA"
        
        OA_comps = [mode(eye_blink_comps)];
        rows_to_select(OA_comps) = false;
    end

    r_comps = rows_to_select;
end

function startPreprocessing(path, build_path, S_Dir, params, flag)

    params_OA = params{1};
    params_OC = params{2};

    % First step:
    % Get ICA components for removal in OA and OC files
    % Second step:

    r_comps_OA = getCompsICA(path, S_Dir, params, 1, "OA");


    disp("Components for Removal in OA: ");
    disp(find(r_comps_OA==0));
    disp(' ');

    % For each directory with time instant
    for i = 1:length(S_Dir.instante)
        % For every group inside each time instant
        for j = 1:length(S_Dir.grupo{i})
            % For every subject inside each group
            for k = 1:length(S_Dir.datos{i}{j})

                % Build path for specific search
                search_path = path+"\"+S_Dir.instante{i}+"\"+...
                                       S_Dir.grupo{i}{j}+"\"+...
                                       S_Dir.datos{i}{j}{k};
                disp('ANALIZANDO: ');
                disp(search_path);

                % Check if the substring _OA.edf exists in the string
                isOA = contains(S_Dir.datos{i}{j}{k}, '_OA.edf');
                % Check if the substring _OC.edf exists in the string
                isOC = contains(S_Dir.datos{i}{j}{k}, '_OC.edf');

                if isOA || isOC
                    % Read .edf file and gather all channels with their 
                    % respective data points
                    S = readEDF(search_path);
                    
                    % EEG matrix (m x n), where m is nChannel and n is values
                    double_EEG = StructToDouble(S);

                    % Extreme initial picks -> IGNORE THEM
                    X = double_EEG(1:end-3,2000:74901);
                    
                    % Get dimension of EEG matrix
                    [m_EEG,~] = size(X);
    
                    % Set parameters for preprocessing filters
                    %fs = 500;
                    %f_cut_off = {0.5,40};
                    
                    % Get channel names
                    %field_names = fieldnames(S);
    
                    % Apply Pass-Band Filter
                    for m = 1:m_EEG
                        X(m,:) = applyPassBand(X(m,:), 500, {1,30});
                    end
                    % Downsampling a 256 Hz
                    %X = downsample(X', round(500 / 256))';
    
                    % Build path for saving filtered channels
                    save_path = build_path+"\"+S_Dir.instante{i}+ ...
                                           "\"+S_Dir.grupo{i}{j};
                end

                    
                if isOA
                    mu_OA = params_OA{1};
                    std_OA = params_OA{2};
                    T_OA = params_OA{3};
                    W_OA = params_OA{4};
                    
                    % Center
                    Xc = bsxfun(@minus,X,mu_OA);

                    % Apply ICA
                    Z_ica = W_OA*(T_OA*(Xc./std_OA));

                    [m_ica, ~] = size(Z_ica);
                    
                    
                    % comps is in {} format -> cell array
                    comps_OA = cell(1,m_ica);
                    for c = 1:m_ica
                        comps_OA{c} = strcat('C',num2str(c));
                    end

                    comps_OA = comps_OA(r_comps_OA);

                    Z_ica = Z_ica(r_comps_OA,:);
                    
                    % Downsampling a 256 Hz
                    Z_ica = downsample(Z_ica', round(500 / 256))';

                    S_OUT_ICA = DoubleToStruct(Z_ica, comps_OA);

                    % Save data
                    saveData(save_path, S_Dir.datos{i}{j}{k}, S_OUT_ICA);

                elseif isOC
                    
                    mu_OC = params_OC{1};
                    std_OC = params_OC{2};
                    T_OC = params_OC{3};
                    W_OC = params_OC{4};
                    
                    % Center
                    Xc = bsxfun(@minus,X,mu_OC);

                    % Apply ICA
                    Z_ica = W_OC*(T_OC*(Xc./std_OC));

                    [m_ica, ~] = size(Z_ica);
                  
                    
                    % comps is in {} format -> cell array
                    comps_OC = cell(1,m_ica);
                    
                    for c = 1:m_ica
                        comps_OC{c} = strcat('C',num2str(c));
                    end
                    
                    % Downsampling a 256 Hz
                    Z_ica = downsample(Z_ica', round(500 / 256))';

                    S_OUT_ICA = DoubleToStruct(Z_ica, comps_OC);

                    % Save data
                    saveData(save_path, S_Dir.datos{i}{j}{k}, S_OUT_ICA);
                
                end

                if flag == 1
                    p = k/length(S_Dir.datos{i}{j})*100;
                    fprintf('Progress: %.2f%%\r',round(p,2));
                    disp("SUCCESSFULLY SAVED: "+S_Dir.datos{i}{j}{k});
                    if p == 100.00
                    clc;
                    end
                end
            end
        end
    end
end

function out = concatenateSignals(path, S_Dir, flag, type)% CHECK --> OK
    
    % Select all signals and concatenate each channel. 
    % (23 x n), where n are all signals in each channel concatenated.

    out = [];

    for i = 1:length(S_Dir.instante)
        % For every group inside each time instant
        for j = 1:length(S_Dir.grupo{i})
            % For every subject inside each group
            for k = 1:length(S_Dir.datos{i}{j})

                % Build path for specific search
                search_path = path+"\"+S_Dir.instante{i}+"\"+...
                              S_Dir.grupo{i}{j}+"\"+S_Dir.datos{i}{j}{k};
                
                
                isOA = contains(S_Dir.datos{i}{j}{k}, '_OA.edf');
                isOC = contains(S_Dir.datos{i}{j}{k}, '_OC.edf');

                if isOA && type == "OA"
                    % Read .edf file and gather all channels with their 
                    % respective data points
                
                    disp('ANALIZANDO: ');
                    disp(search_path);
                    
                    S = readEDF(search_path);
                    
                    % EEG matrix (m x n), m is nChannel and n is values
                    double_EEG = StructToDouble(S);
                    
                    % Extreme initial/final picks -> IGNORE THEM
                    double_EEG = double_EEG(:,2000:74901);

                    % Concatenate all signals in rows
                    out = [out, double_EEG];

                elseif isOC && type == "OC"
                    % Read .edf file and gather all channels with their 
                    % respective data points
                
                    disp('ANALIZANDO: ');
                    disp(search_path);
                    
                    S = readEDF(search_path);
                    
                    % EEG matrix (m x n), m is nChannel and n is values
                    double_EEG = StructToDouble(S);
                    
                    % Extreme initial/final picks -> IGNORE THEM
                    double_EEG = double_EEG(:,2000:74901);
                    
                    % Concatenate all signals in rows
                    out = [out, double_EEG];

                end

                if flag == 1
                    p = k/length(S_Dir.datos{i}{j})*100;
                    fprintf('Progress: %.2f%%\r',round(p,2));
                    disp("SUCCESSFULLY SAVED: "+S_Dir.datos{i}{j}{k});
                    if p == 100.00
                        clc;
                    end
                end
            end
        end
    end


end
