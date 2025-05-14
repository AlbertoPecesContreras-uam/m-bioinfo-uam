

% Conclusiones: tenemos diferentes archivos.mat correspondientes a los
% estimulos generados en las pruebas ODDBALL 1 y 2 para cada paciente.
% Estos archivos poseen puntos donde se generan "std" o estimulos estandar
% y "tgt" o estimulos desviados. Cada punto posee un instante de tiempo
% asociado, junto con el tipo de estimulo producido en la prueba.

% Los archivos ODDBALL poseen dos tipos de estímulos, por ende, vamos a
% crear un dataframe por sujeto, donde cada fila será un estímulo y cada
% columna una característica de interés.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ------------------------------------------------------------------------
% READ DIRECTORY
% ------------------------------------------------------------------------

dir_mat = readDirectory("E:\TFM\DATOS\MATLAB");
dir_eeg = readDirectory("E:\TFM\DATOS\EEG");

% ------------------------------------------------------------------------
% BUILD SAVING DIRECTORY
% ------------------------------------------------------------------------

% Set MAIN path and FOLDER for building save directory
buildDirectory("E:\TFM\PREPROCESADO", "MAT", dir_mat);

buildDirectory("E:\TFM\PREPROCESADO", "TIME", dir_mat);

dir_files = {dir_mat, dir_eeg};

% ------------------------------------------------------------------------
% MAIN
% ------------------------------------------------------------------------

%issue = checkData("E:\TFM\DATOS",dir_files);

extractComponents("E:\TFM\DATOS",dir_files,"E:\TFM\PREPROCESADO\MAT",1);

%extractTimes("E:\TFM\DATOS",dir_files,"E:\TFM\PREPROCESADO\TIME",1);


%% Functions
function issue = checkData(path, S_Dir)
    % S_dir: structure of directories comprised by dir_MAT and dir_EEG
    dir_MAT = S_Dir{1};% structure of MAT directories and info for (.mat)
    dir_EEG = S_Dir{2};% structure of EEG directories and info for (.edf)
    
    % structure to save files with errors
    issue = struct();

    % For each directory with time instant
    for i = 1:length(dir_EEG.instante)
        instante = dir_EEG.instante{i};
        % For every group inside each time instant
        for j = 1:length(dir_EEG.grupo{i})
            grupo = dir_EEG.grupo{i}{j};
            % Seleccionamos todos los archivos.mat dentro del grupo
            estimulos = dir_MAT.datos{i}{j};
            % For every subject inside each group
            for k = 1:length(dir_EEG.datos{i}{j})
                sujeto = dir_EEG.datos{i}{j}{k};

                % Build path for specific search (.edf)
                search_path = path+"\EEG\"+instante+"\"+grupo+"\"+sujeto;

                % Check if the substring ODDBALL exists in the string
                if contains(sujeto, 'ODDBALL')

                    % Mostramos el path de cada archivo.edf
                    disp('ANALIZANDO: ');
                    disp(search_path);

                    % Removemos la extension .edf
                    sujeto = erase(sujeto, ".edf");
                    % Separamos el nombre del archivo usando del="_"
                    fields_sujeto = strsplit(sujeto,"_");
                    % Convertimos el id en double format
                    id_sujeto = str2double(fields_sujeto{4});
                    % Convertimos el nº de ODDBALL en double format
                    n_oddball = str2double(fields_sujeto{7});

                    % Read .edf file and gather all info about timings.
                    % onset_array: instants for stimulis in seconds
                    [~,onset_array] = readEDF(search_path);
                    
                    for n = 1:length(estimulos)
                        % estimulos{n} = POST_Control_64_FLL_1 (.mat)
                        fields_mat = strsplit(estimulos{n}, "_");
                        % ID subject = 64
                        id_mat = str2double(fields_mat{3});
                        % ODDBALL Nº: 1
                        n_test_mat = str2double(fields_mat{5});
                        
                        % Match same .edf and .mat files
                        if n_test_mat == n_oddball && id_mat == id_sujeto

                            % Hay (.edf) que poseen menos de 100 estimulos
                            % y debemos detectarlos.
                            if length(onset_array) < 100
                                % Eliminamos 1º campo en el filename
                                sujeto = regexprep(sujeto, '^\d+', '');
                                % Eliminamos _ restante en el filename
                                sujeto = regexprep(sujeto, '^_', '');
                                % Guardamos filenames y nº de estimulos
                                issue.(sujeto) = length(onset_array);
                            end
                        end
                    end
                end
            end
        end
    end                
end

function extractComponents(path, S_Dir, build_path, flag)
    
    % S_dir: structure of directories comprised by dir_MAT and dir_EEG
    dir_MAT = S_Dir{1};% structure of MAT directories and info for (.mat)
    dir_EEG = S_Dir{2};% structure of EEG directories and info for (.edf)

    plcb = [1,4,7,9,11,13,14,15,17,18,19,24,25,27,28,31,35,40,41, ...
                42,43,44,47,48,49,50,52,61,63,65,69,105];

    % For each directory with time instant
    for i = 1:length(dir_EEG.instante)
        instante = string(dir_EEG.instante{i});
        % For every group inside each time instant
        for j = 1:length(dir_EEG.grupo{i})
            grupo = string(dir_EEG.grupo{i}{j});
            % Seleccionamos todos los archivos.mat dentro del grupo
            estimulos_mat = dir_MAT.datos{i}{j};
            % For every subject inside each group
            for k = 1:length(dir_EEG.datos{i}{j})
                sujeto = dir_EEG.datos{i}{j}{k};

                p = k/length(dir_EEG.datos{i}{j})*100;

                % Build path for specific search (.edf)
                search_path = path+"\EEG\"+instante+"\"+grupo+"\"+sujeto;

                % Build path for saving P600 stimulis
                save_path = build_path+"\"+instante+"\"+grupo;

                % Check if the substring ODDBALL exists in the string
                if contains(sujeto, 'ODDBALL')

                    % Mostramos el path de cada archivo.edf
                    disp('ANALIZANDO: ');
                    disp(search_path);

                    % Removemos la extension .edf
                    sujeto = erase(sujeto, ".edf");
                    % Separamos el nombre del archivo usando del="_"
                    fields_sujeto = strsplit(sujeto,"_");
                    % Convertimos el id en double format
                    id_sujeto = str2double(fields_sujeto{4});
                    % Convertimos el nº de ODDBALL en double format
                    n_oddball = str2double(fields_sujeto{7});

                    % Read .edf file and gather all info about timings.
                    % onset_array: instants for stimulis in seconds
                    [S,onset_array] = readEDF(search_path);
                    
                    % Contamos cuantos estimulos hay registrados en (.edf)
                    n_timings = length(onset_array);

                    % Get channel names
                    channel_names = fieldnames(S);

                    % Turn S into EEG matrix (m x n), where m and n are
                    % nChannels and values
                    X = StructToDouble(S);
                    
                    % No X,Y,Z channels
                    X = X(1:end-3,:);
                    channel_names = channel_names(1:end-3);
                    % Add 'type' field for DataFrame
                    channel_names = [{'comp'}; channel_names];
                    % Add 'comp' field for DataFrame
                    channel_names = [{'type'}; channel_names];
                    % Add 'grupo' field for DataFrame
                    channel_names = [{'id'}; channel_names];
                    % Add 'id' field for DataFrame
                    channel_names = [{'grupo'}; channel_names];
                    % Add 'instante' field for DataFrame
                    channel_names = [{'instante'}; channel_names];
                    % Add 'n_test' field for DataFrame
                    channel_names = [{'n_test'}; channel_names]';

                    for n = 1:length(estimulos_mat)
                        % estimulos{n} = POST_Control_64_FLL_1 (.mat)
                        fields_mat = strsplit(estimulos_mat{n}, "_");
                        % ID subject = 64
                        id_mat = str2double(fields_mat{3});
                        % ODDBALL Nº: 1
                        n_test_mat = str2double(fields_mat{5});
                        
                        % Check if (.mat) and (.edf) matches
                        if n_test_mat == n_oddball && id_mat == id_sujeto

                            % Si hay > 90 timings y no esta vacio, then
                            if ~isempty(n_timings) && n_timings > 90

                                % Ruta hacia archivos (.mat)
                                path_mat = path+"\MATLAB\"+instante+"\"+...
                                           grupo+"\"+estimulos_mat{n};
    
                                % Cargamos informacion del tipo de estimulo
                                % 'std' o 'tgt' en (.mat)
                                data = load(path_mat);

                                % Extraemos cuales son std[0] y tgt[1]
                                stimulis = binarize(data.task.timing);
                                
                                % Si tenemos menos timings en (.edf) que
                                % marcados en (.mat), entonces:
                                if n_timings <= length(stimulis)
                                    stimulis = stimulis(1:n_timings);
                                end

                                % std: seleccionamos los idx donde es 0
                                std_onset = onset_array(stimulis==0);
                                % tgt: seleccionamos los idx donde es 1
                                tgt_onset = onset_array(stimulis==1);

                                % Compute median window for every channel
                                X_std = averageWindows(X, std_onset, ...
                                                       500, "median");
                                X_tgt = averageWindows(X, tgt_onset, ...
                                                       500, "median");

                                % Compute components for every avg window
                                X_std_all = getComponent(X_std,"all");
                                X_tgt_all = getComponent(X_tgt,"all");

                                X_std_bs = getComponent(X_std,"baseline");
                                X_tgt_bs = getComponent(X_tgt,"baseline");
                                
                                X_std_pre = getComponent(X_std,"pre-trigger");
                                X_tgt_pre = getComponent(X_tgt,"pre-trigger");

                                X_std_post = getComponent(X_std,"post-trigger");
                                X_tgt_post = getComponent(X_tgt,"post-trigger");

                                X_std_P1 = getComponent(X_std,"P1");
                                X_tgt_P1 = getComponent(X_tgt,"P1");

                                X_std_N1 = getComponent(X_std,"N1");
                                X_tgt_N1 = getComponent(X_tgt,"N1");

                                X_std_P2 = getComponent(X_std,"P2");
                                X_tgt_P2 = getComponent(X_tgt,"P2");

                                X_std_N2 = getComponent(X_std,"N2");
                                X_tgt_N2 = getComponent(X_tgt,"N2");

                                X_std_P3 = getComponent(X_std,"P3");
                                X_tgt_P3 = getComponent(X_tgt,"P3");
                                
                                % Compute string arrays for components
                                all = repStr("all",size(X_std_all,2));
                                bs = repStr("baseline",size(X_std_bs,2));
                                pre = repStr("pre-trigger",size(X_std_pre,2));
                                post = repStr("post-trigger",size(X_std_post,2));
                                P1 = repStr("P1",size(X_std_P1,2));
                                N1 = repStr("N1",size(X_std_N1,2));
                                P2 = repStr("P2",size(X_std_P2,2));
                                N2 = repStr("N2",size(X_std_N2,2));
                                P3 = repStr("P3",size(X_std_P3,2));

                                % Concatenamos los string arrays por filas
                                % Obtenemos un vector columna (M x 1)
                                comps_std = [all; bs; pre; post; P1; ...
                                             N1; P2; N2; P3];
                                comps_tgt = [all; bs; pre; post; P1; ....
                                             N1; P2; N2; P3];
                              
                                std = repStr("std",size(comps_std,1));
                                tgt = repStr("tgt",size(comps_tgt,1));

                                % Concatenamos las ventanas de componentes
                                X_w_std = [X_std_all'; X_std_bs'; ...
                                           X_std_pre'; X_std_post';...
                                           X_std_P1'; X_std_N1'; ...
                                           X_std_P2';X_std_N2';X_std_P3'];
                                X_w_tgt = [X_tgt_all'; X_tgt_bs'; ...
                                           X_tgt_pre'; X_tgt_post'; ...
                                           X_tgt_P1'; X_tgt_N1'; ...
                                           X_tgt_P2';X_tgt_N2';X_tgt_P3'];

                                % Concatenamos comps_std y comps_tgt 
                                comps = [comps_std; comps_tgt];
                                types = [std; tgt];
                                % Construimos arrays de informacion
                                n_test = repStr(n_oddball,size(comps,1));
                                ins = repStr(instante,size(comps,1));
                                % Controlamos el grupo placebo
                                if ismember(id_sujeto, plcb)
                                    gr = repStr("PLCB",size(comps,1));
                                else
                                    gr = repStr(grupo,size(comps,1));
                                end

                                id = repStr(id_sujeto,size(comps,1));
                                
                                % Concatenamos X_w_std y X_w_tgt
                                X_w = [X_w_std; X_w_tgt];
                                
                                % Build DataFrame
                                df = [array2table(n_test) ...
                                      array2table(ins) ...
                                      array2table(gr) ...  
                                      array2table(id) ...
                                      array2table(types) ...
                                      array2table(comps) ...
                                      array2table(X_w)];
                                
                                % Set column names
                                df.Properties.VariableNames=channel_names;

                                writetable(df,save_path+"\"+sujeto+".csv");
                            end
                        end
                    end
                    if flag == 1
                        fprintf('Progress: %.2f%%\r',round(p,2));
                        disp("SUCCESSFULLY SAVED: "+sujeto);
                    end
                end
                if p == 100.00
                    clc;
                end
            end
        end
    end
end


function extractTimes(path, S_Dir, build_path, flag)
    % S_dir: structure of directories comprised by dir_MAT and dir_EEG
    dir_MAT = S_Dir{1};% structure of MAT directories and info for (.mat)
    dir_EEG = S_Dir{2};% structure of EEG directories and info for (.edf)

    plcb = [1,4,7,9,11,13,14,15,17,18,19,24,25,27,28,31,35,40,41, ...
                42,43,44,47,48,49,50,52,61,63,65,69,105];

    % For each directory with time instant
    for i = 1:length(dir_EEG.instante)
        instante = string(dir_EEG.instante{i});
        % For every group inside each time instant
        for j = 1:length(dir_EEG.grupo{i})
            grupo = string(dir_EEG.grupo{i}{j});
            % Seleccionamos todos los archivos.mat dentro del grupo
            estimulos_mat = dir_MAT.datos{i}{j};
            % For every subject inside each group
            for k = 1:length(dir_EEG.datos{i}{j})
                sujeto = dir_EEG.datos{i}{j}{k};

                p = k/length(dir_EEG.datos{i}{j})*100;

                % Build path for specific search (.edf)
                search_path = path+"\EEG\"+instante+"\"+grupo+"\"+sujeto;

                % Build path for saving P600 stimulis
                save_path = build_path+"\"+instante+"\"+grupo;

                % Check if the substring ODDBALL exists in the string
                if contains(sujeto, 'ODDBALL')

                    % Mostramos el path de cada archivo.edf
                    disp('ANALIZANDO: ');
                    disp(search_path);

                    % Removemos la extension .edf
                    sujeto = erase(sujeto, ".edf");
                    % Separamos el nombre del archivo usando del="_"
                    fields_sujeto = strsplit(sujeto,"_");
                    % Convertimos el id en double format
                    id_sujeto = str2double(fields_sujeto{4});
                    % Convertimos el nº de ODDBALL en double format
                    n_oddball = str2double(fields_sujeto{7});

                    % Read .edf file and gather all info about timings.
                    % onset_array: instants for stimulis in seconds
                    [S,onset_array] = readEDF(search_path);
                    
                    % Contamos cuantos estimulos hay registrados en (.edf)
                    n_timings = length(onset_array);

                    % Get channel names
                    channel_names = fieldnames(S);

                    % Turn S into EEG matrix (m x n), where m and n are
                    % nChannels and values
                    X = StructToDouble(S);
                    
                    % No X,Y,Z channels
                    X = X(1:end-3,:);
                    channel_names = channel_names(1:end-3);
                    % Add 'type' field for DataFrame
                    channel_names = [{'comp'}; channel_names];
                    % Add 'comp' field for DataFrame
                    channel_names = [{'type'}; channel_names];
                    % Add 'grupo' field for DataFrame
                    channel_names = [{'id'}; channel_names];
                    % Add 'id' field for DataFrame
                    channel_names = [{'grupo'}; channel_names];
                    % Add 'instante' field for DataFrame
                    channel_names = [{'instante'}; channel_names];
                    % Add 'n_test' field for DataFrame
                    channel_names = [{'n_test'}; channel_names]';

                    for n = 1:length(estimulos_mat)
                        % estimulos{n} = POST_Control_64_FLL_1 (.mat)
                        fields_mat = strsplit(estimulos_mat{n}, "_");
                        % ID subject = 64
                        id_mat = str2double(fields_mat{3});
                        % ODDBALL Nº: 1
                        n_test_mat = str2double(fields_mat{5});
                        
                        % Check if (.mat) and (.edf) matches
                        if n_test_mat == n_oddball && id_mat == id_sujeto

                            % Si hay > 90 timings y no esta vacio, then
                            if ~isempty(n_timings) && n_timings > 90

                                % Ruta hacia archivos (.mat)
                                path_mat = path+"\MATLAB\"+instante+"\"+...
                                           grupo+"\"+estimulos_mat{n};
    
                                % Cargamos informacion del tipo de estimulo
                                % 'std' o 'tgt' en (.mat)
                                data = load(path_mat);

                                % Extraemos cuales son std[0] y tgt[1]
                                stimulis = binarize(data.task.timing);
                                
                                % Si tenemos menos timings en (.edf) que
                                % marcados en (.mat), entonces:
                                if n_timings <= length(stimulis)
                                    stimulis = stimulis(1:n_timings);
                                end

                                % std: seleccionamos los idx donde es 0
                                std_onset = onset_array(stimulis==0);
                                % tgt: seleccionamos los idx donde es 1
                                tgt_onset = onset_array(stimulis==1);
                                
                                % Extraemos los tiempos de respuesta
                                std_time = avg_time(X,std_onset, 500);
                                tgt_time = avg_time(X,tgt_onset, 500);

                                comps = ["P1" "N1" "P2" "N2" "P3" ...
                                         "P1" "N1" "P2" "N2" "P3"]';

                                std = repStr("std",size(std_time,1));
                                tgt = repStr("tgt",size(tgt_time,1));

                                % Concatenamos std_time y tgt_time 
                                X_w = [std_time; tgt_time];
                                types = [std; tgt];

                                % Construimos arrays de informacion
                                n_test = repStr(n_oddball,size(types,1));

                                ins = repStr(instante,size(types,1));

                                % Controlamos el grupo placebo
                                if ismember(id_sujeto, plcb)
                                    gr = repStr("PLCB",size(types,1));
                                else
                                    gr = repStr(grupo,size(types,1));
                                end

                                id = repStr(id_sujeto,size(types,1));

                                % Build DataFrame
                                df = [array2table(n_test) ...
                                      array2table(ins) ...
                                      array2table(gr) ...  
                                      array2table(id) ...
                                      array2table(types) ...
                                      array2table(comps) ...
                                      array2table(X_w)];
                                
                                % Set column names
                                df.Properties.VariableNames=channel_names;

                                writetable(df,save_path+"\"+sujeto+".csv");
                            end
                        end
                    end
                    if flag == 1
                        fprintf('Progress: %.2f%%\r',round(p,2));
                        disp("SUCCESSFULLY SAVED: "+sujeto);
                    end
                end
                if p == 100.00
                    clc;
                end
            end
        end
    end
end


function time = avg_time(X, onset, fs)

    % X: EEG matrix (m x n), where m and n are nChannels and values. 

    % Get dimension of EEG matrix
    [m_EEG,~] = size(X);

    % onset: array of timings in seconds. Sample rate is fs=500Hz, so we
    % need to turn seconds in signal datapoints as follows: onset*fs.
    onset = onset*fs;
   
    % window = [l_lim, r_lim) = [-200ms, +700ms) = [-100, 350) puntos
    l_lim = 0.2*fs;% 200ms -> 0.2s*500Hz -> 100 puntos
    r_lim = 0.7*fs;% 700ms -> 0.7s*500Hz -> 350 puntos
    
    time = [];

    % For every channel, then:
    for m = 1:m_EEG
        % Apply Pass-Band Filter on every channel X(m,:)
        X_ch = applyPassBand(X(m,:), fs, {1, 30});

        w = [];
        for n = 1:length(onset)

            % Adelantamos el instante de disparo 25 puntos -> 50 ms
            trigger = int32(onset(n)) - 25;

            % post-pre = 450 puntos = 900ms = [-200ms, +700ms)
            pre =  trigger - l_lim;
            post = trigger + r_lim;

            % Hay un sujeto que posee el primer timing muy pronto y no se
            % puede extraer una baseline, por ende, pasamos al siguiente.
            % filename = PRE_Exp_8_MAG_ODDBALL_1.edf
            if pre > 1

                % baseline -> [pre, pre+50) puntos -> [-200, -100) ms 
                baseline = mean(X_ch(pre:pre+50-1));% len(bsln) = 50 puntos

                % X_ch -> [pre, post-1) = [-100, 350) = 450 puntos
                w = [w; X_ch(pre:post-1)-baseline];
            end

        end
        
        
        % Compute median window [-200ms, +700ms) = [-100, +350) puntos
        window_ch = median(w);
        
        P1 = window_ch(140+1:165);% 25 puntos

        %P1_smooth = smoothdata(P1, 'movmean', 20); 
        t_P1 = (41+find(P1 == max(P1)))/500; % (1/500)*1000 s -> ms


        N1 = window_ch(165+1:200);

        %N1_smooth = smoothdata(N1, 'movmean', 20); 
        t_N1 = (66+find(N1 == min(N1)))/500; % (1/500)*1000 s -> ms
        
        P2 = window_ch(200+1:250);

        %P2_smooth = smoothdata(P2, 'movmean', 20); 
        t_P2 = (101+find(P2 == max(P2)))/500; % (1/500)*1000 s -> ms

        N2 = window_ch(250+1:330);

        %N2_smooth = smoothdata(N2, 'movmean', 20); 
        t_N2 = (151+find(N2 == min(N2)))/500; % (1/500)*1000 s -> ms

        P3 = window_ch(330+1:400);

        % Suavizar la señal antes filtro de media movil
        %P3_smooth = smoothdata(P3, 'movmean', 20); 
        t_P3 = (231+find(P3 == max(P3)))/500; % (1/500)*1000 s -> ms

        time = [time; t_P1 t_N1 t_P2 t_N2 t_P3];% (20x5)
    end
    time = time';% (5x20)
end









function out = repStr(str, N)
    out = repmat(str, N, 1);
end

function out = binarize(char_values)
    % Extraemos cuales son std[0] y tgt[1]
    out = zeros(1,99);
    % Structure of structures, loop for getting values
    for i = 1:99
        if strcmp(char_values(i).sndname, 'tgt')
            out(i) = 1;
        end
    end
end



function window_matrix = averageWindows(X, onset, fs, flag)

    % X: EEG matrix (m x n), where m and n are nChannels and values. 

    % Get dimension of EEG matrix
    [m_EEG,~] = size(X);

    % onset: array of timings in seconds. Sample rate is fs=500Hz, so we
    % need to turn seconds into signal datapoints as follows: onset*fs.
    onset = onset*fs;
    
    % window = [l_lim, r_lim) = [-200ms, +700ms) = [-100, 350) puntos
    l_lim = 0.2*fs;% 200ms -> 0.2s*500Hz -> 100 puntos
    r_lim = 0.7*fs;% 700ms -> 0.7s*500Hz -> 350 puntos
    
    % Matrix (20 channels x 450 points)
    window_matrix = [];
    
    % For every channel, then:
    for m = 1:m_EEG
        % Apply Pass-Band Filter on every channel X(m,:)
        X_ch = applyPassBand(X(m,:), fs, {1, 30});
        
        % Por cada canal, guardamos los estimulos
        w = [];
        
        timer = 0;
        % For every timing, then:
        for n = 1:length(onset)

            % Adelantamos el instante de disparo 25 puntos -> 50 ms
            trigger = int32(onset(n)) - 25;

            % post-pre = 450 puntos = 900ms = [-200ms, +700ms)
            pre =  trigger - l_lim;% trigger - 100 puntos
            post = trigger + r_lim;% trigger + 350 puntos

            % Hay un sujeto que posee el primer timing muy pronto y no se
            % puede extraer una baseline, por ende, pasamos al siguiente.
            % filename = PRE_Exp_8_MAG_ODDBALL_1.edf
            if pre > 1

                % baseline -> [pre, pre+50) puntos -> [-200, -100) ms 
                bsln = mean(X_ch(pre:pre+50-1));% len(bsln) = 50 puntos

                % X_ch -> [pre, post-1) = [-100, 350) = 450 puntos
                w = [w; X_ch(pre:post-1)-bsln];


            end
            
        end

        if flag == "mean"
            % Para cada punto en la ventana, calculamos:
            sigma = std(w);% Desviacion estandar de cada punto
            mu = mean(w);% Media de cada punto
            % Calculamos limites superior e inferior.
            up = mu+2*sigma;
            down = mu-2*sigma;

            window_ch = [];
            % Iteramos sobre cada columna
            for col = 1:size(w, 2)  % Usamos size(w, 2) para N cols
                % Seleccionamos los valores de la columna 'col' dentro del 
                % intervalo [down, up] --> Logical index
                indices = w(:, col) >= down(col) & w(:, col) <= up(col);  
                
                % Calculamos la media de los valores de la columna que
                % cumplen la condicion
                value = mean(w(indices, col)); 
                
                % Almacenamos el resultado en 'window_ch'
                window_ch = [window_ch, value];
            end
            

            % Compute average window [-200ms, +700ms) = [-100, +350) puntos
            %window_ch = mean(w);
        else
            % Compute median window [-200ms, +700ms) = [-100, +350) puntos
            window_ch = median(w);

        end

        % Control de posibles errores
        if length(w) ~= (post-pre)
            error("El tamaño de ventana deseado y " + ...
                  "calculado no coinciden." + newline + "Average " + ...
                  "window size (" + num2str(length(window_ch)) + ...
                  ") must be (" + num2str(post - pre) + ")");
        end

        % Save average or median window for every channel
        window_matrix = [window_matrix; window_ch];% (20 x 450)
    end

end


%% Get ERP components from window matrix (20 x 450) 

function comp_matrix = getComponent(X, comp)
    
    % X: EEG matrix (m x n), where m and n are nChannels and values. 

    if comp == "baseline"
        % baseline = [l_lim, r_lim] = [-200ms,-100ms] = [+1, +50] puntos
        l_lim = 0+1;% -200ms -> 1 punto
        r_lim = 50;% -100ms -> 0.1s*500Hz -> 50 puntos
    elseif comp == "pre-trigger"
        % pre-trigger = [l_lim, r_lim] = (-100ms,0ms] = (+50, +100] puntos
        l_lim = 50+1;% -100ms 
        r_lim = 100;% 0ms 
    elseif comp == "post-trigger"
        % post-trigger = [l_lim, r_lim] = (0ms,80ms] = (+100, +140] puntos
        l_lim = 100+1;
        r_lim = 140;
    elseif comp == "P1"
        % window = [l_lim, r_lim] = (+80ms, +130ms] = (+140, +165] puntos
        l_lim = 140+1;
        r_lim = 165;
    elseif comp == "N1"
        % window = [l_lim, r_lim] = (+130ms, +200ms] = (+165, +200] puntos
        l_lim = 165+1;
        r_lim = 200;
    elseif comp == "P2"
        % window = [l_lim, r_lim] = (+200ms, +300ms] = (+200, +250] puntos
        l_lim = 200+1;
        r_lim = 250;
    elseif comp == "N2"
        % window = [l_lim, r_lim] = (+300ms, +360ms] = (+250, +330] puntos
        l_lim = 250+1;
        r_lim = 330;        
    elseif comp == "P3"
        % window = [l_lim, r_lim] = (+360ms, +600ms] = (+330, +400] puntos
        l_lim = 330+1;
        r_lim = 400;
    elseif comp == "all"
        % window = [l_lim, r_lim] = [-200ms, +700ms) = [+1, +450] puntos
        l_lim = 0+1;
        r_lim = 450;
    else
        error("No existe ese nombre de componente.")
    end

    % Get dimension of EEG matrix
    [m_EEG,~] = size(X);

    % Matrix (20 channels x N puntos), where N depends on comp
    comp_matrix = [];

    % For every channel, then:
    for m = 1:m_EEG

        % Select every component
        w_comp = X(m, l_lim:r_lim);
        
        % Save component window of every channel
        comp_matrix = [comp_matrix; w_comp];% (20 x N)
    end
end