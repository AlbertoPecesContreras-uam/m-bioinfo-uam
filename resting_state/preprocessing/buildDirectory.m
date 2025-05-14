% Title: Build Especified Directory (buildDirectory.edf)
% Author: Alberto Peces Contreras
% Date: 05/10/2024
% Working Time: 30 min

%% FUNCTION
function buildDirectory(folder, path, s)
    % arg_1 = folder which will save all subfolders in path
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    
    if ~exist(path, 'dir')
        mkdir(path);
    end

    for i = 1:length(s.instante)

        if ~exist(path+"\"+s.instante{i}, 'dir')
            mkdir(path+"\"+s.instante{i});
        end

        for j = 1:length(s.grupo{i})
            if ~exist(path+"\"+s.instante{i}+"\"+s.grupo{i}{j}, 'dir')
                mkdir(path+"\"+s.instante{i}+"\"+s.grupo{i}{j});
            end
        end
    end
end