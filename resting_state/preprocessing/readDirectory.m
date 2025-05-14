% Title: Read Data Directory (readDirectory.m)
% Author: Alberto Peces Contreras
% Date: 05/10/2024
% Working Time: 4-5 hours

%% FUNCTIONS

% MAIN FUNCTION
function struct_data = readDirectory(path)

    % This function allows to build an information structure that comprises 
    % folders, subfolders, and files from every patient.
    % arg1 = path where files are found
    
    % Define a structure for EEG o MAT path
    struct_data = struct();
    
    % Fulfill the structure with all directory information and filenames 
    % for each patient
    
    % Get a cell array with all folders within EEG o MAT
    cell_instante = getFolders(path);
    struct_data.instante = cell_instante;
    % Get a cell array with all subfolders
    cell_grupo = getSubFolders(path);
    struct_data.grupo = cell_grupo;
    % Get filenames from all patients 
    cell_datos = getFiles(path, struct_data);
    struct_data.datos = cell_datos;

end

% First Search Function
function folders = getFolders(path)

    % This function allows to get all folders in a given path
    % arg1 = it is the path in format -> "C:\Users\...\...\...\Folder"

    folders = {};
    
    dir_path = dir(path);% Structure with directory info

    for i = 1:length(dir_path)
    
        if ~isstring(dir_path(i))
            % Select name from each folder 
            n_folder = dir_path(i).name;
        else
            n_folder = dir_path(i);
        end

        % Ignore folders '.' and '..'
        if ~strcmp(n_folder, '.') && ~strcmp(n_folder, '..')
            folders{end+1} = n_folder;
        end
    end
end

% Second Search Function
function subfolders = getSubFolders(path)
        
    % This function allows to gather all subfolders in a given path
    % arg1 = path where we want to extract subdirectories
    % Format "C:\Users\...\...\...\Folder"

    dir_path = dir(path);% Strcuture with directory info

    subfolders = {};

    for j = 1:length(dir_path)
        
        % Select name from each folder 
        n_subfolder = dir_path(j).name;
        
        % Ignore folders '.' and '..'
        if ~strcmp(n_subfolder, '.') && ~strcmp(n_subfolder, '..')
            subfolders{end+1} = getFolders(path+'\'+n_subfolder);
        end
    end
end

% Third Search Function
function all_files = getFiles(path, s)
    
    % This function allows retrieving all the files located in a 
    % specified path through a structure 's', which contains as field 
    % values the names of the subfolders found within the given path.
    % arg1 = the path of the folder in question, e.g., ".\DATOS\EEG"
    % arg2 = the structure where the values of its fields refer to 
    % the subfolders to search for in the path.

    all_files = {};% Cell array [1x3] (1 x 3 instantes)

    n_instantes = length(s.instante);% n = length(post, pre, seguimiento)
    
    % For each folder of instants
    for i = 1:n_instantes
        % Number of groups
        n_grupos = s.grupo(i);

        files = {};% Cell array [1x2] (1 x 2 grupos)

        % For each group in every instant
        for j = 1:length(n_grupos{1})
            % Cell array individual con size variable
            group_file = {};
            
            % Group path
            files_path = path+'\'+s.instante(i)+'\'+n_grupos{1}{j};
            
            % From every group path, we get filenames of subjects
            filenames = dir(files_path);
            
            % For each subject
            for k = 1:length(filenames)
                filename = filenames(k).name;% Filename
                
                % Avoid files '.', '..' and 'desktop.ini', among others
                if ~strcmp(filename, '.') && ~strcmp(filename, '..') 
                    if ~strcmp(filename, 'INCIDENCIAS') && ...
                       ~strcmp(filename, 'desktop.ini')
                        % Save every file in a cell array
                        group_file{end+1,1} = filename;
                    end
                end
            end

            % The cell array with all files from each group will be saved 
            % in other cell array called 'files' [1x2]
            files{1,j} = group_file;
            
            % Confirmation message
            disp(".\"+s.instante(i)+"\"+n_grupos{1}(j)+" -> ... -> DONE");
        end

        % Save all subjects in a final cell array [1x3]
        all_files{i} = files;
    end
end





