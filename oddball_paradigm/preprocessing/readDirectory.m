% Title: Read Data Directory (files.edf)
% Author: Alberto Peces Contreras
% Date: 05/10/2024
% Working Time: 4-5 hours

%% FUNCTIONS

% MAIN FUNCTION
function struct_data = readDirectory(path)

% Esta funcion nos permite construir una estructura con informacion de 
% carpetas, subcarpetas y archivos de cada paciente.
% arg1 = es el path donde se encuentran los datos

% Construimos una estructura para carpeta EEG o MAT
struct_data = struct();

% Construimos la estructura con toda la informacion de directorios y
% nombres de archivos por cada paciente

% Obtenemos una celda con todas las carpetas dentro de EEG o MAT
cell_instante = getFolders(path);
struct_data.instante = cell_instante;
% Obtenemos una celda con todas las subcarpetas
cell_grupo = getSubFolders(path);
struct_data.grupo = cell_grupo;
% Obtenemos todos los archivos de los pacientes 
cell_datos = getFiles(path, struct_data);
struct_data.datos = cell_datos;







%EEG = struct();
%MAT = struct();

%EEG_instante = getFolders(EEG_path);
%MAT_instante = getFolders(MAT_path);

%EEG_grupo = getSubFolders(EEG_path);
%MAT_grupo = getSubFolders(MAT_path);

%EEG_datos = getFiles(EEG_path, EEG);
%MAT_datos = getFiles(MAT_path, MAT);

%EEG.instante = EEG_instante;
%MAT.instante = MAT_instante;

%EEG.grupo = EEG_grupo;
%MAT.grupo = MAT_grupo;

%EEG_datos = getFiles(EEG_path, EEG);
%MAT_datos = getFiles(MAT_path, MAT);

%EEG.datos = EEG_datos;
%MAT.datos = MAT_datos;

end

% First Search Function
function folders = getFolders(path)

    % Esta funcion permite obtener todas las carpetas dado un path
    % arg1 = es el path en formato "C:\Users\...\...\...\Folder"

    folders = {};
    
    dir_path = dir(path);% Estructura con info del directorio

    for i = 1:length(dir_path)
    
        if ~isstring(dir_path(i))
            % Seleccionamos en cada capeta su campo nombre 
            n_folder = dir_path(i).name;
        else
            n_folder = dir_path(i);
        end

        % Ignoramos carpetas '.' y '..'
        if ~strcmp(n_folder, '.') && ~strcmp(n_folder, '..')
            folders{end+1} = n_folder;
        end
    end
end

% Second Search Function
function subfolders = getSubFolders(path)
        
    % Esta funcion nos permite recopilar las subcarpetas del path
    % introducido.
    % arg1 = path del cual queremos obtener sus subdirectorios
    % Formato "C:\Users\...\...\...\Folder"

    dir_path = dir(path);% Estructura con info del directorio

    subfolders = {};

    for j = 1:length(dir_path)
        
        % Seleccionamos en cada capeta su campo nombre 
        n_subfolder = dir_path(j).name;
        
        % Ignoramos carpetas '.' y '..'
        if ~strcmp(n_subfolder, '.') && ~strcmp(n_subfolder, '..')
            subfolders{end+1} = getFolders(path+'\'+n_subfolder);

        end
    end
end

% Third Search Function
function all_files = getFiles(path, s)

    % Esta funcion permite obtener todos los archivos que se encuentran en
    % determinado path mediante una estructura 's' que posee como valores 
    % de campo los diferentes nombres de las subcarpetas que se encuentran 
    % dentro de dicho path.
    % arg1 = es el propio path de la carpeta en cuestion ".\DATOS\EEG"
    % arg2 = es la estructura donde los valores de sus campos son 
    % referencias a las subcarpetas a buscar en el path.

    all_files = {};% Cell array [1x3] (1 x 3 instantes)

    n_instantes = length(s.instante);% n = length(post, pre, seguimiento)
    
    % Para cada carpeta de instantes
    for i = 1:n_instantes
        % Obtenemos el numero de grupos
        n_grupos = s.grupo(i);

        files = {};% Cell array [1x2] (1 x 2 grupos)

        % Para cada grupo en cada instante
        for j = 1:length(n_grupos{1})
            
            group_file = {};% Cell array individual con size variable
            
            % Ruta de cada grupo
            files_path = path+'\'+s.instante(i)+'\'+n_grupos{1}{j};
            
            % A partir de la ruta de cada grupo, obtenemos los nombres de
            % los archivos de los sujetos
            filenames = dir(files_path);
            
            % Para cada sujeto
            for k = 1:length(filenames)

                % Nombre del archivo
                filename = filenames(k).name;
                
                % Evitamos archivos '.', '..' y 'desktop.ini', entre otros
                if ~strcmp(filename, '.') && ~strcmp(filename, '..') 
                    if ~strcmp(filename, 'INCIDENCIAS') && ~strcmp(filename, 'desktop.ini')
                        % Guardamos cada archivo en un cell array
                        group_file{end+1,1} = filename;
                    end
                end
            end

            % El cell array con los archivos de cada grupo se guardan en
            % otro cell array llamado 'files' [1x2]
            files{1,j} = group_file;
            
            % Emitimos mensaje de confirmacion
            disp(".\"+s.instante(i)+"\"+n_grupos{1}(j)+" -> ... -> DONE");
        end

        % Guardamos todos los sujetos en un cell array final [1x3]
        all_files{i} = files;
    end
end





