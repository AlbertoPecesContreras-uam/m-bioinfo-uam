%% FUNCTION
function buildDirectory(folder, ext, s)
    % arg_1 = folder which will save all subfolders in path
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    
    if ~exist(folder+"\"+ext, 'dir')
        mkdir(folder+"\"+ext);
    end

    for i = 1:length(s.instante)
        instante = s.instante{i};
        if ~exist(folder+"\"+ext+"\"+instante, 'dir')
            mkdir(folder+"\"+ext+"\"+instante);
        end

        for j = 1:length(s.grupo{i})
            grupo = s.grupo{i}{j};
            if ~exist(folder+"\"+ext+"\"+instante+"\"+grupo, 'dir')
                mkdir(folder+"\"+ext+"\"+instante+"\"+grupo);
            end
        end
    end
end