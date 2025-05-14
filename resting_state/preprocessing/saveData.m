%% FUNCTION

function saveData(path, filename, struct_data)
    % Initialize an empty table
    table_data = struct();
    
    % Transpose each field's data (assuming each is a 1xN array)
    fields = fieldnames(struct_data);
    for i = 1:numel(fields)
        field_name = fields{i};
        table_data.(field_name) = struct_data.(field_name)';  % Transpose to Nx1
    end
    
    % Convert the structure to a table
    table = struct2table(table_data);
    l = length(filename);
    writetable(table, path+"\"+filename(1:l-4)+".csv");
end
%{
function saveData(path, s_dir, struct_data)
    for i = 1:length(s_dir.instante)
        for j = 1:length(s_dir.grupo{i})
            for k = 1:length(s_dir.datos{i}{j})
                l = length(s_dir.datos{i}{j}{k});
                table = struct2table(struct_data);
                writetable(table, path+"\"+s_dir.datos{i}{j}{k}(1:l-4)+".csv")
            end
        end
    end
end

%}
    