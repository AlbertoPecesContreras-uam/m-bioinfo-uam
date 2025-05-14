"""
Title: FEATURE EXTRACTION AND CSV FILES BUILDING FROM ODDBALL DATA
Author: Alberto Peces Contreras
Date: 18/11/2024
"""

import os
import numpy as np
import pandas as pd

from features import Features



def get_files(path_in): # CHECK --> OK
    """
    [INPUT]
    path_in: list with several paths in string format
    path_in = ["E:\\...\\POST\\CTRL","E:\\...\\POST\\EXP",
               "E:\\...\\PRE\\CTRL","E:\\...\\PRE\\EXP",
               "E:\\...\\SEG\\CTRL","E:\\...\\SEG\\EXP"]
    [FUNCTION]
    For each path, we insert an empty list into file_dir.
    For each path, we list all files and folders.
    file_path = path_in[i]+filename

    [OUTPUT]
    [["E:\\...\\POST\\CTRL\\file1.txt", "E:\\...\\POST\\CTRL\\file2.csv"],
    ["E:\\...\\POST\\EXP\\exp1.docx"],
    ["E:\\...\\PRE\\CTRL\\pre1.log", "E:\\...\\PRE\\CTRL\\pre2.txt"],
    ["E:\\...\\PRE\\CTRL\\pre5.log", "E:\\...\\PRE\\CTRL\\pre3.txt"],
    ["E:\\...\\SEG\\CTRL\\seg1.png", "E:\\...\\SEG\\CTRL\\seg2.jpg"],
    ["E:\\...\\SEG\\EXP\\exp2.mp4"]]
    """
    # N of paths for searching and saving files
    N = len(path_in)

    file_dir = []
    for i in range(N):
        file_dir.append([])
        # List files in path_in[i]
        for name in os.listdir(path_in[i]):
            # Join filename to path_in[i]
            file_path = os.path.join(path_in[i], name)
            # Select only files
            if not os.path.isdir(file_path):
                file_dir[i].append(file_path)

    return file_dir

def read_file_csv(list_csv_paths, flag = 0):# CHECK --> OK 

    """
    [INPUT]

    [["E:\\...\\POST\\CTRL\\file1.txt", "E:\\...\\POST\\CTRL\\file2.csv"],
    ["E:\\...\\POST\\EXP\\exp1.docx"],
    ["E:\\...\\PRE\\CTRL\\pre1.log", "E:\\...\\PRE\\CTRL\\pre2.txt"],
    ["E:\\...\\PRE\\CTRL\\pre5.log", "E:\\...\\PRE\\CTRL\\pre3.txt"],
    ["E:\\...\\SEG\\CTRL\\seg1.png", "E:\\...\\SEG\\CTRL\\seg2.jpg"],
    ["E:\\...\\SEG\\EXP\\exp2.mp4"]]

    if flag!=0, some info will be displayed.
    
    [FUNCTION]

    From a list of sublists, each inner list has paths from .csv files.
    From each filename, ID and GROUP parameters are extracted.
    Information is read and concatenated from every .csv file.

    [OUTPUT]
    A DataFrame with all files (.csv) concatenated.
    """

    list_df = []

    N = len(list_csv_paths)

    for i in range(N):
        if flag != 0:
            print()
            print("-) EXTRACTING INFO PROCESS STATUS: starting...")
            print()
        # For every (.csv), then
        for icont, csv_path in enumerate(list_csv_paths[i]):
            
            # INFO about Subject
            filename = csv_path.split("\\")[-1]

            if flag != 0:
                print("Processing:",filename)

            # Turn .csv into DataFrame  
            # Columns: [n_test, instante, grupo, id, type, comp, P8, ..., F3, F7, T7, P7]
            # Header with info: ['n_test', 'instante', 'grupo', 'id', 'type', 'comp']
            # Channels: ['P8', 'T8', 'F8', 'F4', 'C4', 'P4', 'Fp2', 'Fp1', 'Fz', 'Cz', 'O1', 'Oz', 'O2', 'Pz', 'P3', 'C3', 'F3', 'F7', 'T7', 'P7']
            # Components: ['all' 'baseline' 'P1' 'N1' 'P2' 'N2' 'P3']
            df = pd.read_csv(csv_path, header=0)

            # Save every DataFrame 
            list_df.append(df)
            
            if flag != 0:
                p = icont/len(list_csv_paths[i])
                print("Percentage:",round(p*100, 2),"%")
    print()
    # Concatenate all DataFrames -> All subjects (.csv) are concatenated in df_out
    df_out = pd.concat(list_df, ignore_index=True)

    return df_out

def extract_features(df, n_tests, channels, components, flag = "channels_as_cols"):# CHECK --> OK

    """
    Extraemos caracteristicas de media y varianza para cada sujeto
    con sus respectivos canales y componentes.

    Cada columna es un vector que contiene la ventana del estimulo
    organizado por canales. La columna "comp" posee informacion sobre 
    el tipo de componente ['all' 'baseline' 'P1' 'N1' 'N2' 'P3'].

    Cada componente posee su ventana promedio y canal, extraemos 
    caracteristicas de media y varianza.
    """

    # Seleccionamos los IDs unicos
    id_unique = df["id"].unique()

    # Channels as columns
    if flag == "channels_as_cols":
        # Guarda los DataFrames de caracteristica de cada sujeto
        feat_list = []
        # Definimos zeros matrix que contendra las caracteristicas
        feat_matrix = np.zeros((2, len(channels)), dtype=float)# (N features x len(channels))

        # Por cada sujeto, then:
        for icont, id in enumerate(id_unique):
            p = icont/len(id_unique)
            print("Percentage:",round(p*100, 2),"%")
            for n in n_tests:
                df_id_n = df[(df["n_test"]==n)&(df["id"]==id)]
                if not df_id_n.empty:
                    # Por cada componente, then:
                    for comp in components:
                        # Por cada canal, then:
                        for j,ch in enumerate(channels):
                            # Seleccionamos el sujeto 'id',componente 'comp' y canal 'ch'
                            X = df_id_n[(df_id_n["comp"]==comp)][ch]
                            # Calculamos media de actividad EEG
                            mu = np.mean(X)
                            # Calculamos desviacion de actividad EEG
                            sigma = np.std(X)
                            # Vector de caracteristicas
                            feats = [mu, sigma]
                            # Guardamos [mu, sigma] en cada columna de feat_matrix
                            feat_matrix[:,j] = feats
                        # DataFrame donde repetimos el 'n' len(features) veces
                        df_n = pd.DataFrame([n]*len(feats), columns=["n_test"])# (2 x 1)
                        # DataFrame donde repetimos el 'id' len(features) veces
                        df_id = pd.DataFrame([id]*len(feats), columns=["id"])# (2 x 1)
                        # DataFrame donde repetimos el 'comp' len(features) veces
                        df_comp = pd.DataFrame([comp]*len(feats), columns=["comp"])# (2 x 1)
                        # DataFrame donde indicamos el nombre de las caracteristicas
                        df_feat = pd.DataFrame(["mean","std"],columns=["feat"])# (2 x 1)
                        # DataFrame donde guardamos los valores de las caracteristicas
                        df_values = pd.DataFrame(feat_matrix, columns=channels)# (2 x 20)
                        # Guardamos la matriz de caracteristicas en una lista para cada sujeto
                        feat_list.append(pd.concat([df_n, df_id, df_comp, df_feat, df_values], axis = 1))# [(2 x 23),(2 x 23),...,(2 x 23)]

        # Este DataFrame posee las caracteristicas media y varianza para cada sujeto con sus respectivos canales y componentes
        df_features = pd.concat(feat_list, ignore_index=True)# Nos faltan los campos con informacion sobre el instante y grupo
        print()

        grupo  = []# Esta lista guardara la columna con informacion de los grupos
        instante = []# Esta lista guardara la columna con informacion de los instantes

        for id in id_unique:
            # Seleccionamos primer valor en el campo 'grupo' de un sujeto especifico 
            grupo_0 = df[df["id"]==id]["grupo"].iloc[0]# ["CONTROL","CONTROL",...,"CONTROL"] -> se repite lo mismo varias veces
            # Seleccionamos primer valor en el campo 'instante' de un sujeto especifico 
            instante_0 = df[df["id"]==id]["instante"].iloc[0]# ["PRE","PRE",...,"PRE"] -> se repite lo mismo varias veces
            # Seleccionamos las veces que aparece ese sujeto en el otro DataFrame
            N = len(df_features[df_features["id"]==id])
            # Actualizamos las listas para construir columnas del DataFrame faltantes
            grupo = grupo + [grupo_0]*N
            instante = instante + [instante_0]*N
        
        # Construimos en DataFrame la lista grupo
        df_grupo = pd.DataFrame(grupo, columns=["grupo"])
        # Construimos en DataFrame la lista instante
        df_instante = pd.DataFrame(instante, columns=["instante"])

        # Actualizamos el DataFrame con las columnas que le faltaban
        df_features = pd.concat([df_features["n_test"], df_instante, df_grupo, df_features.iloc[:,1:]], axis=1)

    # Features as columns
    else:
        df_features = pd.DataFrame([],columns=["n_test","instante","grupo","id","comp","ch","mean","std"])
        # Por cada sujeto, then:
        for icont, id in enumerate(id_unique):
            p = icont/len(id_unique)
            print("Percentage:",round(p*100, 2),"%")
            for n in n_tests:
                df_id_n = df[(df["n_test"]==n)&(df["id"]==id)]
                if not df_id_n.empty:
                    instante = df_id_n["instante"].unique()[0]
                    grupo = df_id_n["grupo"].unique()[0]
                    # Por cada componente, then:
                    for comp in components:
                        # Por cada canal, then:
                        for j,ch in enumerate(channels):
                            # Seleccionamos el sujeto 'id',componente 'comp' y canal 'ch'
                            X = df_id_n[(df_id_n["comp"]==comp)][ch]
                            # Calculamos media de actividad EEG
                            mu = np.mean(X)
                            # Calculamos desviacion estandar de actividad EEG
                            sigma = np.std(X)
                            # Guardamos [mu, sigma] en cada columna de feat_matrix
                            df_features.loc[len(df_features)] = [n, instante, grupo, id, comp, ch, mu, sigma]

    return df_features


def reduce_stimulis(df_in):

    """
    Seleccionamos todas las ventanas "std" y "tgt" para cada canal
    y calculamos sus respectivas ventanas promedio. Pasamos de 100
    a 2 ventanas. Tendremos dos columnas promedio para estimulo 
    std y tgt.
    """

    instante = str(df_in.iloc[:,0].unique()[0])
    id = int(df_in.iloc[:,1].unique()[0])
    grupo = str(df_in.iloc[:,2].unique()[0])
    ch = str(df_in.iloc[:,3].unique()[0])

    new_cols = ["instante", "id", "grupo", "ch", "std", "tgt"]

    idx_std = []
    idx_tgt = []

    columns = list(df_in.columns)

    for col in columns:
        if col.find("std") != -1:
            std_i = columns.index(col)
            idx_std.append(std_i)
        if col.find("tgt") != -1:
            tgt_i = columns.index(col)
            idx_tgt.append(tgt_i)
    
    # Mean by rows
    series_std = df_in.iloc[:,idx_std].mean(axis=1)
    series_tgt = df_in.iloc[:,idx_tgt].mean(axis=1)


    series_values = [instante, id, grupo, ch, series_std, series_tgt]
    
    return pd.DataFrame(dict(zip(new_cols, series_values)))




def rearrange_features(df_in, stimuli_in, byChannel = False):# CHECK --> OK

    """
    Reagrupamos las caracteristicas, es decir, colocamos las
    caracteristicas extraidas de cada canal en las columnas.
    Podemos juntar todas las medias, medianas, ..., de cada
    canal o colocar el mismo canal y diferentes caracteristicas. 
    """

    # Seleccionamos el tipo de estimulo de interes: "std" o "tgt"
    df_in = df_in[df_in["stimuli"]==stimuli_in]
    # Seleccionamos los canales unicos
    channels = list(df_in.iloc[:,3].unique())

    # Seleccionamos las caracteristicas que hemos extraido por canal
    features = list(df_in.columns)[6:]

    new_cols = []
    values = []
    if byChannel is False:
        # Por cada caracteristica: mean, median, std, ...
        for f_i in features:
            # Por cada canal: P8, Fp1, Fp2, ...
            for ch_i in channels:
                df_ch_i = df_in[df_in["ch"]==ch_i]
                new_cols.append(ch_i+"_"+f_i)# P8_mean, Fp1_mean, ...
                values.append(list(df_ch_i[f_i]))# 
    else:
        for ch_i in channels:
            df_ch_i = df_in[df_in["ch"]==ch_i]
            for f_i in features:
                new_cols.append(ch_i+"_"+f_i)
                values.append(list(df_ch_i[f_i]))       

    df_info = pd.DataFrame({"instante":list(df_ch_i.iloc[:,0]),
                            "id":list(df_ch_i.iloc[:,1]),
                            "grupo":list(df_ch_i.iloc[:,2]),
                            "type":list(df_ch_i.iloc[:,4])})
    
    return pd.concat([df_info, pd.DataFrame(dict(zip(new_cols,values)))],axis=1)

# Especifica la ruta del archivo CSV
post_ctrl_path = 'E:\\TFM\\PREPROCESADO\\MAT_N_ESTIMULO\\POST\\CONTROL'
post_expt_path = 'E:\\TFM\\PREPROCESADO\\MAT_N_ESTIMULO\\POST\\EXP'

pre_ctrl_path = 'E:\\TFM\\PREPROCESADO\\MAT_N_ESTIMULO\\PRE\\CONTROL'
pre_expt_path = 'E:\\TFM\\PREPROCESADO\\MAT_N_ESTIMULO\\PRE\\EXP'

seg_ctrl_path = 'E:\\TFM\\PREPROCESADO\\MAT_N_ESTIMULO\\SEGUIMIENTO\\CONTROL'
seg_expt_path = 'E:\\TFM\\PREPROCESADO\\MAT_N_ESTIMULO\\SEGUIMIENTO\\EXP'


"""
post_ctrl_path = 'E:\\TFM\\preprocesado_borrar\\MAT\\POST\\CONTROL'
post_expt_path = 'E:\\TFM\\preprocesado_borrar\\MAT\\POST\\EXP'

pre_ctrl_path = 'E:\\TFM\\preprocesado_borrar\\MAT\\PRE\\CONTROL'
pre_expt_path = 'E:\\TFM\\preprocesado_borrar\\MAT\\PRE\\EXP'

seg_ctrl_path = 'E:\\TFM\\preprocesado_borrar\\MAT\\SEGUIMIENTO\\CONTROL'
seg_expt_path = 'E:\\TFM\\preprocesado_borrar\\MAT\\SEGUIMIENTO\\EXP'

post_ctrl_path = 'E:\\TFM\\PREPROCESADO_ALL_ICA\\MAT\\POST\\CONTROL'
post_expt_path = 'E:\\TFM\\PREPROCESADO_ALL_ICA\\MAT\\POST\\EXP'

pre_ctrl_path = 'E:\\TFM\\PREPROCESADO_ALL_ICA\\MAT\\PRE\\CONTROL'
pre_expt_path = 'E:\\TFM\\PREPROCESADO_ALL_ICA\\MAT\\PRE\\EXP'

seg_ctrl_path = 'E:\\TFM\\PREPROCESADO_ALL_ICA\\MAT\\SEGUIMIENTO\\CONTROL'
seg_expt_path = 'E:\\TFM\\PREPROCESADO_ALL_ICA\\MAT\\SEGUIMIENTO\\EXP'

"""
all_path = [post_ctrl_path, 
            post_expt_path, 
            pre_ctrl_path, 
            pre_expt_path, 
            seg_ctrl_path, 
            seg_expt_path]

# Recopilamos las rutas de todos los archivos .csv
all_csv = get_files(all_path)

# Read and load all information from every (.csv) file
df = read_file_csv(all_csv, 1)


# Select names of channels
channels = df.columns[6:].to_list()# ['P8', 'T8', 'F8', 'F4', 'C4', 'P4', 'Fp2', 'Fp1', 'Fz', 'Cz', 'O1', 'Oz', 'O2', 'Pz', 'P3', 'C3', 'F3', 'F7', 'T7', 'P7']
# Select info column names
info = df.columns[:6].to_list()# ['n_test', 'instante', 'grupo', 'id', 'type', 'comp']
# Select names of components
components = df["comp"].unique()# ['all' 'baseline' 'P1' 'N1' 'P2' 'N2' 'P3']
# Select number of tests --> ODDBALL Nº1 or Nº2
n_tests = df["n_test"].unique()# ['1', '2']

print("-) First view:")
print()
print(df.head())
print()

# Select DataFrame with windows of standard stimulus
df_std = df[df["type"] == "std"]
# Select DataFrame with windows of target stimulus
df_tgt = df[df["type"] == "tgt"]

print("-) STD view:")
print()
print(df_std.head())
print()
print("-) TGT view:")
print()
print(df_tgt.head())
print()
print("...")
print()

# Extract features from windows of standard stimulus
df_feats_std_pre = extract_features(df_std[df_std["instante"]=="PRE"], n_tests, channels, components, flag="features_as_columns")
df_feats_std_post = extract_features(df_std[df_std["instante"]=="POST"], n_tests, channels, components, flag="features_as_columns")
df_feats_std_seg = extract_features(df_std[df_std["instante"]=="SEGUIMIENTO"], n_tests, channels, components, flag="features_as_columns")
# Extract features from windows of target stimulus
df_feats_tgt_pre = extract_features(df_tgt[df_tgt["instante"]=="PRE"], n_tests, channels, components, flag="features_as_columns")
df_feats_tgt_post = extract_features(df_tgt[df_tgt["instante"]=="POST"], n_tests, channels, components, flag="features_as_columns")
df_feats_tgt_seg = extract_features(df_tgt[df_tgt["instante"]=="SEGUIMIENTO"], n_tests, channels, components, flag="features_as_columns")

# Concatenate features from every instant --> standard stimulus
df_feats_std = pd.concat([df_feats_std_pre, df_feats_std_post, df_feats_std_seg], ignore_index=True)
# Concatenate features from every instant --> target stimulus
df_feats_tgt = pd.concat([df_feats_tgt_pre, df_feats_tgt_post, df_feats_tgt_seg], ignore_index=True)

print("-) Features -> STD view:")
print()
print(df_feats_std.head())
print()
print("-) Features -> TGT view:")
print()
print(df_feats_tgt.head())
print()
print("...")
print()

save_path = "E:/TFM/CLUSTERING_ALL_ICA_by_segments/ODDBALL/"
os.makedirs(save_path, exist_ok=True)  # exist_ok=True evita un error si el directorio ya existe

# GUARDAR: estimulos
df_std.to_csv(save_path+"/"+'stimulus_std.csv', sep=';', index=False)
df_tgt.to_csv(save_path+"/"+'stimulus_tgt.csv', sep=';', index=False)
# GUARDAR: caracteristicas
df_feats_std.to_csv(save_path+"/"+'features_std.csv', sep=';', index=False)
df_feats_tgt.to_csv(save_path+"/"+'features_tgt.csv', sep=';', index=False)
print("(-) DONE!")