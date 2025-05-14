"""
Title: FEATURE EXTRACTION AND CSV FILES BUILDING FROM ODDBALL DATA
Author: Alberto Peces Contreras
Date: 18/11/2024
"""

import os
import pandas as pd

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

# Especifica la ruta del archivo CSV
post_ctrl_path = 'E:\\TFM\\PREPROCESADO\\TIME\\POST\\CONTROL'
post_expt_path = 'E:\\TFM\\PREPROCESADO\\TIME\\POST\\EXP'

pre_ctrl_path = 'E:\\TFM\\PREPROCESADO\\TIME\\PRE\\CONTROL'
pre_expt_path = 'E:\\TFM\\PREPROCESADO\\TIME\\PRE\\EXP'

seg_ctrl_path = 'E:\\TFM\\PREPROCESADO\\TIME\\SEGUIMIENTO\\CONTROL'
seg_expt_path = 'E:\\TFM\\PREPROCESADO\\TIME\\SEGUIMIENTO\\EXP'

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

save_path = "E:/TFM/CLUSTERING_ALL_ICA_by_segments/ODDBALL/"
os.makedirs(save_path, exist_ok=True)  # exist_ok=True evita un error si el directorio ya existe

# GUARDAR: times
df_std.to_csv(save_path+"/"+'times_std.csv', sep=';', index=False)
df_tgt.to_csv(save_path+"/"+'times_tgt.csv', sep=';', index=False)

print("(-) DONE!")