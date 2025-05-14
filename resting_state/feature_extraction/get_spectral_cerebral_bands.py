"""
Title: SPECTRAL DOMAIN AND CSV FILES BUILDING 
Author: Alberto Peces Contreras
Date: 27/10/2024
Revision: OK
"""

import os
import pandas as pd
import numpy as np
from scipy.signal import spectrogram, welch

from features import Features
from subject import Subject


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
    That information is saved in Subject() class where it will be classified
    by instant (PRE,POST,SEG), and groups (EXP, CTRL).

    [OUTPUT]

    Subject() object class loaded with information.
    """
    test_types = ["OA", "OC"]
    N = len(list_csv_paths)

    for i in range(N):
        if flag != 0:
            print("-) EXTRACTING INFO PROCESS STATUS: starting...")
            print()
        for icont, csv_path  in enumerate(list_csv_paths[i]):

            # INFO about Subject
            filename = csv_path.split("\\")[-1]
            filename_split = filename.split(".")[0].split("_")

            if flag != 0:
                print("-) Processing:",filename)

            # Turn .csv into DataFrame 
            df = pd.read_csv(csv_path, header=0)

            # Turn dataframe into matrix of type=float
            matrix_csv = df.values.astype(float)

            # For tipo in [OA, OC]:
            for tipo in test_types:
                if tipo in filename_split:
                    # Save data in Subject() class
                    class_subject.set_tipo_test(tipo)
                    class_subject.set_filename(filename)
                    class_subject.set_instante(filename_split[1])
                    class_subject.set_grupo(filename_split[2])
                    class_subject.set_id(filename_split[3])
                    class_subject.set_channels(df.columns.tolist())
                    class_subject.set_signals(matrix_csv)

            if flag != 0:
                p = 100*round((icont+1)/len(list_csv_paths[i]),2)
                print()
                print("\t·) Progress:",p,"%")
                print("-) EXITO:",filename)
                print()

        if flag != 0:            
            print()
            print("-) EXTRACTING INFO PROCESS STATUS: FINISHED --> OK!")
            print()

        print()
        #os.system('cls')

    return 


def process_matrix(M, info, lim, by_segments = False):# CHECK --> OK

    """
    [INPUT]
    M: [matrix_97_PRE_CTRL,matrix_86_PRE_CTRL,...,matrix_23_PRE_CTRL,matrix_12_PRE_CTRL]
    M: list of matrices type=float. Each inner matrix has (m x n) size, where:
    m and n are signal values and electrodes/channels/components, respectively.
    - m: filas -> signal values
    - n: columnas -> signal type -> signal_1,...,signal_n

    info: [id_pre, grupo_pre], where id_pre and grupo_pre are lists with respective ID and
    group for each subject/matrix.

    [FUNCTION]
    Extrae caracteristicas y add columnas de ID y grupo (info).

    [OUTPUT]
    Devuelve un dataframe con las caracteristicas extraidas por
    signal "n" en matriz M[i], es decir, el dataframe resultante
    tiene tantas filas como signals/columnas en M[i] y columnas como
    caracteristicas especificadas en "Features.extract()".
    """

    IDs,grupos,channels = info[0],info[1],info[2]# [["PRE","PRE",...,"PRE"],["CTRL","CTRL",...,"CTRL"]]

    df_feats_list = []

    # For each i in M list, then:
    for i in range(len(M)):
        
        m = M[i]# Select each matrix m = M[i]

        cols = m.shape[1]# Get columns/signals number

        feats = []

        # For each signal, then:
        for j in range(cols):
            
            frequencies, psd = welch(m[:,j], fs=256, nperseg=2*256, noverlap=(2*256)//2, window='hann', scaling="density")

            # Select positions from frequencies of interest (alpha, beta, ...)
            idx_band = np.where((frequencies>=lim[0]) & (frequencies<=lim[1]))[0]
            if by_segments == False:
                # Select all amplitude values of that band, turn them into flat vector, calculate power
                # for each value. Later, that power vector is processed and some features are extracted.
                # Finally, those features are saved in feats list. They are features from ONE signal.
                # The process is repeated for all signals/columns in that matrix m = M[i]
                #feats.append(Features.extract(10*np.log10(Sxx[idx_band,:]), "spectral"))
                #feats.append(Features.extract(Sxx[idx_band,:], "spectral"))

                # Las caracteristicas de una distribucion logaritmica no representa la disperision real de los datos.
                # Emplear Welch o Sxx da los mismos resultados. Es más fácil de explicar Welch.
                # psd: es una curva positiva de potencias en [V^2/Hz], debemos integrar no promediar o sumar.
                
                # La función spectrogram devuelve una matriz NxM, donde N son las frecuencias y M las ventanas 
                # temporales de 2 segundos. Por cada ventana de dos segundos, se calcula la densidad espectral 
                # de potencia por cada frecuencia. Debemos integrar la curva de densidad de potencia de cada ventana.

                pwd = np.trapz(psd[idx_band], frequencies[idx_band])

                feats.append({"power": pwd})
                
            else:
                feats_by_segments = []
                # type(Sxx[idx_band,:]) = numpy.ndarray (80 x 256)
                # Sxx = numpy.ndarray (257 x 256)

                # n_segments = Sxx.shape[1]# 256 segmentos of 2s

                # for k in range(n_segments):
                #     # feats_by_segments.append(Features.extract(10*np.log10(Sxx[idx_band,k]), 256))
                #     # Las caracteristicas de una distribucion logaritmica no representa la dispersion real de los datos.
                #     feats_by_segments.append(Features.extract(Sxx[idx_band, k], frequencies[idx_band], "spectral"))# dict = {"mean":10,"std":0.5,...}


                df_feats_by_segments = pd.DataFrame(feats_by_segments)
                # To keep results as dictionary format
                mean_features = df_feats_by_segments.median().to_dict()

                feats.append(mean_features)
                
                feats_by_segments.clear()# Free memory

        # Turn that feats list (full of feature dictionaries) into dataframe.
        df_feats = pd.DataFrame(feats)
        # Add ID and Group columns in order to identify all those values for each subject
        # Same value of ID and Group is repeated for all rows. Those features are only from
        # one subject.
        df_feats["grupo"] = grupos[i]
        df_feats["id"] = IDs[i]
        df_feats["ch"] = channels[i]

        # A dataframe for each subject, where features were extracted from his signals, is saved.
        df_feats_list.append(df_feats)
        print(round(((i+1)/len(M))*100,2),"%")

        feats.clear()# Free memory
        
    # Concatenate all dataframes by rows in order to have all subjects in a single .csv file.
    return pd.concat(df_feats_list, ignore_index=True)


def combine_signals(instante, ID, grupo, ch, signals):
    dfs = [pd.DataFrame(signals[i], columns=ch[i]).assign(grupo=grupo[i], id=ID[i], instante=instante) for i in range(len(grupo))]
    return pd.concat(dfs, ignore_index=True)

def cut_extreme_picks(params, signals_instante):

    for m in signals_instante:
        # Comprueba si los valores en las filas cumplen: punto < lower_limit y punto > upper_limmit
        # Si algun punto de cada fila cumple alguna condicion, entonces se coloca lower_limit o upper_limit
        # Se actualiza toda la matriz sin bucles adicionales
        m[:] = np.clip(m, params["lower_limit"], params["upper_limit"])

    return signals_instante

# Especifica la ruta del archivo CSV

"""
post_ctrl_path = 'E:\\TFM\\preprocesado_borrar\\EEG\\POST\\CONTROL'
post_expt_path = 'E:\\TFM\\preprocesado_borrar\\EEG\\POST\\EXP'

pre_ctrl_path = 'E:\\TFM\\preprocesado_borrar\\EEG\\PRE\\CONTROL'
pre_expt_path = 'E:\\TFM\\preprocesado_borrar\\EEG\\PRE\\EXP'

seg_ctrl_path = 'E:\\TFM\\preprocesado_borrar\\EEG\\SEGUIMIENTO\\CONTROL'
seg_expt_path = 'E:\\TFM\\preprocesado_borrar\\EEG\\SEGUIMIENTO\\EXP'



post_ctrl_path = 'E:\\TFM\\PREPROCESADO_ALL_PCA\\EEG\\POST\\CONTROL'
post_expt_path = 'E:\\TFM\\PREPROCESADO_ALL_PCA\\EEG\\POST\\EXP'

pre_ctrl_path = 'E:\\TFM\\PREPROCESADO_ALL_PCA\\EEG\\PRE\\CONTROL'
pre_expt_path = 'E:\\TFM\\PREPROCESADO_ALL_PCA\\EEG\\PRE\\EXP'

seg_ctrl_path = 'E:\\TFM\\PREPROCESADO_ALL_PCA\\EEG\\SEGUIMIENTO\\CONTROL'
seg_expt_path = 'E:\\TFM\\PREPROCESADO_ALL_PCA\\EEG\\SEGUIMIENTO\\EXP'

"""

post_ctrl_path = 'E:\\TFM\\PREPROCESADO_ALL_ICA\\EEG\\POST\\CONTROL'
post_expt_path = 'E:\\TFM\\PREPROCESADO_ALL_ICA\\EEG\\POST\\EXP'

pre_ctrl_path = 'E:\\TFM\\PREPROCESADO_ALL_ICA\\EEG\\PRE\\CONTROL'
pre_expt_path = 'E:\\TFM\\PREPROCESADO_ALL_ICA\\EEG\\PRE\\EXP'

seg_ctrl_path = 'E:\\TFM\\PREPROCESADO_ALL_ICA\\EEG\\SEGUIMIENTO\\CONTROL'
seg_expt_path = 'E:\\TFM\\PREPROCESADO_ALL_ICA\\EEG\\SEGUIMIENTO\\EXP'



all_path = [post_ctrl_path, 
            post_expt_path, 
            pre_ctrl_path, 
            pre_expt_path, 
            seg_ctrl_path, 
            seg_expt_path]


# Every time script runs, previous information is removed
class_subject = Subject()


# Declaramos el tipo de pruebas que podemos recopilar
tipo_test = ["OA", "OC"]
print("¿Que tipo de prueba necesitas {0:'OA',1:'OC'}?")
i = int(input("Here:"))

# Recopilamos las rutas de todos los archivos .csv
all_csv = get_files(all_path)# OK

# Loading all information in Subject() class object
read_file_csv(all_csv, 1)# OK

# Seleccionamos las posiciones de la prueba de interes tipo_test[0] = OJOS ABIERTOS
idx_test = np.where(class_subject.tipo_test == tipo_test[i])

# Values are saved as they were read and loaded in read_file_csv()
instantes = class_subject.instante[idx_test]# ["POST","POST",...,"PRE","PRE",...,"SEG","SEG"]
grupos = class_subject.grupo[idx_test]# ["CONTROL","CONTROL",...,"EXP","EXP"]
ids = class_subject.id[idx_test]# [97,86,...,23,12]
channels = [class_subject.channels[i] for i in idx_test[0]]
signals = [class_subject.signals[i] for i in idx_test[0]]# [matrix_97,matrix_86,...,matrix_23,matrix_12]
 
# Seleccionamos las posiciones de los archivos de cada instante
idx_pre = np.where(instantes == "PRE")
idx_post = np.where(instantes == "POST")
idx_seg = np.where(instantes == "SEG")

# Seleccionamos los grupos de cada instante
grupo_pre = [grupos[i] for i in idx_pre[0]]
grupo_post = [grupos[i] for i in idx_post[0]]
grupo_seg = [grupos[i] for i in idx_seg[0]]

# Seleccionamos IDs de cada instante
id_pre = ids[idx_pre]
id_post = ids[idx_post]
id_seg = ids[idx_seg]

# Seleccionamos el nombre de los canales que componen las signals de cada sujeto
ch_pre = [channels[i] for i in idx_pre[0]]
ch_post = [channels[i] for i in idx_post[0]]
ch_seg = [channels[i] for i in idx_seg[0]]

# Seleccionamos las signals de cada sujeto de cada instante
signals_pre = [signals[i] for i in idx_pre[0]]
signals_post = [signals[i] for i in idx_post[0]]
signals_seg = [signals[i] for i in idx_seg[0]]



# Concatenamos todas las señales en un solo DataFrame y lo guardamos
s_pre = combine_signals('PRE', id_pre, grupo_pre, ch_pre, signals_pre)

s_post = combine_signals('POST', id_post, grupo_post, ch_post, signals_post)

s_seg = combine_signals('SEG', id_seg, grupo_seg, ch_seg, signals_seg)


# Guardamos todas las señales en una matriz 's' (N records, 36451) y
# extraemos parametros para remover picos por posicion.
# params = [mu - 3*sigma, mu + 3*sigma] 
s = np.vstack([np.vstack([df_i[ch].values for id, df_i in df_s.groupby('id') for ch in df_s.columns[:-3]]) for df_s in [s_pre, s_post, s_seg]])

mu = np.mean(s, axis = 0)[:, None]# (36451, 1)
sigma = np.std(s, axis = 0)[:, None]# (36451, 1)

# mu = np.mean(s)# (1, 1)
# sigma = np.std(s)# (1, 1)

params = {"upper_limit": mu + 3*sigma,
          "lower_limit": mu - 3*sigma}

# Removemos picos bruscos en las señales
signals_pre = cut_extreme_picks(params, signals_pre)
signals_post = cut_extreme_picks(params, signals_post)
signals_seg = cut_extreme_picks(params, signals_seg)



# Sample rate -> Data was downsampled from 500 to 256 Hz in preprocessing step
#fs = 256  # Hz

# Definir las bandas cerebrales con sus rangos de frecuencia
bands = {
    "all": [1, 30],
    "delta": [1, 4],
    "theta": [4, 8],
    "alpha": [8, 12],
    "beta": [12, 30],
    "d_t_a": [1, 12],
    "t_a": [4, 12],
    "a_b": [8, 30],
}

for key in bands.keys():

    print("-) "+key+": working...")

    # Extraemos señales de todos los sujetos y convertimos a dataframes
    df_pre = process_matrix(signals_pre, [id_pre, grupo_pre, ch_pre], bands[key], by_segments=False)# OK
    df_post = process_matrix(signals_post, [id_post, grupo_post, ch_post], bands[key], by_segments=False)# OK
    df_seg = process_matrix(signals_seg, [id_seg, grupo_seg, ch_seg], bands[key], by_segments=False)# OK

    # Especifica ruta para guardado
    if i == 0:
        save_path = "E:/TFM/CLUSTERING_ALL_ICA_by_segments/OA/spectral_domain/"+key
        os.makedirs(save_path, exist_ok=True)  # exist_ok=True evita un error si el directorio ya existe

        # GUARDAR: POST CONTROL Y EXPERIMENTO
        df_pre.to_csv(save_path+"/"+'PRE_OA.csv', sep=';', index=False)
        df_post.to_csv(save_path+"/"+'POST_OA.csv', sep=';', index=False)
        df_seg.to_csv(save_path+"/"+'SEG_OA.csv', sep=';', index=False)

    else:
        save_path = "E:/TFM/CLUSTERING_ALL_ICA_by_segments/OC/spectral_domain/"+key

        os.makedirs(save_path, exist_ok=True)  # exist_ok=True evita un error si el directorio ya existe

        # GUARDAR: POST CONTROL Y EXPERIMENTO
        df_pre.to_csv(save_path+"/"+'PRE_OC.csv', sep=';', index=False)
        df_post.to_csv(save_path+"/"+'POST_OC.csv', sep=';', index=False)
        df_seg.to_csv(save_path+"/"+'SEG_OC.csv', sep=';', index=False)

    print("\t·) DONE!")

# LIBERAMOS MEMORIA
class_subject.clear()

"""
# Ahora tienes las matrices separadas por banda
delta_matrix = dict_pre_bands["Delta"]
theta_matrix = dict_pre_bands["Theta"]
alpha_matrix = dict_pre_bands["Alpha"]
beta_matrix = dict_pre_bands["Beta"]

# Ejemplo de cómo ver una de las matrices filtradas (por ejemplo, Delta)
print("Delta Band Matrix:")
print(delta_matrix[0][:5, :])  # Mostrar las primeras 5 filas y todas las columnas
"""

