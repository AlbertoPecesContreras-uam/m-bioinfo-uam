import os
import pandas as pd
import numpy as np

class Utils:


    def read_file_csv(list_csv_paths): # CHECK --> OK
        df_list = []
        for csv_path  in (list_csv_paths):
            # Convertir el csv en un DataFrame 
            df = pd.read_csv(csv_path, header=0, sep=";")

            # INFO about Subject
            filename = csv_path.split("\\")[-1] # "POST_OA.csv" o "PRE_OA.csv" o "SEG_OA.csv" 
            # Add info in DataFrame
            df["instante"] = filename.split(".")[0].split("_")[0] # ["POST","OA"] o ["PRE","OA"] o ["SEG","OA"]
            # Save DataFrames
            df_list.append(df)
        return  pd.concat(df_list, ignore_index=True)

    
    def build_df(d_X_in, d_info_in, f_names_in, instante_in, channels):# CHECK --> OK
        """
        A partir de un diccionario, donde las claves y valores
        son los sujetos y caracteristicas, respectivamente, se 
        construye un dataframe.

        d_X_in = {'1': [[f1_0, f1_1, f1_2, ..., f1_22],
                        [f2_0, f2_1, f2_2, ..., f2_22,
                        [...],
                        [f10_0, f10_1, f10_2, ..., f10_22]]]}
        Un total de 10 caracteristicas por sujeto, pero cada sujeto
        tiene 19 componentes y, por ende, 19x10 valores.

        En el dataframe cada fila y columna es un sujeto y caracteristica, 
        respectivamente. Si cada sujeto tiene 19 componentes y se calcula
        la media para cada una de ellas, entonces tenemos 19 features que 
        representan la media, 19 medians, 19 std, ... etc.                
        """
        # Save colnames for rearranged dataframe
        cols = [f_names_in[i]+"_"+ch for i in range(len(f_names_in)) for ch in channels]

        # Build an empty dataframe
        df = pd.DataFrame(columns=cols)

        # Para cada ID(sujeto), en el mismo orden que en el diccionario
        for key in d_X_in.keys(): # 1,2,3,4,5,6,...,91
            # Concatenate all feature values in one row
            row = [f_i for features in d_X_in[key] for f_i in features]
            # Save those features in one row per subject
            df.loc[len(df)] = row

        df.insert(0,'instante', instante_in)# OK
        df.insert(1,'id', list(d_info_in.keys()))# OK
        df.insert(2,'grupo',list(d_info_in.values()))# OK

        return df
    
    def get_files(path_in): 
 
        """
        A partir de un path concreto "path_in", obtenemos y 
        guardamos todos los files.csv
        """

        file_dir = []
 
        # Listar archivos y carpetas en el directorio
        for name in os.listdir(path_in):
            file_path = os.path.join(path_in, name)
            if not os.path.isdir(file_path):
                file_dir.append(file_path)

        return file_dir

    
    def select_common_ids(df_0, df_1, sort_by = ["id","ch"]): # CHECK --> OK
        
        """
        Dados dos dataframes, selecciona los sujetos comunes en ambos
        dataframes. Devuelve como output dos dataframes que poseen los
        mismos sujetos.
        """

        # Rseteamos index de ambos dataframes, evitamos posibles errores
        df_0 = df_0.reset_index(drop=True)
        df_1 = df_1.reset_index(drop=True)

        # Seleccionamos los ids comunes de cada dataframe
        common_ids = set(df_0["id"]) & set(df_1["id"])

        df_0 = df_0[df_0["id"].isin(common_ids)].reset_index(drop = True)
        df_1 = df_1[df_1["id"].isin(common_ids)].reset_index(drop = True)

        # Ordenamos ambos DataFrames por "id" para asegurar que estén en las mismas filas
        df_0 = df_0.sort_values(by=sort_by).reset_index(drop=True)
        df_1 = df_1.sort_values(by=sort_by).reset_index(drop=True)

        return df_0,df_1
    
    def get_scatter_distribution(df, N, dict_format = True): # CHECK --> OK

        """
        Calculamos el centroide de un grupo y la distancia 
        con cada uno de sus puntos. Obtenemos distribución 
        de distancias. N es el numero de columnas que 
        debemos ignorar en el dataframe.

        output = {1: 12.7, 2: 10.4, ..., id_n: d_n}
        """

        X = df.iloc[:,N:].values.astype(float)

        centroide = np.mean(X, axis=0)# By columns

        diff = np.sqrt(np.sum((X-centroide)**2, axis=1))# By rows

        if dict_format:
            return dict(zip(list(df["id"]),diff.tolist()))
        else:
            return diff.tolist()
    
    def get_distance_between_subjects(df_0, df_1, N, f = [], dict_format = True): # CHECK --> OK

        """
        Introducimos dos dataframes: df_0 y df_1. Ambos deben
        tener sujetos comunes, es decir, los sujetos que 
        aparecen en df_0 están en df_1 y viceversa.

        N es el numero de columnas que debemos ignorar.

        Devuelve un diccionario, donde las keys son los ids unicos
        de los sujetos en ambos dataframes y los valores son sus 
        distancias entre par de instantes.

        output = {1: 15.4, 2: 20.4, ..., id_n: d_n}
        """

        X0 = df_0.iloc[:,N:].values.astype(float)
        X1 = df_1.iloc[:,N:].values.astype(float)

        ids_unique = list(df_0["id"].unique())# Da igual el dataframe que se seleccione

        # Numpy ndarray que almacenara las distancias entre cada par de sujetos
        d = np.array([],dtype=float)
        id = []
        
        # Iteramos por cada ID unico
        for id_i in ids_unique:

            # Fila del id_i en df_0
            i_0 = list(df_0["id"]).index(id_i) 

            if id_i in list(df_1["id"]):   
                # Fila del id_i en df_1
                i_1 = list(df_1["id"]).index(id_i)

                if len(f) != 0:
                    # Calculamos distancia entre los sujetos coincidentes en ambas matrices
                    distance = np.sqrt(np.sum((X0[i_0,f]-X1[i_1,f])**2))
                else:
                    # Calculamos distancia entre los sujetos coincidentes en ambas matrices
                    distance = np.sqrt(np.sum((X0[i_0,:]-X1[i_1,:])**2))

                # Guardamos la distancia en lista d
                d = np.append(distance, d)
                # Guardamos id_i en lista id
                id.append(id_i)

        if dict_format:
            return dict(zip(id,d))
        else:
            return d

    def get_summary(data_in, bLim = 0): # CHECK --> OK

        """
        Estadsitica descriptiva de una lista de valores:
        - Media, Mediana, minimo y maximo
        - Q1,Q2,Q3, low_lim, and up_lim
        """

        # Admite un diccionario o un array
        if isinstance(data_in, dict):
            values = list(data_in.values())
        else:
            values = data_in
        

        # Calcular los percentiles: Q1, Mediana (Q2), Q3
        Q1 = np.percentile(values, 25)  # Primer cuartil (25%)
        Q2 = np.percentile(values, 50)  # Mediana (50%)
        Q3 = np.percentile(values, 75)  # Tercer cuartil (75%)

        # Calcular el rango intercuartílico (IQR)
        IQR = Q3 - Q1

        # Calcular los límites de los bigotes
        lower_whisker = 0
        upper_whisker = np.mean(values)+3*np.std(values)#Q3 +1.5*IQR##

        if bLim == 1:

            out = {"min":np.min(values), 
                    "low_lim": lower_whisker,
                    "Q1":round(Q1,2), 
                    "Median":round(Q2,2), 
                    "Mean":round(np.mean(values),2), 
                    "Var":round(np.var(values),2),
                    "Q3":round(Q3,2), 
                    "up_lim":round(upper_whisker,2),
                    "Max":np.max(values)}
        else:
            out = {"min":np.min(values), 
                    "Q1":round(Q1,2), 
                    "Median":round(Q2,2), 
                    "Mean":round(np.mean(values),2), 
                    "Var":round(np.var(values),2),
                    "Q3":round(Q3,2), 
                    "Max":np.max(values)}
        
        return out
    
    def get_outliers(dict_values, up_lim): # CHECK --> OK

        """
        Calcula los outliers que superen cierto umbral.
        Guarda en una lista los id que debemos eliminar.
        """

        remove = []
        for key,value in dict_values.items():

            if value > up_lim:
                remove.append(key)

        return remove

    def get_counts(df, byGroup = False): # 
        """
        Realizamos un conteo de los sujetos, ya sea
        por grupos o por instantes. Esta funcion sirve
        para introducir los ids que se van a eliminar
        por ser outliers y te dice a que grupo/clase
        pertenecen. Por ello, debemos introducir una 
        lista ids que posee los sujetos que vamos a
        eliminar.
        """
        # Resetamos la columna index del dataframe
        df = df.reset_index(drop=True)

        # Seleccionamos el conteo por grupos
        if byGroup:
            return {g: len(df[df["grupo"].isin([g])]) for g in df["grupo"].unique()}
        else:
            return {i: len(df[df["instante"].isin([i])]) for i in df["instante"].unique()}
        
    def get_cohens_d(g1, g2, hedge = False): # CHECK --> OK
        """
        Calcula el size effect entre dos grupos.
        """
        # Calcular las medias y desviaciones estándar
        media1, media2 = np.mean(g1), np.mean(g2)
        std1, std2 = np.std(g1, ddof=1), np.std(g2, ddof=1)

        # Tamaños de las muestras
        n1, n2 = len(g1), len(g2)

        # Calcular la desviación estándar combinada (s_pooled)
        s_pooled = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        # Calcular el tamaño del efecto (Cohen's d)
        cohens_d = (media1 - media2) / s_pooled

        if hedge:
            """Calcula Hedges' g corrigiendo el sesgo por tamaño de muestra."""
            n1, n2 = len(g1), len(g2)
            correction = 1 - (3 / (4*(n1 + n2) - 9))
            cohens_d = cohens_d * correction

        return cohens_d

    # Eliminamos sujetos ruidosos
    def remove_subjects(df, ids_for_removal):
        """
        Creamos una mascara booleana True para los valores a eliminar.
        Invertimos la mascara, manteniendo solo las filas que no estan en
        ids_for_removal.
        Restablecemos el indice despues de eliminar filas. 
        """
        return df[~df["id"].isin(ids_for_removal)].reset_index(drop=True)

    # Añadimos el grupo placebo mediante sus IDs
    def rename_subjects(df, ids_for_rename):
        df.loc[df["id"].isin(ids_for_rename), "grupo"] = "Placebo"
        return df
    
    # Renombramos caracteristicas situadas a partir de columna N inclusive
    def rename_feats(df, nCol, val):
        cols = df.columns.to_list()
        for i in range(nCol, nCol+len(cols[nCol:])):
            cols[i] = cols[i]+"_"+val
        df.columns = cols
        return df
    
    # Reorganizamos las ultimas N columnas
    def reset_cols_order(df, N):
        # Identificar las últimas 3 columnas
        last_three_columns = df.columns[-N:][::-1]
        # Reorganizar las columnas: mover las últimas 3 al principio
        new_column_order = last_three_columns.tolist() + df.columns[:-N].tolist()
        df = df[new_column_order]
        return df
    
    # Añadimos columna si no existe
    def add_column(df, col_name, val, idx):

        if col_name not in df.columns:
            df.insert(idx, col_name, val)
        return df
    
    def merge_or_diff(df1, df0, nCol, option = "diff", by_axis=1):
        if option == "diff":
            return pd.concat([df1.iloc[:, :nCol], df1.iloc[:, nCol:] - df0.iloc[:, nCol:]], axis=1)
        else:
            return pd.concat([df1, df0.iloc[:, nCol:]], axis=by_axis).reset_index(drop=True)
        
    def group_by(df, col, val): return [df[df[col]==v].reset_index(drop=True) for v in val]


    
    # ----------------------
    # DISPLAY DATA STRUCTURE
    # ----------------------
    def display_data_structure(flag=0):

        print("-) DATOS:")
        if flag == 0:
            print("\t | MEAN | MEDIAN | VAR | STD | KURTOSIS | SKEWNESS | RANGE | ENERGY | DOMINANT_F | MAX_INTENSITY |")
            print("|--------|")
            for i in range(3):
                for j in range(3):
                    print(" S"+str(i+1))
                    print("|--------|")
        else:
            print("\t | mean_1 | mean_2 | ... | mean_N | median_1 | median_2 | ... | median_N | ... | max_intensity_N |")
            print("|--------|")
            for i in range(5):
                print(" S"+str(i+1))
                print("|--------|")
        return
    
    # -----------------
    # DISPLAY DATA MENU
    # -----------------
    def show_menu():
        option = int(input("Here:"))
        print()
        print("1) ALL")
        print("2) Delta (1 - 4) Hz")
        print("3) Theta (4 - 8) Hz")
        print("4) Alpha (8 - 12) Hz")
        print("5) Beta (12 - 30) Hz")
        print("6) Delta+Theta+Alpha (1 - 12) Hz")
        print("7) Theta+Alpha (4 - 12) Hz")
        print("8) Alpha+Beta (8 - 30) Hz")
        print()

        j = int(input("Here:"))
        return option,j
    
    # ---------------------------------------------
    # DISPLAY INFORMATION ABOUT GROUPS AND INSTANTS
    # ---------------------------------------------
    def get_info_distribution(df, byGroup = False, byInstant = False):
        N_exp = len(df[df["grupo"]=="Exp"]["id"].unique())
        N_ctrl = len(df[df["grupo"]=="Control"]["id"].unique())

        N_pre_exp = len(df[(df["instante"]=="PRE")&(df["grupo"]=="Exp")])
        N_pre_ctrl = len(df[(df["instante"]=="PRE")&(df["grupo"]=="Control")])

        N_post_exp = len(df[(df["instante"]=="POST")&(df["grupo"]=="Exp")])
        N_post_ctrl = len(df[(df["instante"]=="POST")&(df["grupo"]=="Control")])

        N_seg_exp = len(df[(df["instante"]=="SEG")&(df["grupo"]=="Exp")])
        N_seg_ctrl = len(df[(df["instante"]=="SEG")&(df["grupo"]=="Control")])

        if byGroup:
            print("# --------- Groups Distribution --------- #")
            print()
            print("(-) Nº EXP:",N_exp)
            print("(-) Nº CTRL:",N_ctrl)
            print()
        if byInstant:
            print("# --------- Instant Distribution --------- #")
            print()
            print("(-) Nº PRE:",N_pre_exp+N_pre_ctrl)
            print("    - Nº Exp:",N_pre_exp)
            print("    - Nº Ctrl:",N_pre_ctrl)
            print()
            print("(-) Nº POST:",N_post_exp+N_post_ctrl)
            print("    - Nº Exp:",N_post_exp)
            print("    - Nº Ctrl:",N_post_ctrl)
            print()
            print("(-) Nº SEG:",N_seg_exp+N_seg_ctrl)
            print("    - Nº Exp:",N_seg_exp)
            print("    - Nº Ctrl:",N_seg_ctrl)
        return
    
    def display_info_stats(df_stats_list):

        for i in range(len(df_stats_list)):
            x_ctrl,x_exp = df_stats_list[i]

            if i == 0 and len(df_stats_list[i])!=0:
                print("(-) PRE:")
                h_ctrl,val_ctrl = str(x_ctrl).split("\n")
                h_exp,val_exp = str(x_exp).split("\n")

                print(h_ctrl+"\t\t"+h_exp)
                print(val_ctrl+"\t\t"+val_exp)
                print()
            elif i == 1 and len(df_stats_list[i])!=0:
                print("(-) POST:")
                h_ctrl,val_ctrl = str(x_ctrl).split("\n")
                h_exp,val_exp = str(x_exp).split("\n")

                print(h_ctrl+"\t\t"+h_exp)
                print(val_ctrl+"\t\t"+val_exp)
                print()
            elif i == 2 and len(df_stats_list[i])!=0:
                print("(-) SEG:")

                h_ctrl,val_ctrl = str(x_ctrl).split("\n")
                h_exp,val_exp = str(x_exp).split("\n")

                print(h_ctrl+"\t\t"+h_exp)
                print(val_ctrl+"\t\t"+val_exp)
                print()
        return