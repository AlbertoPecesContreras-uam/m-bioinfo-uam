import os
import pandas as pd
import numpy as np

class Utils:
    
    def select_common_ids(df_0, df_1): # CHECK --> OK
        
        """
        Dados dos dataframes, selecciona los sujetos comunes en ambos
        dataframes. Devuelve como output dos dataframes que poseen los
        mismos sujetos.
        """

        # Rseteamos index de ambos dataframes, evitamos posibles errores
        df_0 = df_0.reset_index(drop=True)
        df_1 = df_1.reset_index(drop=True)

        n_test = [1, 2]
        # Declaramos las lsitas que guardaran las filas de sujetos comunes
        list_df_0 = []
        list_df_1 = []
        for n in n_test:
            df_0_n = df_0[df_0["n_test"]==n]
            df_1_n = df_1[df_1["n_test"]==n]

            ids_0 = df_0_n["id"].unique()
            ids_1 = df_1_n["id"].unique()

            for id_0 in ids_0:
                if id_0 in ids_1:
                    list_df_0.append(df_0_n[df_0_n["id"]==id_0])
                    list_df_1.append(df_1_n[df_1_n["id"]==id_0])


        return pd.concat(list_df_0,axis=0),pd.concat(list_df_1,axis=0)
    
    def remove_subjects(df, ids_for_removal):
        
        df.reset_index(drop=True, inplace=True)

        for idx, value in enumerate(df["id"]):
            if value in ids_for_removal:
                df = df.drop(idx)

        # After removing subjects, reset index column in order to avoid errors
        df.reset_index(drop=True, inplace=True)

        return df

    def select_channels(df, stimuli_channels, N):
        """
        Introducimos el dataframe con todas las caracteristicas
        y canales. Seleccionamos las caracteristicas de canales
        especificos para analizar un tipo de estimulo.

        N: es el numero de columnas que debemos ignorar
        """
        idx = []
        for iCont, col in enumerate(df.columns):
            ch = col.split("_")[0]
            if ch in stimuli_channels:
                idx.append(iCont)
        return list(range(0,N))+idx
    
    def get_scatter_distribution(df, N): # CHECK --> OK

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

        return dict(zip(list(df["id"]),diff.tolist()))
    
    def get_distance_between_subjects(df_0, df_1, N): # CHECK --> OK

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
            # Fila del id_i en df_1
            i_1 = list(df_1["id"]).index(id_i)

            # Calculamos distancia entre los sujetos coincidentes en ambas matrices
            distance = np.sqrt(np.sum((X0[i_0,:]-X1[i_1,:])**2))

            # Guardamos la distancia en lista d
            d = np.append(distance, d)
            # Guardamos id en lista id_i
            id.append(id_i)

        return dict(zip(id,d))

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
    

    def get_counts(df, ids, byGroup = False): # CHECK --> OK
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
        df.reset_index(drop=True)

        # Seleccionamos el conteo por grupos
        if byGroup:
            # Seleccionamos grupos
            df_ctrl = df[df["grupo"]=="Control"]
            df_exp = df[df["grupo"]=="Exp"]
            # Contadores inicializados a cero
            n_ctrl = 0
            n_exp = 0
            # Por cada id suma 1 en el grupo correspondiente
            for id in ids:
                if id in list(df_ctrl["id"]):
                    n_ctrl+=1
                if id in list(df_exp["id"]):
                    n_exp+=1

            return [n_ctrl, n_exp] 
        else:
            # Seleccionamos los instantes del dataframe
            df_pre = df[df["instante"]=="PRE"]
            df_post = df[df["instante"]=="POST"]
            df_seg = df[df["instante"]=="SEG"]
            
            # Contadores inicializados a cero
            n_pre = 0
            n_post = 0
            n_seg = 0

            # Por cada id suma 1 en el instante correspondiente
            for id in ids:
                if id in list(df_pre["id"]):
                    n_pre+=1
                if id in list(df_post["id"]):
                    n_post+=1
                if id in list(df_seg["id"]):
                    n_seg+=1

            return [n_pre, n_post, n_seg]
    

    def avg_window(df, comp, channels, flag = "mean"):# CHECK --> OK

        """
        Example:
        df = df_tgt_pre_ctrl, all CTRL group from PRE instant. This
        DataFrame contains average windows from every channel and 
        component. DataFrame information:
        
        - Header with info: ['n_test', 'instante', 'grupo', 'id', 'type', 'comp']
        - Channels: ['P8', 'T8', 'F8', 'F4', 'C4', 'P4', 'Fp2', 'Fp1', 'Fz', 'Cz', 'O1', 'Oz', 'O2', 'Pz', 'P3', 'C3', 'F3', 'F7', 'T7', 'P7']
        - Components: ['all' 'baseline' 'P1' 'N1' 'P2' 'N2' 'P3']
        
        For channel "ch", we have the following windows:
        
        - all: [-200, +700] ms
        - baseline: [-200, 0] ms
        - P1: [+80, +130] ms
        - N1: [+130, +200] ms
        - P2: [+200, +300] ms
        - N2: [+300, +360] ms
        - P3: [+360, +600] ms
        
        However, some subjects have 2 different tests:
        - n_test = 1 = ODDBALL_1
        - n_test = 2 = ODDBALL_2

        Take all subjects with a specific component and channel, then compute average window.
        Consideration: it takes into account both n_tests.
        """

        # Necesitamos conocer la longitud de la ventana de cada componente
        id0 = df[df['n_test']==1]['id'].unique()[0]
        nCols = df[(df['n_test']==1)&(df['id']==id0)&(df['comp']==comp)].shape[0]
        
        # Guardara la suma de ventanas para ambas pruebas -> [1, 2]
        m = np.zeros((0,nCols), dtype=float)

        # Por cada prueba oddball -> [1, 2]
        # Por cada id, entonces:
        for (n,id), sub_df in df.groupby(["n_test","id"]):
            
            for ch in channels:
                # Seleccionamos la ventana del componente y canal correspondiente
                w_0 = sub_df[sub_df['comp']==comp][ch].to_numpy()

                # Guardamos los valores de ventana en filas
                m = np.vstack([m, w_0])

        # Calculamos la ventana promedio
        if flag == "mean":
            out = np.mean(m, axis = 0)
        # Calculamos la ventana mediana 
        else:
            out = np.median(m, axis = 0)
             
        return out

    # Funcion para aplicar un filtro pasa banda
    def bandpass_filter(data, params, order=4):

        lowcut, highcut, fs = params['lowcut'],params['highcut'],params['fs']

        from scipy.signal import butter, filtfilt
        nyquist = 0.5 * fs  # Frecuencia de Nyquist
        low = lowcut / nyquist  # Frecuencia minima normalizada
        high = highcut / nyquist  # Frecuencia maxima normalizada
        b, a = butter(order, [low, high], btype='band')  # Diseño del filtro
        return filtfilt(b, a, data)  # Aplicar el filtro

    # Funcion para test no parametricos pareados (Wilcoxon) entre PRE y POST
    def get_stats(df, features, components):

        from scipy.stats import wilcoxon, norm
        from statsmodels.stats.multitest import multipletests

        list_df_stats = []
        # Por cada grupo, then:
        for gr in ["CONTROL", "PLCB", "EXP"]:   

            # Seleccionamos sujetos comunetes en ambos instantes PRE y POSt
            df_pre, df_post = Utils.select_common_ids(df[(df["instante"]=="PRE")&(df["grupo"]==gr)], 
                                                    df[(df["instante"]=="POST")&(df["grupo"]==gr)])
            # Por cada caracteristica
            for f_i in features:
                # Inicializamos DataFrame para guardar stats
                df_stats = pd.DataFrame([],columns=["grupo","comp","feat","contrast","cohens_d","p_val","adj_p_val"])
                # Por cada componente
                for comp in components:

                    mu_pre = df_pre[df_pre["comp"]==comp][f_i].mean()
                    mu_post = df_post[df_post["comp"]==comp][f_i].mean()

                    # Test Wilcoxon para muestras pareadas
                    t,p = wilcoxon(df_pre[df_pre["comp"]==comp][f_i], df_post[df_post["comp"]==comp][f_i])

                    # Obtener el estadístico Z
                    z = norm.ppf(p / 2) * -1  # Transformación de p a Z

                    # Calcular r
                    N = len(df_pre[df_pre["comp"]==comp][f_i])  # Número de pares
                    r = z / np.sqrt(N)

                    # Show contrast info and cohen's d value
                    if mu_pre < mu_post:
                        contrast = "PRE < POST"
                        cohen_d = round((2 * r) / np.sqrt(1 - r**2),2)
                    else:
                        contrast = "PRE > POST"
                        cohen_d = round((2 * r) / np.sqrt(1 - r**2),2)

                    # Guardamos stats
                    df_stats.loc[len(df_stats)] = [gr, comp, f_i, contrast, cohen_d, p, "NaN"]
                # Realizamos correccion FDR por los N tests realizados 
                _, p_adjusted, _, _ = multipletests(df_stats["p_val"], method='fdr_bh')
                # Guardamos p-valores ajustados
                df_stats["adj_p_val"] = p_adjusted
                # Guardamos un DataFrame por caracteristica
                list_df_stats.append(df_stats)

        #df_stats_by_components = pd.concat(list_df_stats)
        return pd.concat(list_df_stats)
    
    def merge_or_diff(df1, df0, nCol, option = "diff", by_axis=1):
        if option == "diff":
            return pd.concat([df1.iloc[:, :nCol], df1.iloc[:, nCol:] - df0.iloc[:, nCol:]], axis=1)
        else:
            return pd.concat([df1, df0.iloc[:, nCol:]], axis=by_axis).reset_index(drop=True)
        
    def group_by(df, col, val): return [df[df[col]==v].reset_index(drop=True) for v in val]
    


    def average_by_feature(data, features):
        out = []
        for df in data:
            results = []
            # Iterar sobre cada grupo (combinación única de n_test e id)
            for (n_test, id_, g), sub_df in df.groupby(["n_test", "id", "grupo"]):
                # Calcular la media de las columnas ["feat0","feat1",...,"featN"] para el grupo actual
                mean_values = sub_df[features].mean().values
                # Guardar id, mean y var en la lista de resultados
                results.append({**{ "id": id_, "grupo":g}, **dict(zip(features, mean_values))})
            out.append(pd.DataFrame(results))
        return out

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
        print("# --------- MENU --------- #")
        print()
        print("1) CLUSTERING_ALL_ICA: OA")
        print("2) CLUSTERING_ALL_ICA: OC")
        print("3) CLUSTERING_ONE_ICA: OA")
        print("4) CLUSTERING_ONE_ICA: OC")
        print()
        option = int(input("Here:"))
        print()
        print("1) ALL")
        print("2) Delta (0.5 - 4) Hz")
        print("3) Theta (4 - 8) Hz")
        print("4) Alpha (8 - 12) Hz")
        print("5) Beta (12 - 30) Hz")
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