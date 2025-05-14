import scipy.signal as signal
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Coherence:
    # Funcion para extraer coherencia dentro de un rango de frecuencias
    def extract_coherence_by_band(f, Cxy, band):
        band_frequencies = (f >= band[0]) & (f <= band[1])
        return f[band_frequencies], Cxy[band_frequencies]
    
    # Funcion para calcular matriz de coherencia a partir de DataFrame
    def compute_matrix(df, channels, comp, fs, nperseg, band_range):
        
        N = len(channels)# Number of channels

        # Diccionario que guardara las matrices de coherencia para cada sujeto
        dict_coherence = {}

        # Por cada sujeto en df, then:
        for id in df['id'].unique():
            # Agregamos nueva clave/id al diccionario
            dict_coherence[id] = {}

            # Por cada tipo de prueba, then:
            for n in [1, 2]:
                # Seleccionamos el tipo de prueba, sujeto y componente especifico
                df_n = df[(df['n_test']==n)&(df['id']==id)&(df['comp']==comp)]

                # Si existen datos, then:
                if not df_n.empty:

                    # Definimos matriz de coherencia vacia
                    m = np.zeros((N,N), dtype = float)

                    # Para cada combinacion 1 vs 1 entre canales, then:
                    for i in range(len(channels)):
                        for j in range(i, len(channels)):
                            # Seleccionamos la respuesta al estimulo en channel[i]
                            s_ch_i = df_n[channels[i]].to_numpy(float)
                            # Seleccionamos la repsuesta al estimulo en channel[j]
                            s_ch_j = df_n[channels[j]].to_numpy(float)

                            # Calculamos coherencia entre señales
                            f, Cxy = signal.coherence(s_ch_i, s_ch_j, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

                            # Seleccionamos los valores de coherencia en una banda especifica
                            band_frequencies, band_coherence = Coherence.extract_coherence_by_band(f, Cxy, band_range)

                            # Calculamos el promedio de los valores anteriores y rellenamos matriz
                            m[i, j] = np.mean(band_coherence)
                            m[j, i] = m[i, j]  # La matriz de coherencia es simetrica

                    # Actualizamos el diccionario introducido por sujeto 'id',
                    # con una clave 'n' y valor 'm' por el tipo de prueba y 
                    # matriz de coherencia, respectivamente.
                    dict_coherence[id][n] = m

        return dict_coherence

    
    def get_diff(dict_post, dict_pre): 
        # Calcula la diferencia entre dos diccionarios devueltos 
        # por 'compute_matrix()', donde ambos poseen las mismos 
        # sujetos y numero de pruebas.
        return [dict_post[id][i] - dict_pre[id][i] for id,n in dict_pre.items() for i in list(n.keys())]

    def check_consistency(dict_coherence_pre, dict_coherence_post):
        """
        Vamos a comprobar si tenemos los sujetos colocados en las mismas posiciones
        de ambos diccionarios y, así, poder trabajar con ellos adecuadamente.

        Para realizar contrastes, un requisito necesario es que cada punto posea una
        pareja. No podemos tener diccionarios con un numero de valores distinto entre
        si, ni mucho menos combinar erroneamente sujetos.
        """
        # Si ambos diccionarios poseen los mismos sujetos y en el mismo orden, OK!
        if list(dict_coherence_pre.keys()) != list(dict_coherence_post.keys()):
            print("ERROR!")

        # Si en cada sujeto el numero de pruebas entre instantes coincide, OK!
        for i in list(dict_coherence_pre.keys()):
            bOK = len(dict_coherence_pre[i].keys())==len(dict_coherence_post[i].keys())
            if not bOK:
                print("ERROR")
        return

    def plot_diff(matrices, channels, titles = ["Target", "Delta", "CTRL","PLCB","EXP"]):

        # Creamos figuras y ejes
        fig, axes = plt.subplots(1, 4, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

        # Seleccionamos el titulo general de la figura
        plt.suptitle('['+titles[0]+' - '+titles[1]+']'+r' Coherence ($\Delta$ POST-PRE)', fontsize=16)

        # Calculamos valores minimos y maximos para la barra de colores
        vmin = min(matrix.min() for matrix in matrices)  # Encuentra el minimo global
        vmax = max(matrix.max() for matrix in matrices)  # Encuentra el maximo global

        # Dibujamos los heatmaps
        for i, ax in enumerate(axes[:-1]):
            im = sns.heatmap(matrices[i], center=0, annot=False, 
                        xticklabels=channels, yticklabels=channels, cmap="bwr", 
                        vmin=vmin, vmax=vmax,linewidths=0.5, cbar=False, ax=ax)
            
            ax.set_title(titles[i+2], fontsize=14)
            if i == 0:
                ax.set_ylabel("Channels", fontsize=12)
            if i == 1:
                ax.set_xlabel("Channels", fontsize=12)

            ax.tick_params(axis='x', rotation=45, labelsize=9)  # Rotacion y tamaño de ticks en X
            ax.tick_params(axis='y', labelsize=9)  # Tamaño de ticks en Y

        # Ocultamos los ejes de la cuarta figura
        axes[-1].axis('off')

        fig.colorbar(im.collections[0], ax=axes, orientation="vertical", fraction=0.02, pad=0.02)

        # Ajustamos diseño manualmente
        fig.subplots_adjust(left=0.05, right=0.9, wspace=0.15)

        plt.show()

        return