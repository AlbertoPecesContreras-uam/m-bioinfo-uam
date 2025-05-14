from statsmodels.stats.multitest import multipletests
from tools.class_utils import Utils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats




class Entropy:
    def compute_paired_test(pre, post, channels, test = "wilcoxon"):
        # DataFrame con informacion de los test realizados
        df = pd.DataFrame([],columns=["grupo","region","contrast","p_value"])

        combined_dict = {key: [pre[key], post[key]] for key in pre}

        # Iteramos por los datos de cada grupo e instante
        for grupo, (pre,post) in combined_dict.items():
            
            for ch in channels:
            #for region, ch_group in eeg_regions.items():

                # Seleccionamos los electrodos que conforman cada region cerebral
                post_data = post[post['ch'].isin([ch])]
                pre_data = pre[pre['ch'].isin([ch])]

                # Promediamos los canales por caracteristicas
                #avg_pre, avg_post = Utils.average_by_feature([pre_data, post_data], ["H"])

                # Perform Wilcoxon Rank Test
                if test == "wilcoxon":
                    stat,p_value = stats.wilcoxon(pre_data["H"], post_data["H"])
                else:
                    stat,p_value = stats.ttest_rel(pre_data["H"], post_data["H"])

                # Compute mean values
                mu_pre = np.mean(pre_data["H"])
                mu_post = np.mean(post_data["H"])

                # Show contrast info and cohen's d value
                contrast = "PRE < POST" if mu_pre < mu_post else "PRE > POST"

                # Guardamos fila en DataFrame
                df.loc[len(df)] = [grupo, ch, contrast, p_value]

        return df

    def compute_shEn(s):

        # Dividimos la señal en bins, es decir, en un numero de ventanas determinado.
        # En nuestro caso, dividimos la señal en 50 ventanas. Puede darse el caso de 
        # que len(signal)/bins sea impar. En ese caso, se ajusta automaticamente.
        # len(hist) = len(bins), cada valor en hist posee el conteo de valores de la
        # señal que caen en ese bin. 
        hist, bin_edges = np.histogram(s, bins=50)

        # Convertimos los conteos o frecuencias en probabilidades (todo suma 1)
        hist = hist/hist.sum()

        # Ignoramos los valores que sean 0 porque dan problemas en la formula de Shannon
        hist = hist[hist>0]

        # Calculate Shannon entropy
        H_shannon = -np.sum(hist * np.log2(hist))

        return H_shannon

    def get_shEn(df, channels):

        # Lista para guardar entropias
        entropy = []
        # Para cada sujeto, then:
        for n in [1,2]:
            df_n = df[df['n_test']==n]
            # Seleccionamos los IDs unicos del DataFrame
            IDs = df_n["id"].unique()
            for id in IDs:
                # Para cada canal, then:
                for ch in channels:
                    s = df_n[df_n['id']==id][ch].to_numpy(float)
                    entropy.append(Entropy.compute_shEn(s))
        return entropy

    # Calculamos r para cada canal y componente de forma global para todos los grupos
    def compute_r(list_dfs, channels, components, params = {}):

        dict_r = {comp: {ch: [] for ch in channels} for comp in components}

        # Por cada DataFrame, then:
        for df in list_dfs:
            # Por cada componente, then:
            for comp in components:
                # Por cada canal, then:
                for ch in channels:
                    for id in df['id'].unique():
                        for n in [1, 2]:
                            # Seleccionamos la ventana
                            s_ch = df[(df['n_test']==n)&(df['id']==id)][ch].to_numpy(float)
                            # Si la ventana no esta vacia, then:
                            if len(s_ch) != 0:
                                # Si hay parametros para filtro, then:
                                if params:
                                    s_ch = Utils.bandpass_filter(s_ch, params, order = 4)

                                # Guardamos la desviacion estandar de cada ventana
                                dict_r[comp][ch].append(np.std(s_ch))

        # Calculamos la desviacion estandar promedio de todas las ventanas
        return {comp: {ch: np.mean(dict_r[comp][ch]) for ch in channels} for comp in components}

    # Get DataFrame with Approximation Entropy 
    def get_apEn_dataframe(list_dfs, channels, components, params = {}):

        # Calculamos r para cada canal en el DataFrame Global --> dict_r = {'comp':{'ch':apEn,...,},...}
        r = Entropy.compute_r(list_dfs, channels, components)

        # Lista para almacenar los resultados de todos los DataFrames
        all_results = []

        # Iteramos sobre los DataFrames
        for i,df in enumerate(list_dfs):
            instante = df['instante'].unique()[0]# Seleccionamos el instante del DataFrame introducido
            grupo = df['grupo'].unique()[0]# Seleccionamos el grupo del DataFrame introducido

            # Por cada id, then:
            for id in df['id'].unique():

                # Iteramos por cada prueba, ya que tenemos 2
                for n in [1, 2]:
                    # Seleccionamos las ventanas de cada sujeto y prueba especifica
                    df_id = df[(df['n_test']==n)&(df['id']==id)]

                    # Si el DataFrame no esta vacio, then:
                    if not df_id.empty:
                    
                        # Si hay parametros, aplicamos PassBand Filter
                        if params:
                            apEn = [Entropy.app_entropy(Utils.bandpass_filter(df_id[df_id['comp']==comp][ch], params), r[comp][ch])
                                    for comp in components for ch in channels]
                        else:
                            # 9x10 values of Approximation Entropy --> 9 components X 10 channels 
                            apEn = [Entropy.app_entropy(df_id[df_id['comp']==comp][ch], r[comp][ch]) for comp in components for ch in channels]

                        # Crear un DataFrame con los resultados de este componente
                        component_results = pd.DataFrame({
                            'n_test': [n] * len(apEn),
                            'instante': [instante] * len(apEn),
                            'grupo': [grupo] * len(apEn),
                            'id': [id] * len(apEn),
                            'comp': [comp for comp in components for _ in channels],
                            'ch': [ch for _ in components for ch in channels],
                            'H': apEn
                        })
                                
                        # Guardamos los resultados a la lista principal
                        all_results.append(component_results)

            p = (i+1)/len(list_dfs)

            print("-) Progress:",round(p*100,2),"%")

        # Combinar todos los resultados en un solo DataFrame
        return pd.concat(all_results, ignore_index=True)

    def apEn_phi(s,m,r):
        # Dividimos la señal en todos los bins de tamaño m posibles
        bins = np.array([s[i:i+m] for i in range(len(s)-m+1)])

        count = np.zeros(len(s)-m+1, dtype=float)

        for i in range(len(s)-m+1):
            # Calculamos todas las diferencias absolutas y seleccionamos la maxima
            dist = np.max(np.abs(bins-bins[i]),axis=1)

            # Calculamos la proporcion de patrones similares
            count[i] = np.sum(dist <= r)/(len(s)-m+1)

        return np.sum(np.log(count))/(len(s)-m+1)

    # m = 2 for approximation entropy
    def app_entropy(df, r): return max(Entropy.apEn_phi(df,2, 0.1*r)-Entropy.apEn_phi(df,2+1, 0.1*r), 0)

    # Sample Entropy
    def sampEn_phi(s, m, r):
        """
        Función auxiliar para calcular componentes de Sample Entropy (SampEn).
        No incluye self-matches y calcula A y B por separado.
        
        Args:
            s (array): Señal de entrada.
            m (int): Longitud del patrón (típicamente 2).
            r (float): Tolerancia (usual: 0.2*std(s)).
        
        Returns:
            tuple: (A, B) donde:
                - B = Número de patrones similares de longitud m (excluyendo self-matches).
                - A = Número de patrones similares de longitud m+1 (excluyendo self-matches).
        """
        n = len(s)
        
        # Generar patrones de longitud m y m+1
        patterns_m = np.array([s[i:i+m] for i in range(n - m + 1)])
        patterns_m1 = np.array([s[i:i+m+1] for i in range(n - m)])
        
        B = 0  # Contador para patrones de longitud m
        A = 0  # Contador para patrones de longitud m+1
        
        for i in range(n - m + 1):
            # Distancias para patrones de longitud m (excluyendo i=j)
            if i < n - m:
                dist_m = np.max(np.abs(patterns_m - patterns_m[i]), axis=1)
                similar_m = np.sum(dist_m <= r) - 1  # Restamos 1 para excluir self-match
                B += similar_m
            
            # Distancias para patrones de longitud m+1 (solo si i no es el último)
            if i < n - m:
                dist_m1 = np.max(np.abs(patterns_m1 - patterns_m1[i]), axis=1)
                similar_m1 = np.sum(dist_m1 <= r) - 1  # Excluir self-match
                A += similar_m1
        
        # Normalizamos por el número total de comparaciones posibles
        B /= (n - m) * (n - m - 1)
        A /= (n - m) * (n - m - 1)
        
        return A, B

    def sampEn_entropy(s, r):
        """
        Calcula Sample Entropy (SampEn) de una señal.
        
        Args:
            s (array): Señal de entrada.
            m (int): Longitud del patrón (default: 2).
            r (float): Tolerancia como fracción de la std (default: 0.2).
                    Se calcula como r * np.std(s).
        
        Returns:
            float: Valor de SampEn. Retorna 0 si no hay patrones similares.
        """

        A, B = Entropy.sampEn_phi(s, 2, 0.1*r)
        
        if A == 0 or B == 0:
            return 0  # Evitar división por cero o log(0)
        
        return -np.log(A / B)

    def compute_diff(df1, df0):
        # Definimos DataFrame para las diferencias 
        # de entropia (POST-PRE) de cada sujeto y canal
        df_out = pd.DataFrame([], columns=df0['ch'].unique())
        for (n,id),df0_id in df0.groupby(['n_test','id']):

            # Seleccionamos las mismas filas en ambos DataFrames
            df1_id = df1[(df1['n_test']==n)&(df1['id']==id)]

            # Calculamos la diferencia en porcentaje
            diff = (df1_id['H'].values - df0_id['H'].values)/df0_id['H'].values

            # Guardamos una lista de diferencias por 
            # sujeto, donde cada valor es un canal
            df_out.loc[len(df_out)] = diff

        return df_out
    
    def plot_by_instants(apEn_pre, apEn_post, comp, title = "Approximation Entropy"):

        # Crear una cuadrícula de subgráficos de 3 filas y 1 columna, compartiendo el eje X
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True, sharey=True)  # sharex=True hace que compartan el eje X

        # Primer gráfico (CTRL)
        axs[0].hist(apEn_pre[0], bins=30, color='green', edgecolor='black', alpha=0.4, label='PRE', density=True)
        axs[0].hist(apEn_post[0], bins=30, color='brown', edgecolor='black', alpha=0.2, label='POST', density=True)

        # Añadir la curva de densidad con Seaborn
        sns.kdeplot(apEn_pre[0], color='green', lw=2, alpha=1.0, ax=axs[0])
        sns.kdeplot(apEn_post[0], color='brown', lw=2, alpha=0.7, ax=axs[0])
        
        # get_kde_max_value(apEn_pre[0])
        # get_kde_max_value(apEn_post[0])

        # Obtener el valor máximo de la curva KDE y agregar una línea vertical
        axs[0].axvline(np.median(apEn_pre[0]), color='green', linestyle='--', lw=1, label=f'PRE - Median')
        axs[0].axvline(np.median(apEn_post[0]), color='brown', linestyle='--', lw=1, label=f'POST - Median')

        # Títulos y etiquetas para el primer gráfico
        axs[0].set_title('[CTRL] - '+title)
        axs[0].set_ylabel('Density')
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.7)

        # Segundo gráfico (PLCB)
        axs[1].hist(apEn_pre[1], bins=30, color='green', edgecolor='black', alpha=0.4, label='PRE', density=True)
        axs[1].hist(apEn_post[1], bins=30, color='brown', edgecolor='black', alpha=0.2, label='POST', density=True)

        # Añadir la curva de densidad con Seaborn
        sns.kdeplot(apEn_pre[1], color='green', lw=2, alpha=1.0, ax=axs[1])
        sns.kdeplot(apEn_post[1], color='brown', lw=2, alpha=0.7, ax=axs[1])

        # get_kde_max_value(apEn_pre[1])
        # get_kde_max_value(apEn_post[1])

        # Obtener el valor máximo de la curva KDE y agregar una línea vertical
        axs[1].axvline(np.median(apEn_pre[1]), color='green', linestyle='--', lw=1, label=f'PRE - Median')
        axs[1].axvline(np.median(apEn_post[1]), color='brown', linestyle='--', lw=1, label=f'POST - Median')

        # Títulos y etiquetas para el segundo gráfico
        axs[1].set_title('[PLCB] - '+title)
        axs[1].set_ylabel('Density')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.7)

        # Tercer gráfico (EXP)
        axs[2].hist(apEn_pre[2], bins=30, color='green', edgecolor='black', alpha=0.4, label='PRE', density=True)
        axs[2].hist(apEn_post[2], bins=30, color='brown', edgecolor='black', alpha=0.2, label='POST', density=True)

        # Añadir la curva de densidad con Seaborn
        sns.kdeplot(apEn_pre[2], color='green', lw=2, alpha=1.0, ax=axs[2])
        sns.kdeplot(apEn_post[2], color='brown', lw=2,  alpha=0.7, ax=axs[2])

        # Obtener el valor máximo de la curva KDE y agregar una línea vertical
        axs[2].axvline(np.median(apEn_pre[2]), color='green', linestyle='--', lw=1, label=f'PRE - Median')
        axs[2].axvline(np.median(apEn_post[2]), color='brown', linestyle='--', lw=1, label=f'POST - Median')

        # Títulos y etiquetas para el tercer gráfico
        axs[2].set_title('[EXP] - '+title)
        axs[2].set_xlabel('Values')  # Solo el tercer gráfico tiene la etiqueta del eje X
        axs[2].set_ylabel('Density')
        axs[2].legend()
        axs[2].grid(True, linestyle='--', alpha=0.7)

        fig.suptitle('Component: {}'.format(comp), fontsize=16, fontweight='bold')

        # Ajustar la distribución de los subgráficos para que no se sobrepongan
        plt.tight_layout()

        # Mostrar la figura
        plt.show()
        return

    def plot_by_groups(apEn_pre, apEn_post, comp, title='Approximation Entropy'):
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex = True, sharey = True)  # Ajusta el tamaño de la figura

        # First plot (PRE)
        axs[0].hist(apEn_pre[0], bins=40, color='blue', edgecolor='black', alpha=0.4, label='CTRL', density=True)
        axs[0].hist(apEn_pre[1], bins=40, color='black', edgecolor='black', alpha=0.2, label='PLCB', density=True)
        axs[0].hist(apEn_pre[2], bins=40, color='lightcoral', edgecolor='black', alpha=0.4, label='EXP', density=True)

        # Añadir la curva de densidad con Seaborn
        sns.kdeplot(apEn_pre[0], color='blue', lw=2, alpha=1.0, ax=axs[0])
        sns.kdeplot(apEn_pre[1], color='black', lw=2, alpha=0.7, ax=axs[0])
        sns.kdeplot(apEn_pre[2], color='red', lw=2, alpha=0.7, ax=axs[0])

        # get_kde_max_value(apEn_pre[0])
        # get_kde_max_value(apEn_pre[1])
        # get_kde_max_value(apEn_pre[2])
        axs[0].axvline(np.median(apEn_pre[0]), color='blue', linestyle='--', lw=1, label=f'CTRL - Median')
        axs[0].axvline(np.median(apEn_pre[1]), color='black', linestyle='--', lw=1, label=f'PLCB - Median')
        axs[0].axvline(np.median(apEn_pre[2]), color='red', linestyle='--', lw=1, label=f'EXP - Median')

        # Títulos y etiquetas para el primer gráfico
        axs[0].set_title('[PRE] - '+title)
        axs[0].set_xlabel('Values')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.7)

        # Segundo gráfico (POST)
        axs[1].hist(apEn_post[0], bins=40, color='blue', edgecolor='black', alpha=0.4, label='CTRL', density=True)
        axs[1].hist(apEn_post[1], bins=40, color='black', edgecolor='black', alpha=0.2, label='PLCB', density=True)
        axs[1].hist(apEn_post[2], bins=40, color='lightcoral', edgecolor='black', alpha=0.4, label='EXP', density=True)

        # Añadir la curva de densidad con Seaborn
        sns.kdeplot(apEn_post[0], color='blue', lw=2, alpha=1.0, ax=axs[1])
        sns.kdeplot(apEn_post[1], color='black', lw=2, alpha=0.7, ax=axs[1])
        sns.kdeplot(apEn_post[2], color='red', lw=2, alpha=0.7, ax=axs[1])

        # get_kde_max_value(apEn_post[0])
        # get_kde_max_value(apEn_post[1])
        # get_kde_max_value(apEn_post[2])
        axs[1].axvline(np.median(apEn_post[0]), color='blue', linestyle='--', lw=1, label=f'CTRL - Median')
        axs[1].axvline(np.median(apEn_post[1]), color='black', linestyle='--', lw=1, label=f'PLCB - Median')
        axs[1].axvline(np.median(apEn_post[2]), color='red', linestyle='--', lw=1, label=f'EXP - Median')

        # Títulos y etiquetas para el segundo gráfico
        axs[1].set_title('[POST] - '+title)
        axs[1].set_xlabel('Values')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.7)

        fig.suptitle('Component: {}'.format(comp), fontsize=16, fontweight='bold')

        # Ajustar la distribución de los subgráficos para que no se sobrepongan
        plt.tight_layout()

        # Mostrar la figura
        plt.show()
        return
    
    def combine_channels(df, eeg_groups):
        dict_H = {}
        for region, channels in eeg_groups.items():
            
            distribution = np.concatenate([df[df['ch'] == ch]['H'].to_numpy(float) for ch in channels])

            dict_H[region] = np.median(distribution)

        return dict_H

    def plot_by_regions(CTRL, PLCB, EXP, keys):

        bar_width = 0.4  # Ancho de las barras
        x_pos = np.arange(len(keys))  # Posiciones para el eje X

        # Lista de datasets y configuraciones
        datasets = [
            (CTRL['PRE'], CTRL['POST'], "green", "brown", "PRE", "POST", "CTRL"),
            (PLCB['PRE'], PLCB['POST'], "green", "brown", "PRE", "POST", "PLCB"),
            (EXP['PRE'], EXP['POST'], "green", "brown", "PRE", "POST", "EXP"),
        ]

        # Crear 3 gráficos en una fila
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

        # Bucle para generar cada gráfico
        for i, (vals1, vals2, color1, color2, label1, label2, title) in enumerate(datasets):
            axes[i].bar(x_pos, vals1, width=bar_width, color=color1, label=label1, alpha=0.6)
            axes[i].bar(x_pos + bar_width, vals2, width=bar_width, color=color2, label=label2, alpha=0.6)
            
            # Configuración del gráfico
            axes[i].set_xlabel("Regions", fontsize=12)
            axes[i].set_title(title, fontsize=14)
            axes[i].set_xticks(x_pos + bar_width / 2)
            axes[i].set_xticklabels(keys, rotation=45, fontsize=10)
            axes[i].legend()
            axes[i].grid()

        # Ajustar espacio entre gráficos
        plt.tight_layout()
        plt.show()
        return 
    
    def plot_diff(apEn_arrays, electrodes, title = "Title", subtitles = ["CTRL","PLCB","EXP"]):

        df0_col, df1_col, df2_col = apEn_arrays

        # Ordenamos los valores de entropia para que se correspondan con los electrodos
        ctrl = [df0_col.loc[key] for key in electrodes.keys()]
        plcb = [df1_col.loc[key] for key in electrodes.keys()]
        exp = [df2_col.loc[key] for key in electrodes.keys()]

        data = [ctrl, plcb, exp]

        # Calculamos el valor minimo y maximo para normalizar los colores
        min_value = min(df0_col.min(), df1_col.min(), df2_col.min())
        max_value = max(df0_col.max(), df1_col.max(), df2_col.max())

        lim = max(abs(min_value), abs(max_value))

        # Circunferencias concentricas 
        circunferencias = np.arange(0, 1.5, 0.2)

        # Extraemos las coordenadas
        x = [coord[0] for coord in electrodes.values()]
        y = [coord[1] for coord in electrodes.values()]

        # Crear una figura con 1 fila y 3 columnas
        fig, axes = plt.subplots(1, 4, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})
        plt.suptitle('['+title+']'+r' Approximation Entropy ($\Delta$ POST-PRE)', fontsize=16)

        for i, ax in enumerate(axes[:-1]):

            sc = ax.scatter(x, y, c=data[i], cmap="coolwarm", s=200, vmin=-lim, vmax=+lim)
            ax.set_facecolor('lightgray')
            ax.set_title(subtitles[i])
            ax.set_xticks([])  # Eliminar las marcas en el eje X
            ax.set_yticks([])  # Eliminar las marcas en el eje Y
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.0, 1.0])
            ax.grid(True)

            # Etiquetas de los electrodos en cada grafico
            for label, (i, j) in electrodes.items():
                ha = "left" if i > 0 else "right" if i < 0 else "center"
                ax.text(i, j, label, fontsize=14, ha=ha)

            # Dibujar las circunferencias en el gráfico
            for r in circunferencias:
                circle = plt.Circle((0,0), r, color='gray', fill=False, linestyle='--', linewidth=0.5)
                ax.add_artist(circle)

        # Ocultamos los ejes de la cuarta figura
        axes[-1].axis('off')

        # Crear la barra de colores en la cuarta columna
        fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)

        # Ajustamos diseño manualmente
        fig.subplots_adjust(left=0.05, right=0.9, wspace=0.15)

        # Mostrar el grafico
        plt.show()
        return