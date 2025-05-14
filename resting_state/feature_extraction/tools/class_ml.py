from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np


class ML:
    def stack_subject_by_channel(list_df, col, f_to_extract, ids = False): 
        
        ids = []
        stacks = []
        for df in list_df:
            ids.append(df['id'].unique())

            stacks.append(np.vstack([df_c[f_to_extract].values for c, df_c in df.groupby([col])]).T)
        if ids:
            return stacks,ids
        else:
            return stacks
    
    def rearrange_features(df, N):

        # Indicamos donde empiezan las columnas con caracteristicas
        features = df.columns[N:]

        # Definimos nombre de las columnas en nuevo DataFrame 
        cols = ['id', 'grupo'] + [f+"_"+ch for f in features for ch in df['ch'].unique()]

        # Build an empty dataframe
        df_out = pd.DataFrame(columns=cols)

        for id, sub_df in df.groupby('id'):
            # Cada ID posee un grupo, pero ese valor esta repetido
            grupo = sub_df['grupo'].values[0]
            
            # Hay un valor por componente y varias caracteristicas, así que
            # seleccionamos el conjuntos de valores por cada caracteristica.
            row = [sub_df[f].values for f in features]

            # Rellenamos el DataFrame
            df_out.loc[len(df_out)] = [id, grupo] + list(np.concatenate(row))

        return df_out
    
    def scale_features(df, features, N):
        
        # Construimos diccionario para almacenar las posiciones de las caracteristicas
        idx = {f: [idx for idx, col in enumerate(df.columns) if f in col] for f in features}

        # Definimos una lista que posee columnas de informacion
        df_out = [df.iloc[:,:N]]


        for f in features:
            # Seleccionamos todas las filas y columnas con caracteristicas
            df_f = df.iloc[:, idx[f]]
            # Calculamos media y std
            mu, sigma = df_f.values.flatten().mean(), df_f.values.flatten().std()
                
            # Normalizamos cada conjunto usando su media y desviacion estandar global
            df_out.append((df_f - mu) / sigma)

        return pd.concat(df_out, ignore_index=False, axis=1)
    
    def scatter_plot_groups(df, f = ['f_0','f_1'], c = ["blue", "gray", "red"], l = ["CTRL", "PLCB", "EXP"], title = "TITLE", subtitle = ["PRE", "POST"]):

        # Filtrar los datos para 'Control' y 'Placebo' en momentos 'PRE' y 'POST'
        pre_ctrl = df[(df['grupo']=='Control')&(df['instante']=="PRE")]
        post_ctrl = df[(df['grupo']=='Control')&(df['instante']=="POST")]

        pre_plcb = df[(df['grupo']=='Placebo')&(df['instante']=="PRE")]
        post_plcb = df[(df['grupo']=='Placebo')&(df['instante']=="POST")]

        pre_exp = df[(df['grupo']=='Exp')&(df['instante']=="PRE")]
        post_exp = df[(df['grupo']=='Exp')&(df['instante']=="POST")]

        # Crear la figura con 1 fila y 2 columnas
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex = False, sharey=False)  # 1 fila, 2 columnas

        fig.suptitle(title, fontsize=16)

        # Gráfico de dispersión para PRE (izquierda)
        axes[0].scatter(pre_ctrl[f[0]], pre_ctrl[f[1]], color=c[0], edgecolors='k', alpha=0.3, label=l[0])
        axes[0].scatter(pre_plcb[f[0]], pre_plcb[f[1]], color=c[1], edgecolors='k', alpha=0.3, label=l[1])
        axes[0].scatter(pre_exp[f[0]], pre_exp[f[1]], color=c[2], edgecolors='k', alpha=0.3, label=l[2])
        axes[0].set_xlabel(f[0])
        axes[0].set_ylabel(f[1])
        axes[0].set_title(subtitle[0])
        #axes[0].set_ylim([-2, 4])
        #axes[0].set_xlim([-0.9, 2])
        axes[0].grid()
        axes[0].legend()

        # Gráfico de dispersión para POST (derecha)
        axes[1].scatter(post_ctrl[f[0]], post_ctrl[f[1]], color=c[0], edgecolors='k', alpha=0.3, label=l[0])
        axes[1].scatter(post_plcb[f[0]], post_plcb[f[1]], color=c[1], edgecolors='k', alpha=0.3, label=l[1])
        axes[1].scatter(post_exp[f[0]], post_exp[f[1]], color=c[2], edgecolors='k', alpha=0.3, label=l[2])
        axes[1].set_xlabel(f[0])
        axes[1].set_ylabel(f[1])
        axes[1].set_title(subtitle[1])
        #axes[1].set_ylim([-2, 4])
        #axes[1].set_xlim([-0.9, 2])
        axes[1].grid()
        axes[1].legend()

        plt.tight_layout()
        plt.show()
    
        return
    
    def scatter_plot_instants(df, f = ['f_0','f_1'], c = ["blue", "gray", "red"], l = ["PRE", "POST"], title = "TITLE", subtitle = ["CTRL", "PLCB", "EXP"]):

        # Filtrar los datos para 'Control' y 'Placebo' en momentos 'PRE' y 'POST'
        pre_ctrl = df[(df['grupo']=='Control')&(df['instante']=="PRE")]
        post_ctrl = df[(df['grupo']=='Control')&(df['instante']=="POST")]

        pre_plcb = df[(df['grupo']=='Placebo')&(df['instante']=="PRE")]
        post_plcb = df[(df['grupo']=='Placebo')&(df['instante']=="POST")]

        pre_exp = df[(df['grupo']=='Exp')&(df['instante']=="PRE")]
        post_exp = df[(df['grupo']=='Exp')&(df['instante']=="POST")]

        # Crear una figura con 2 filas y 1 columna (un gráfico para cada grupo)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey = False)  # 1 fila, 2 columnas

        fig.suptitle(title, fontsize=16)

        # Primer gráfico: Control
        axes[0].scatter(pre_ctrl[f[0]], pre_ctrl[f[1]], color='green', edgecolors='k', alpha=0.3, label=l[0])
        axes[0].scatter(post_ctrl[f[0]], post_ctrl[f[1]], color='brown', edgecolors='k', alpha=0.3, label=l[1])
        axes[0].set_xlabel(f[0])
        axes[0].set_ylabel(f[1])
        axes[0].set_title(subtitle[0])
        axes[0].grid()
        # axes[0].set_ylim([-2, 4])
        # axes[0].set_xlim([-0.5, 1])
        axes[0].legend()

        # Segundo gráfico: Placebo
        axes[1].scatter(pre_plcb[f[0]], pre_plcb[f[1]], color='green', edgecolors='k', alpha=0.3, label=l[0])
        axes[1].scatter(post_plcb[f[0]], post_plcb[f[1]], color='brown', edgecolors='k', alpha=0.3, label=l[1])
        axes[1].set_xlabel(f[0])
        axes[1].set_ylabel(f[1])
        axes[1].set_title(subtitle[1])
        axes[1].grid()
        # axes[1].set_ylim([-2, 4])
        # axes[1].set_xlim([-0.5, 1])
        axes[1].legend()

        axes[2].scatter(pre_exp[f[0]], pre_exp[f[1]], color='green', edgecolors='k', alpha=0.3, label=l[0])
        axes[2].scatter(post_exp[f[0]], post_exp[f[1]], color='brown', edgecolors='k', alpha=0.3, label=l[1])
        axes[2].set_xlabel(f[0])
        axes[2].set_ylabel(f[1])
        axes[2].set_title(subtitle[2])
        axes[2].grid()
        # axes[2].set_ylim([-2, 4])
        # axes[2].set_xlim([-0.5, 1])
        axes[2].legend()

        plt.tight_layout()
        plt.show()

        return
    
    def pca_plot(X, idx = [], c = ["blue","black","red"], l = ["CTRL","PLCB","EXP"], a = [0.4, 0.4, 0.4]):
    
        pca = PCA(n_components=3)
        data_pca = pca.fit_transform(X)

        # Mostrar la varianza explicada por cada componente
        print("Explained Variance for every component:", pca.explained_variance_ratio_)
        print("Total Explained Variance:", sum(pca.explained_variance_ratio_))

        # Definir los índices para cada grupo
        idx_ctrl, idx_plcb, idx_exp = idx[0], idx[1], idx[2]

        # Crear los gráficos en 2D para las relaciones entre las componentes principales
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharex=False, sharey=False)

        # PC1 vs PC2 plot
        ax1.scatter(data_pca[idx_ctrl, 0], data_pca[idx_ctrl, 1], c=c[0], edgecolors='k', alpha=a[0], label = l[0])
        ax1.scatter(data_pca[idx_exp, 0], data_pca[idx_exp, 1], c=c[2], edgecolors='k', alpha=a[2], label = l[2])
        ax1.scatter(data_pca[idx_plcb, 0], data_pca[idx_plcb, 1], c=c[1], edgecolors='k', alpha=a[1], label = l[1])

        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.set_title('PC1 vs PC2')
        ax1.legend()
        ax1.grid()

        # PC1 vs PC3 plot
        ax2.scatter(data_pca[idx_ctrl, 0], data_pca[idx_ctrl, 2], c=c[0], edgecolors='k', alpha=a[0], label = l[0])
        ax2.scatter(data_pca[idx_exp, 0], data_pca[idx_exp, 2], c=c[2], edgecolors='k', alpha=a[2], label = l[2])        
        ax2.scatter(data_pca[idx_plcb, 0], data_pca[idx_plcb, 2], c=c[1], edgecolors='k', alpha=a[1], label = l[1])


        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 3')
        ax2.set_title('PC1 vs PC3')
        ax2.legend()
        ax2.grid()

        # PC2 vs PC3 plot
        ax3.scatter(data_pca[idx_ctrl, 1], data_pca[idx_ctrl, 2], c=c[0], edgecolors='k', alpha=a[0], label = l[0])
        ax3.scatter(data_pca[idx_exp, 1], data_pca[idx_exp, 2], c=c[2], edgecolors='k', alpha=a[2], label = l[2])
        ax3.scatter(data_pca[idx_plcb, 1], data_pca[idx_plcb, 2], c=c[1], edgecolors='k', alpha=a[1], label = l[1])

        ax3.set_xlabel('Principal Component 2')
        ax3.set_ylabel('Principal Component 3')
        ax3.set_title('PC2 vs PC3')
        ax3.legend()
        ax3.grid()

        # Ajustar el layout para evitar superposiciones
        plt.tight_layout()

        # Mostrar los gráficos
        plt.show()
        return
    
    def bar_plot(diff_ctrl, diff_plcb, diff_exp, N, title = "titulo"):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)  # 1 fila, 3 columnas

        # Graficar las medianas de cada columna
        axes[0].bar(np.arange(1, N + 1), np.median(diff_ctrl, axis=0), color='skyblue')
        axes[0].set_title("CTRL")
        # axes[0].set_xlabel("Columnas")
        axes[0].set_ylabel("Median ($\\Delta$ POST-PRE)")
        axes[0].set_xlim([0,N+1])
        axes[0].set_xticks(np.arange(1, N + 1))  # Asegurar que haya un tick por columna
        axes[0].set_xticklabels([f'C{i}' for i in range(1, N + 1)], rotation=45)  # Etiquetas C1, C2, C3, ...
        axes[0].grid(True)

        axes[1].bar(np.arange(1, N + 1), np.median(diff_plcb, axis=0), color='gray')
        axes[1].set_title("PLCB")
        axes[1].set_xlabel("ICA Components")
        # axes[1].set_ylabel("Mediana de los valores")
        axes[1].set_xlim([0,N+1])
        axes[1].set_xticks(np.arange(1, N + 1))  # Asegurar que haya un tick por columna
        axes[1].set_xticklabels([f'C{i}' for i in range(1, N + 1)], rotation=45)  # Etiquetas C1, C2, C3, ...
        axes[1].grid(True)

        axes[2].bar(np.arange(1, N + 1), np.median(diff_exp, axis=0), color='salmon')
        axes[2].set_title("EXP")
        # axes[2].set_xlabel("Columnas")
        # axes[2].set_ylabel("Mediana de los valores")
        axes[2].set_xlim([0,N+1])
        axes[2].set_xticks(np.arange(1, N + 1))  # Asegurar que haya un tick por columna
        axes[2].set_xticklabels([f'C{i}' for i in range(1, N + 1)], rotation=45)  # Etiquetas C1, C2, C3, ...
        axes[2].grid(True)

        plt.suptitle(title, fontsize=16)

        # Ajustar el espacio entre los subgráficos
        plt.tight_layout()

        # Mostrar el gráfico
        plt.show()

        return
    
    def mean_diff_plot(dict_diff, f = ["x","y"], bAverage = True, bID = False, groups = ["CTRL", "PLCB", "EXP"], c = ["b","black","r"], title = "title", text = ["x_label","y_label"], a = [0.3, 0.3, 0.3]):

        # Crear una figura con 1 fila y 2 columnas
        fig, ax = plt.subplots(1, 2, figsize=(16, 5))  # 1 fila, 2 columnas

        # Recorremos los datos
        for k, key in enumerate(dict_diff.keys()):
            df = dict_diff[key]
            for i,g in enumerate(groups):
                
                ids = df[df["grupo"]==g]['id'].values
                x = df[df["grupo"]==g][f[0]].values
                y = df[df["grupo"]==g][f[1]].values

                ax[k].scatter(x, y, color=c[i], marker='o', alpha=a[i], edgecolors='k', label=g)

                # Añadimos los identificadores en cada punto
                for idx, (x0, y0) in enumerate(zip(x, y)):
                    ax[k].text(x0, y0, str(ids[idx]), fontsize=8, alpha=0.7, color=c[i], ha='right', va='bottom')

                # Etiquetas y título
                ax[k].set_xlabel(text[0])
                ax[k].set_ylabel(text[1])
                if key == "oa":
                    ax[k].set_title("Ojos Abiertos")
                else:
                    ax[k].set_title("Ojos Cerrados")

                ax[k].legend(title="Grupos", loc="upper right", fontsize=10)

        fig.suptitle(title, fontsize=16)

        # Mostrar la cuadrícula en ambos subgráficos
        for ax_ in ax:
            ax_.grid(True, linestyle='--', alpha=0.6)

        # Mostrar la figura con ambos subgráficos
        plt.tight_layout()  # Ajusta el layout para que no se superpongan
        plt.show()
        return
    
    def mean_plot(dict_data, f = "feature", bAverage = True, bID = False, groups = ["Control", "Placebo", "Exp"], c = ["b","black","r"], title = "title", text = ["x_label","y_label"], a = [0.3, 0.3, 0.3]):

        # Crear una figura con 1 fila y 2 columnas
        fig, ax = plt.subplots(1, 2, figsize=(16, 5), sharex=True, sharey=True)  # 1 fila, 2 columnas

        # Recorremos los datos
        for k, key in enumerate(dict_data.keys()):
            df_pre, df_post = dict_data[key]

            for i, g in enumerate(groups):
                ids = df_pre[df_pre["grupo"]==g]['id'].values
                x_pre = df_pre[df_pre["grupo"]==g][f].values
                x_post = df_post[df_post["grupo"]==g][f].values

                # Graficamos en el primer subgráfico (columna 0)
                ax[k].scatter(x_pre, x_post, color=c[i], marker='o', alpha=a[i], edgecolors='k', label=g)
                
                # Añadimos los identificadores en cada punto
                for idx, (x, y) in enumerate(zip(x_pre, x_post)):

                    ax[k].text(x, y, str(ids[idx]), fontsize=8, alpha=0.7, color=c[i], ha='right', va='bottom')

            # Etiquetas y título
            ax[k].set_xlabel(text[0])
            ax[k].set_ylabel(text[1])
            if key == "oa":
                ax[k].set_title("Ojos Abiertos")
            else:
                ax[k].set_title("Ojos Cerrados")

            # Dibujar la línea y = x (diagonal sin cambio)
            min_val = min(np.min(x_pre), np.min(x_post))
            max_val = max(np.max(x_pre), np.max(x_post))

            ax[k].plot([min_val-0.25, max_val+0.25], [min_val-0.25, max_val+0.25], color='green', linestyle='--', linewidth=1, label='Sin cambio')

            ax[k].legend(title="Grupos", loc="lower right", fontsize=10)

        fig.suptitle(title, fontsize=16)

        # Mostrar la cuadrícula en ambos subgráficos
        for ax_ in ax:
            ax_.grid(True, linestyle='--', alpha=0.6)

        # Mostrar la figura con ambos subgráficos
        plt.tight_layout()  # Ajusta el layout para que no se superpongan
        plt.show()
        return
    

    def plot_log_reg(dict_datasets, class_names = ['Control', 'Placebo', 'Exp'], title = "title", subtitle = ["t0","t1","t2"]):

        dict_report = {}

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(title, fontsize=16)
        for i, (key, [X,y]) in enumerate(dict_datasets.items()):

            dict_results= ML.perform_log_reg(X, y, class_names)
            cm, dict_report[key] = dict_results["cm"], dict_results["info"]

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[i])
            axes[i].set_xlabel("Predicción")
            axes[i].set_ylabel("Valor Real")
            axes[i].set_title(subtitle[i])

        # Ajustar diseño
        plt.tight_layout()
        plt.show()

        return dict_report
    


    def normalize(df, N):
        df_scaled = df.copy()  # Crear una copia del DataFrame original
        df_scaled[df_scaled.columns[N:]] = StandardScaler().fit_transform(df.iloc[:, N:])# Escalar las columnas
        return df_scaled

    def compute_mean(df, nCol , group_by = ["colA", "colB"]):
    
        # Inicializar una lista para almacenar las filas
        mean_rows = []

        # Iterar sobre el grupo por 'id' y 'grupo'
        for (id, grupo), sub_dfi in df.groupby(group_by):

            # Calcular la media de las columnas
            mean_row = sub_dfi.iloc[:, nCol:].mean().to_frame().T
            
            # Agregar las columnas 'id' y 'grupo'
            mean_row['grupo'] = grupo
            mean_row['id'] = id
            # Añadir la fila al listado
            mean_rows.append(mean_row)
            
        # Concatenar todas las filas en un solo DataFrame
        return pd.concat(mean_rows, ignore_index=True)

    def perform_log_reg(X, y, class_names):

        # Dividimos en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True, stratify=y)

        # Parametros de GridSearchCV()
        param_grid = {
            'C': [0.1, 0.02, 0.04, 0.06, 0.08, 0.1,0.2,0.4,0.6,0.8,1.0],  # Valores para el parametro de regularizacion
            'penalty': ['l2'],  # Tipo de penalizacion (Ridge)
            'solver': ['lbfgs','sag'] # Sovlers compatibles con problema mutliclase
        }

        # Definimos modelo de Regresion Logistica
        log_reg = LogisticRegression(max_iter=500)

        # Cargamos parametros del modelo en GridSearchCV
        grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid, cv=10, scoring="accuracy")

        # Ajustamos el modelo a los datos de entrenamiento
        grid_search.fit(X_train, y_train)

        # Seleccionamos el mejor modelo
        best_model = grid_search.best_estimator_

        # Realizamos prediccion
        y_pred = best_model.predict(X_test)

        # Obtenemos matriz de confusion y declaramos 'labels'
        cm = confusion_matrix(y_test, y_pred, labels=class_names)

        return {"cm":cm, "info":{"best_params": grid_search.best_params_,"report": classification_report(y_test, y_pred, zero_division=0),"accuracy": accuracy_score(y_test, y_pred)}}


