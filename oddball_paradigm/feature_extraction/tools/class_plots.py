import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from tools.class_utils import Utils
class Plots:

    def scree_plot(explained_variance, elbow_point):
        # Crear el gráfico
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', label='Explained Variance')
        plt.axvline(x=elbow_point, color='r', linestyle='--', label=f'Elbow Point: {elbow_point}', alpha = 0.5)
        plt.title('Scree Plot')
        plt.xlabel('Componente Principal')
        plt.ylabel('Varianza Explicada')
        plt.xticks(range(1, len(explained_variance) + 1))
        plt.legend()
        plt.grid(True)
        plt.show()
        return  
    
    def plot_groups_by_features(df, channels, comp = "all", group = [], params = [], title = "CTRL, PLCB and EXP", l = [], c = [], size = (18, 17), b_id = False):
        # Determinar el número de filas y columnas para la cuadrícula de subplots
        nrows,ncols = params
        # Crear la figura y la cuadrícula de subplots
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=size, sharex=False, sharey=False)

        # Asegurar que ax sea un array bidimensional
        ax = np.array(ax).reshape(nrows, ncols)

        # Iterar sobre cada canal y graficarlo en su subplot correspondiente
        for idx, ch in enumerate(channels):
            i, j = divmod(idx, ncols)  # Convertir índice lineal a coordenadas de la cuadrícula
            
            if len(group) == 2:
                # Get feature values for group_0
                mu_0 = df[(df["grupo"]==group[0])&(df["comp"]==comp)&(df["ch"]==ch)]["mean"]
                var_0 = df[(df["grupo"]==group[0])&(df["comp"]==comp)&(df["ch"]==ch)]["std"]
                # Get feature values for group_1
                mu_1 = df[(df["grupo"]==group[1])&(df["comp"]==comp)&(df["ch"]==ch)]["mean"]
                var_1 = df[(df["grupo"]==group[1])&(df["comp"]==comp)&(df["ch"]==ch)]["std"]

                # Graficar puntos group_0
                ax[i, j].scatter(mu_0, var_0, marker='o', color=c[0], label=l[0], alpha=0.5)
                # Graficar puntos group_1
                ax[i, j].scatter(mu_1, var_1, marker='o', color=c[1], label=l[1], alpha=0.5)

                # Get IDs for subjects in each group
                id_ctrl = df[(df["grupo"]==group[0])&(df["comp"]==comp)&(df["ch"]==ch)]["id"]
                id_plcb = df[(df["grupo"]==group[1])&(df["comp"]==comp)&(df["ch"]==ch)]["id"]

                if b_id: 
                    # Añadir etiquetas al lado de cada punto
                    for k, txt in enumerate(id_ctrl):
                        ax[i, j].text(mu_0.iloc[k], var_0.iloc[k], str(txt), fontsize=9, verticalalignment='bottom', horizontalalignment='right')
                    for k, txt in enumerate(id_plcb):
                        ax[i, j].text(mu_1.iloc[k], var_1.iloc[k], str(txt), fontsize=9, verticalalignment='bottom', horizontalalignment='right')

            if len(group) == 3:
                # Get feature values for CTRL
                mu_0 = df[(df["grupo"]==group[0])&(df["comp"]==comp)&(df["ch"]==ch)]["mean"]
                var_0 = df[(df["grupo"]==group[0])&(df["comp"]==comp)&(df["ch"]==ch)]["std"]
                # Get feature values for PLCB
                mu_1 = df[(df["grupo"]==group[1])&(df["comp"]==comp)&(df["ch"]==ch)]["mean"]
                var_1 = df[(df["grupo"]==group[1])&(df["comp"]==comp)&(df["ch"]==ch)]["std"]
                # Get feature values for EXP
                mu_2 = df[(df["grupo"]==group[2])&(df["comp"]==comp)&(df["ch"]==ch)]["mean"]
                var_2 = df[(df["grupo"]==group[2])&(df["comp"]==comp)&(df["ch"]==ch)]["std"]
                
                # Graficar puntos CTRL
                ax[i, j].scatter(mu_0, var_0, marker='o', color=c[0], label=l[0], alpha=0.5)
                # Graficar puntos PLCB
                ax[i, j].scatter(mu_1, var_1, marker='o', color=c[1], label=l[1], alpha=0.5)
                # Graficar puntos EXP
                ax[i, j].scatter(mu_2, var_2, marker='o', color=c[2], label=l[2], alpha=0.5)
                
                # Get IDs for subjects in each group
                id_ctrl = df[(df["grupo"]==group[0])&(df["comp"]==comp)&(df["ch"]==ch)]["id"]
                id_plcb = df[(df["grupo"]==group[1])&(df["comp"]==comp)&(df["ch"]==ch)]["id"]
                id_exp = df[(df["grupo"]==group[2])&(df["comp"]==comp)&(df["ch"]==ch)]["id"]

                if b_id: 
                    # Añadir etiquetas al lado de cada punto
                    for k, txt in enumerate(id_ctrl):
                        ax[i, j].text(mu_0.iloc[k], var_0.iloc[k], str(txt), fontsize=9, verticalalignment='bottom', horizontalalignment='right')
                    for k, txt in enumerate(id_plcb):
                        ax[i, j].text(mu_1.iloc[k], var_1.iloc[k], str(txt), fontsize=9, verticalalignment='bottom', horizontalalignment='right')
                    for k, txt in enumerate(id_exp):
                        ax[i, j].text(mu_2.iloc[k], var_2.iloc[k], str(txt), fontsize=9, verticalalignment='bottom', horizontalalignment='right')

            # Etiquetas y título del subplot
            ax[i, j].set_xlabel('mean')
            ax[i, j].set_ylabel('std')
            ax[i, j].set_title(f'Canal {ch}')
            ax[i, j].legend()
            ax[i, j].grid(True)

        # Ocultar ejes vacíos si hay más subplots de los necesarios
        for idx in range(len(channels), nrows * ncols):
            if idx >= len(channels):
                i, j = divmod(idx, ncols)
                fig.delaxes(ax[i, j])  # Eliminar subplots vacíos

        # Ajustar diseño
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.99])  # Deja espacio para el título arriba
        plt.show()
        return

    def plot_instants_by_features(df, channels, comp = "all", instant = ["PRE", "POST"], params = [], title = "PRE vs POST", l = ["PRE", "POST"], c = ["green", "brown"], size = (18, 7)):
        # Determinar el número de filas y columnas para la cuadrícula de subplots
        nrows,ncols = params
        # Crear la figura y la cuadrícula de subplots
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=size, sharex=False, sharey=False)

        # Asegurar que ax sea un array bidimensional
        ax = np.array(ax).reshape(nrows, ncols)

        # Iterar sobre cada canal y graficarlo en su subplot correspondiente
        for idx, ch in enumerate(channels):
            i, j = divmod(idx, ncols)  # Convertir índice lineal a coordenadas de la cuadrícula

            # Get feature values for group_0
            mu_0 = df[(df["instante"]==instant[0])&(df["comp"]==comp)&(df["ch"]==ch)]["mean"]
            var_0 = df[(df["instante"]==instant[0])&(df["comp"]==comp)&(df["ch"]==ch)]["std"]
            # Get feature values for group_1
            mu_1 = df[(df["instante"]==instant[1])&(df["comp"]==comp)&(df["ch"]==ch)]["mean"]
            var_1 = df[(df["instante"]==instant[1])&(df["comp"]==comp)&(df["ch"]==ch)]["std"]

            # Graficar puntos CTRL
            ax[i, j].scatter(mu_0, var_0, marker='o', color=c[0], label=l[0], alpha=0.5)

            # Graficar puntos EXP
            ax[i, j].scatter(mu_1, var_1, marker='o', color=c[1], label=l[1], alpha=0.5)

            # Etiquetas y título del subplot
            ax[i, j].set_xlabel('mean')
            ax[i, j].set_ylabel('std')
            ax[i, j].set_title(f'Canal {ch}')
            ax[i, j].legend()
            ax[i, j].grid(True)

        # Ocultar ejes vacíos si hay más subplots de los necesarios
        for idx in range(len(channels), nrows * ncols):
            if idx >= len(channels):
                i, j = divmod(idx, ncols)
                fig.delaxes(ax[i, j])  # Eliminar subplots vacíos

        # Ajustar diseño
        fig.suptitle(title, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.99])  # Deja espacio para el título arriba
        plt.show()
        return
    

    def plot_three_groups(df_0, df_1, df_2, channels, params = [], title = "[TARGET] PRE: CTRL-PLCB-EXP", l = ["CTRL", "PLCB", "EXP"], c = ["b","black","r"], line = ["-","-","-"], size = (18, 7)):# CHECK --> OK

        rows,cols = params

        # Crear una figura con subplots de 'rows' filas y 'cols' columnas
        fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True, sharey=True)

        # Recorre cada canal y grafica en su respectivo subplot
        for idx, canal in enumerate(channels):

            i, j = divmod(idx, cols)# Convertir indice lineal a coordenadas de matriz (fila, columna)

            # Get average window from all subjects in df_0 with component = "all" and channel = canal
            s_0 = Utils.avg_window(df_0, "all", [canal])
            s_1 = Utils.avg_window(df_1, "all", [canal])
            s_2 = Utils.avg_window(df_2, "all", [canal])

            t = np.linspace(-200, 700, 450)  # Mantiene 450 valores con pasos de 2 ms

            # Plot average windows
            axes[i, j].plot(t, s_0, label=l[0], color = c[0], linestyle = line[0], alpha = 0.5)
            axes[i, j].plot(t, s_1, label=l[1], color = c[1], linestyle = line[1], alpha = 0.5)
            axes[i, j].plot(t, s_2, label=l[2], color = c[2], linestyle = line[2], alpha = 0.5)

            # Set legends
            axes[i, j].legend()

            # Set X and Y limits
            axes[i,j].set_ylim([-4, 6])
            axes[i,j].set_xlim([-200,700])

            # Set titles and grid
            axes[i, j].set_title(canal)
            axes[i, j].grid()

            # Set labels fow x and y axes
            if j == 0:
                axes[i, j].set_ylabel('[$\\mu V$]')
            if i == rows-1:
                axes[i, j].set_xlabel('[$ms$]')
                
        # Hide empty subplots (if necessary)
        for idx in range(len(channels), rows * cols):
            i, j = divmod(idx, cols)
            fig.delaxes(axes[i, j])

        # Set main title
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

        return
    
    def plot_two_groups(df_0, df_1, channels, params = [], title = "[TARGET] PRE: CTRL vs EXP", l = ["CTRL", "EXP"], c = ["b","r"], line = ["-","-"], size = (18, 7)):# CHECK --> OK

        rows,cols = params

        # Crear una figura con subplots de 'rows' filas y 'cols' columnas
        fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True, sharey=True)

        # Recorre cada canal y grafica en su respectivo subplot
        for idx, canal in enumerate(channels):

            i, j = divmod(idx, cols)# Convertir indice lineal a coordenadas de matriz (fila, columna)

            # Get average window from all subjects in df_0 with component = "all" and channel = canal
            s_0 = Utils.avg_window(df_0, "all", [canal])
            s_1 = Utils.avg_window(df_1, "all", [canal])

            t = np.linspace(-200, 700, 450)  # Mantiene 450 valores con pasos de 2 ms

            # Plot average windows
            axes[i, j].plot(t, s_0, label=l[0], color = c[0], linestyle = line[0], alpha = 0.5)
            axes[i, j].plot(t, s_1, label=l[1], color = c[1], linestyle = line[1], alpha = 0.5)

            # Set legends
            axes[i, j].legend()

            # Set X and Y limits
            axes[i,j].set_ylim([-4, 6])
            axes[i,j].set_xlim([-200,700])

            # Set titles and grid
            axes[i, j].set_title(canal)
            axes[i, j].grid()

            # Set labels fow x and y axes
            if j == 0:
                axes[i, j].set_ylabel('[$\\mu V$]')
            if i == rows-1:
                axes[i, j].set_xlabel('[$ms$]')
                
        # Hide empty subplots (if necessary)
        for idx in range(len(channels), rows * cols):
            i, j = divmod(idx, cols)
            fig.delaxes(axes[i, j])

        # Set main title
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

        return
    
    def plot_one_group(df_0, channels, flag = "mean", params = [], title = "[TARGET] CTRL", l = ["CTRL"], c = ["b"], size = (18, 7)):# CHECK --> OK

        rows,cols = params

        # Crear una figura con subgráficos de 2 filas y 5 columnas
        fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True, sharey=True)

        # Recorre cada canal y grafica en su respectivo subplot
        for idx, canal in enumerate(channels):

            i, j = divmod(idx, cols)# Convertir indice lineal a coordenadas de matriz (fila, columna)

            # Get average window from all subjects in df_0 with component = "all" and channel = canal
            if flag == "mean":
                s_0 = Utils.avg_window(df_0, "all", canal)
            # Get median window from all subjects in df_0 with component = "all" and channel = canal
            else:
                s_0 = Utils.median_window(df_0, "all", canal)

            t = np.linspace(-200, 700, 450)  # Mantiene 450 valores con pasos de 2 ms

            # Plot window
            axes[i, j].plot(t, s_0, label=l[0], color = c[0], alpha = 0.5)

            # Set legends
            axes[i, j].legend()

            # Set X and Y limits
            axes[i,j].set_ylim([-4, 6])
            axes[i,j].set_xlim([-200,700])

            # Set titles and grid
            axes[i, j].set_title(canal)
            axes[i, j].grid()

            # Set labels fow x and y axes
            if j == 0:
                axes[i, j].set_ylabel('[$\\mu V$]')
            if i == rows-1:
                axes[i, j].set_xlabel('[$ms$]')
                
        # Hide empty subplots (if necessary)
        for idx in range(len(channels), rows * cols):
            i, j = divmod(idx, cols)
            fig.delaxes(axes[i, j])

        # Set main title
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

        return
    
    def plot_components(df_0, df_1, df_2, channel, title = 'title', l = ["CTRL","PLCB","EXP"], size = (10,5)):# CHECK --> OK

        fig, axes = plt.subplots(1, 3, figsize=size, gridspec_kw={'width_ratios': [2.5, 2.5, 1]})  # 1 fila, 2 columnas

        t = np.linspace(-200, 700, 450)  # Mantiene 450 valores con pasos de 2 ms

        # Get average window from all subjects in df_0 with component = "all" and channel = canal
        avg_s_0 = Utils.avg_window(df_0, "all", channel, flag="mean")
        avg_s_1 = Utils.avg_window(df_1, "all", channel, flag="mean")
        avg_s_2 = Utils.avg_window(df_2, "all", channel, flag="mean")

        # Get median window from all subjects in df_0 with component = "all" and channel = canal
        med_s_0 = Utils.avg_window(df_0, "all", channel, flag="median")
        med_s_1 = Utils.avg_window(df_1, "all", channel, flag="median")
        med_s_2 = Utils.avg_window(df_2, "all", channel, flag="median")

        # Graficar la señal
        axes[0].plot(t, avg_s_0, label = l[0], alpha = 0.75)  # Representación de señal discreta
        axes[0].plot(t, avg_s_1, label = l[1], color="black", alpha = 0.65)  # Representación de señal discreta
        axes[0].plot(t, avg_s_2, label = l[2], color="red", alpha = 0.5)  # Representación de señal discreta

        axes[1].plot(t, med_s_0, label = l[0], alpha = 0.75)  # Representación de señal discreta
        axes[1].plot(t, med_s_1, label = l[1], color="black", alpha = 0.65)  # Representación de señal discreta
        axes[1].plot(t, med_s_2, label = l[2], color="red", alpha = 0.5)  # Representación de señal discreta

        for i in [0,1]:
            axes[i].axvline(x=-200, color='black', linestyle='--', linewidth=1, alpha = 1.0)
            axes[i].axvspan(-200, -100, color='black', alpha=0.1, label='baseline [-200, -100] ms')
            axes[i].axvline(x=-100, color='black', linestyle='--', linewidth=1, alpha = 1.0)
            axes[i].axvspan(-100, 0, color='gray', alpha=0.1, label='pre-trigger [-100, 0] ms')
            axes[i].axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha = 1.0)
            axes[i].axvspan(0, 80, color='#A8B98D', alpha=0.2, label='post-trigger [0, 80] ms')
            axes[i].axvline(x=80, color='gray', linestyle='--', linewidth=1, alpha = 1.0)
            axes[i].axvspan(80, 130, color='green', alpha=0.2, label='P1 [80, 130] ms')
            axes[i].axvline(x=130, color='gray', linestyle='--', linewidth=1, alpha = 1.0)
            axes[i].axvspan(130, 200, color='blue', alpha=0.1, label='N1 [130, 200] ms')
            axes[i].axvline(x=200, color='gray', linestyle='--', linewidth=1, alpha = 1.0)
            axes[i].axvspan(200, 300, color='red', alpha=0.1, label='P2 [200, 300] ms')
            axes[i].axvline(x=300, color='gray', linestyle='--', linewidth=1, alpha = 1.0)
            axes[i].axvspan(300, 360, color='orange', alpha=0.1, label='N2 [300, 360] ms')
            axes[i].axvline(x=360, color='gray', linestyle='--', linewidth=1, alpha = 1.0)
            axes[i].axvspan(360, 600, color='yellow', alpha=0.1, label='P3 [360, 600] ms')
            axes[i].axvline(x=600, color='gray', linestyle='--', linewidth=1, alpha = 1.0)

            # Configurar la grafica
            if i == 0:
                axes[i].set_title('[Mean - '+title)
            if i == 1:
                axes[i].set_title('[Median - '+title)
            axes[i].set_xlabel('[ms]')
            axes[i].set_ylabel('[$\\mu V$]')
            axes[i].set_xlim([-205, 700])
            axes[i].set_ylim([-5, 6])
            axes[i].grid()

        # Gráfico 3: Solo leyenda (sin ejes)
        axes[2].axis("off")  # Oculta los ejes
        # Crear proxies para la leyenda
        legend_patches = [
            mpatches.Patch(color='blue', alpha=1.0, label='CTRL'),
            mpatches.Patch(color='black', alpha=1.0, label='PLCB'),
            mpatches.Patch(color='red', alpha=1.0, label='EXP'),
            mpatches.Patch(color='black', alpha=0.1, label='baseline [-200, -100] ms'),
            mpatches.Patch(color='gray', alpha=0.1, label='pre-trigger [-100, 0] ms'),
            mpatches.Patch(color='#A8B98D', alpha=0.2, label='post-trigger [0, 80] ms'),
            mpatches.Patch(color='green', alpha=0.2, label='P1 [80, 130] ms'),
            mpatches.Patch(color='blue', alpha=0.1, label='N1 [130, 200] ms'),
            mpatches.Patch(color='red', alpha=0.1, label='P2 [200, 300] ms'),
            mpatches.Patch(color='orange', alpha=0.1, label='N2 [300, 360] ms'),
            mpatches.Patch(color='yellow', alpha=0.1, label='P3 [360, 600] ms')
        ]
        axes[2].legend(handles=legend_patches, loc="center", title="Leyenda")


        plt.tight_layout()
        plt.show()

        return
    
    def plot_band_two_groups(df_0, df_1, channels, band = "", params = [], title = "[TARGET] PRE: CTRL vs EXP", l = ["CTRL", "EXP"], c = ["b","r"], line = ["-","-"], size = (18, 7)):# CHECK --> OK

        rows,cols = params

        # Crear una figura con subplots de 'rows' filas y 'cols' columnas
        fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True, sharey=True)

        # Recorre cada canal y grafica en su respectivo subplot
        for idx, canal in enumerate(channels):

            i, j = divmod(idx, cols)# Convertir indice lineal a coordenadas de matriz (fila, columna)

            # Get average window from all subjects in df_0 with component = "all" and channel = canal
            s_0 = Utils.avg_window(df_0, "all", [canal])
            s_1 = Utils.avg_window(df_1, "all", [canal])

            # Apply PassBand-Filter 
            if band == "delta":
                params = {'lowcut':1,'highcut':4,'fs':500}
                s_0 = Utils.bandpass_filter(s_0, params, order=4)
                s_1 = Utils.bandpass_filter(s_1, params, order=4)
            if band == "theta":
                params = {'lowcut':4,'highcut':8,'fs':500}
                s_0 = Utils.bandpass_filter(s_0, params, order=4)
                s_1 = Utils.bandpass_filter(s_1, params, order=4)
            if band == "alpha":
                params = {'lowcut':8,'highcut':12,'fs':500}
                s_0 = Utils.bandpass_filter(s_0, params, order=4)
                s_1 = Utils.bandpass_filter(s_1, params, order=4)
            if band == "beta":
                params = {'lowcut':12,'highcut':30,'fs':500}
                s_0 = Utils.bandpass_filter(s_0, params, order=4)
                s_1 = Utils.bandpass_filter(s_1, params, order=4)

            t = np.linspace(-200, 700, 450)  # Mantiene 450 valores con pasos de 2 ms

            # Plot average windows
            axes[i, j].plot(t, s_0, label=l[0], color = c[0], linestyle = line[0], alpha = 0.5)
            axes[i, j].plot(t, s_1, label=l[1], color = c[1], linestyle = line[1], alpha = 0.5)

            # Set legends
            axes[i, j].legend()

            # Set X and Y limits
            #axes[i,j].set_ylim([-2.5, 2.5])
            axes[i,j].set_xlim([-200,700])

            # Set titles and grid
            axes[i, j].set_title(canal)
            axes[i, j].grid()

            # Set labels fow x and y axes
            if j == 0:
                axes[i, j].set_ylabel('[$\\mu V$]')
            if i == rows-1:
                axes[i, j].set_xlabel('[$ms$]')
                
        # Hide empty subplots (if necessary)
        for idx in range(len(channels), rows * cols):
            i, j = divmod(idx, cols)
            fig.delaxes(axes[i, j])

        # Set main title
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

        return
    
    def plot_band_three_groups(df_0, df_1, df_2, channels, band = "", params = [], title = "[...] PRE: CTRL, PLCB and EXP", l = ["CTRL", "PLCB", "EXP"], c = ["b","black","r"], line = ["-","-","-"], size = (18, 7)):# CHECK --> OK

        rows,cols = params

        # Crear una figura con subplots de 'rows' filas y 'cols' columnas
        fig, axes = plt.subplots(rows, cols, figsize=size, sharex=True, sharey=True)

        # Recorre cada canal y grafica en su respectivo subplot
        for idx, canal in enumerate(channels):

            i, j = divmod(idx, cols)# Convertir indice lineal a coordenadas de matriz (fila, columna)

            # Get average window from all subjects in df_0 with component = "all" and channel = canal
            s_0 = Utils.avg_window(df_0, "all", [canal])
            s_1 = Utils.avg_window(df_1, "all", [canal])
            s_2 = Utils.avg_window(df_2, "all", [canal])

            # Apply PassBand-Filter 
            if band == "delta":
                params = {'lowcut':1,'highcut':4,'fs':500}
                s_0 = Utils.bandpass_filter(s_0, params, order=4)
                s_1 = Utils.bandpass_filter(s_1, params, order=4)
                s_2 = Utils.bandpass_filter(s_2, params, order=4)
            if band == "theta":
                params = {'lowcut':4,'highcut':8,'fs':500}
                s_0 = Utils.bandpass_filter(s_0, params, order=4)
                s_1 = Utils.bandpass_filter(s_1, params, order=4)
                s_2 = Utils.bandpass_filter(s_2, params, order=4)
            if band == "alpha":
                params = {'lowcut':8,'highcut':12,'fs':500}
                s_0 = Utils.bandpass_filter(s_0, params, order=4)
                s_1 = Utils.bandpass_filter(s_1, params, order=4)
                s_2 = Utils.bandpass_filter(s_2, params, order=4)
            if band == "beta":
                params = {'lowcut':12,'highcut':30,'fs':500}
                s_0 = Utils.bandpass_filter(s_0, params, order=4)
                s_1 = Utils.bandpass_filter(s_1, params, order=4)
                s_2 = Utils.bandpass_filter(s_2, params, order=4)

            t = np.linspace(-200, 700, 450)  # Mantiene 450 valores con pasos de 2 ms

            # Plot average windows
            axes[i, j].plot(t, s_0, label=l[0], color = c[0], linestyle = line[0], alpha = 0.5)
            axes[i, j].plot(t, s_1, label=l[1], color = c[1], linestyle = line[1], alpha = 0.5)
            axes[i, j].plot(t, s_2, label=l[2], color = c[2], linestyle = line[2], alpha = 0.5)

            # Set legends
            axes[i, j].legend()

            # Set X and Y limits
            axes[i,j].set_ylim([-2.5, 2.5])
            axes[i,j].set_xlim([-200,700])

            # Set titles and grid
            axes[i, j].set_title(canal)
            axes[i, j].grid()

            # Set labels fow x and y axes
            if j == 0:
                axes[i, j].set_ylabel('[$\\mu V$]')
            if i == rows-1:
                axes[i, j].set_xlabel('[$ms$]')
                
        # Hide empty subplots (if necessary)
        for idx in range(len(channels), rows * cols):
            i, j = divmod(idx, cols)
            fig.delaxes(axes[i, j])

        # Set main title
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

        return
    
    def all_stimuli(dfs, group, c, size = (15,5)):
        fig, axs = plt.subplots(1, 2, figsize=size, sharey=True)

        fig.suptitle('Channel: Fz - Group: '+group, fontsize=16)

        t = np.linspace(-200, 700, 450)  # Mantiene 450 valores con pasos de 2 ms

        for i,df in enumerate(dfs):

            # Guardara la suma de ventanas para ambas pruebas -> [1, 2]
            m = np.zeros((0,450), dtype=float)

            # Por cada prueba oddball -> [1, 2]
            for n in [1, 2]:
                # Seleccionamos los sujetos con la prueba 1 o 2
                df_n = df[df["n_test"] == n]

                # Seleccionamos el id de los sujetos que tienen oddball 1 o 2
                ids = df_n['id'].unique()

                # Por cada id, entonces:
                for id in ids:
                    
                    # Seleccionamos la ventana del componente y canal correspondiente
                    w_0 = df_n[(df_n["comp"] == "all") & (df_n["id"] == id)]["Fz"].to_numpy()

                    axs[i].scatter(t,w_0, alpha = 0.1, c = c[0])
                    
                    # Guardamos los valores de ventana en filas
                    m = np.vstack([m, w_0])

            mu = np.mean(m, axis = 0)
            sigma = np.std(m, axis = 0)

            axs[i].plot(t, mu, alpha = 1.0, c = c[1], label="$\\mu$")
            axs[i].plot(t, mu+sigma, alpha = 1.0, c = c[2], linestyle="--", label="$\\mu + \\sigma$")
            axs[i].plot(t, mu-sigma, alpha = 1.0, c = c[3], linestyle="--", label="$\\mu - \\sigma$")

            if i == 0:
                axs[i].set_title("PRE")
            else:
                axs[i].set_title("POST")

            axs[i].set_ylabel("Amplitude $[\\mu V]$")
            axs[i].set_xlabel("Time [ms]")
            axs[i].set_xlim([-200,700])
            axs[i].legend()
            axs[i].grid()    

        plt.tight_layout()
        plt.show()

        return