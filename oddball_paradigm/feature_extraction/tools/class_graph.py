import matplotlib.pyplot as plt
import numpy as np

class Graph:
    def plot_graph(sync_matrix, data_apEn, channels, electrodes, title="TITLE", subtitle=["CTRL","PLCB","EXP"]):

        # df0_col, df1_col, df2_col = data_apEn

        # # Ordenamos los valores de entropia para que se correspondan con los electrodos
        # ctrl = [df0_col.loc[key] for key in electrodes.keys()]
        # plcb = [df1_col.loc[key] for key in electrodes.keys()]
        # exp = [df2_col.loc[key] for key in electrodes.keys()]

        # data_apEn = [ctrl, plcb, exp]

        # # Calculamos el valor minimo y maximo para normalizar los colores
        # lim_Apen = max(abs(min(df0_col.min(), df1_col.min(), df2_col.min())), 
        #             abs(max(df0_col.max(), df1_col.max(), df2_col.max())))
        
        circunferencias = np.arange(0, 1.5, 0.2)
        vmin = min(matrix.min() for matrix in sync_matrix)  # Encuentra el minimo global
        vmax = max(matrix.max() for matrix in sync_matrix)  # Encuentra el maximo global
        cut_off = np.median(np.abs(sync_matrix))

        # Dibujar los nodos
        x_coords, y_coords = zip(*electrodes.values())

        # Crear una figura con 1 fila y 4 columnas
        fig, axes = plt.subplots(1, 4, figsize=(18, 6), gridspec_kw={"width_ratios": [1, 1, 1, 0.05]})

        # Colocamos Titulo de la figura
        plt.suptitle('['+title+']'+r' Neural Response Graph ($\Delta$ POST-PRE)', fontsize=16)

        # Por cada eje o subplot
        for k, ax in enumerate(axes[:-1]):

            # Dibujamos aristas como nivel de coherencia
            for i in range(len(channels)):
                for j in range(i + 1, len(channels)):  # Evitar duplicados

                    weight = sync_matrix[k][i, j]# mismo orden en columnas y filas que lista 'channels'

                    if abs(weight) > cut_off:  # Filtrar conexiones debiles
                        # Nos aseguramos de seleccionar correctamente los canales en orden
                        x1, y1 = electrodes[channels[i]]
                        x2, y2 = electrodes[channels[j]]

                        normalized_weight = (weight - (-0.2)) / (0.2 - (-0.2))# Escala entre 0 y 1

                        color = plt.cm.seismic(normalized_weight)# Necesita valores entre 0 y 1 para adjudicar colores correctamente

                        linewidth = abs(weight)*15  # Grosor proporcional
                        # Graficamos
                        sc = ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, alpha=0.8)

            # Fondo gris
            ax.set_facecolor('lightgray')

            ax.set_title(subtitle[k])
            #sc = ax.scatter(x_coords, y_coords, c=data_apEn[k], cmap="coolwarm", s=200, edgecolors="black", vmin=-lim_Apen, vmax=+lim_Apen, zorder=3)
            ax.scatter(x_coords, y_coords, c="gray", s=200, edgecolors="black", zorder=3)
            ax.set_xticks([])  # Eliminar las marcas en el eje X
            ax.set_yticks([])  # Eliminar las marcas en el eje Y
            ax.set_xlim([-1.2, 1.2])
            ax.set_ylim([-1.0, 1.0])

            # Agregar etiquetas con alineacion condicional
            for label, (x, y) in electrodes.items():
                ha = "left" if x > 0 else "right" if x < 0 else "center"
                ax.text(x, y+0.05, label, fontsize=12, ha=ha, va='bottom')

            # Dibujar las circunferencias en el gráfico
            for r in circunferencias:
                circle = plt.Circle((0,0), r, color='gray', fill=False, linestyle='--', linewidth=0.5)
                ax.add_artist(circle)

        # Ocultamos los ejes de la cuarta y quinta figura
        axes[-1].axis('off')

        # Crear la barra de colores en la cuarta columna 
        cbar = fig.colorbar(sc, ax=axes, orientation='vertical', fraction=0.02, pad=0.01)
        cbar.set_clim(vmin, vmax)  # Establecer los límites de la colorbar entre 0 y 1
        cbar.set_label('Increase (red) to Decrease (blue)', fontsize=12, rotation=270, labelpad=20)
        cbar.set_ticks([])

        # Ajustamos diseño manualmente
        fig.subplots_adjust(left=0.05, right=0.9, wspace=0.15)
        plt.show()
        return