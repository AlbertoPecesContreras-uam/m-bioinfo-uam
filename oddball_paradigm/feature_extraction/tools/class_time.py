import matplotlib.pyplot as plt
import numpy as np

class Time:
    def plot_distributions(dfs, cols, components, title = "[---] Tiempos de respuesta"):

        # Iterar sobre cada componente y crear una figura individual para cada uno
        for comp in components:  
            # Crear una figura con una grilla de 1 fila × 3 columnas (una por grupo)
            fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)

            fig.suptitle(f"{title} - Componente {comp}", fontsize=16)

            # Diccionario con los datos de tiempo
            dict_time = {
                'CTRL': [dfs['PRE'][0][dfs['PRE'][0]['comp'] == comp].loc[:, cols].to_numpy(float).flatten()*1000,
                        dfs['POST'][0][dfs['POST'][0]['comp'] == comp].loc[:, cols].to_numpy(float).flatten()*1000],
                'PLCB': [dfs['PRE'][1][dfs['PRE'][1]['comp'] == comp].loc[:, cols].to_numpy(float).flatten()*1000,
                        dfs['POST'][1][dfs['POST'][1]['comp'] == comp].loc[:, cols].to_numpy(float).flatten()*1000],
                'EXP': [dfs['PRE'][2][dfs['PRE'][2]['comp'] == comp].loc[:, cols].to_numpy(float).flatten()*1000,
                        dfs['POST'][2][dfs['POST'][2]['comp'] == comp].loc[:, cols].to_numpy(float).flatten()*1000]
            }

            # Iterar sobre cada grupo y cada columna
            for col_idx, (grupo, instante) in enumerate(dict_time.items()):

                ax = axs[col_idx]  # Seleccionar el subplot correspondiente

                # Graficar histogramas
                ax.hist(instante[0], bins=30, alpha=0.5, density=True, label="PRE", color="green", edgecolor="gray")
                ax.hist(instante[1], bins=30, alpha=0.5, density=True, label="POST", color="brown", edgecolor="gray")

                # Líneas de media
                ax.axvline(np.median(instante[0]), color="green", linestyle="solid", linewidth=1.5, label='Median')
                ax.axvline(np.median(instante[1]), color="brown", linestyle="dashed", linewidth=1.5, label='Median')

                # Configuración de cada subplot
                ax.set_title(f"{grupo}")  # Etiqueta del grupo en la parte superior
                ax.set_xlabel("Tiempo [ms]")

                ax.legend()
                ax.grid()

                if col_idx == 0:
                    ax.set_ylabel(f"{comp}")  # Etiqueta del componente solo en la primera columna
                
            # Ajustar el diseño para evitar solapamientos
            fig.tight_layout()
            # Mostrar la figura
            plt.show()
        return