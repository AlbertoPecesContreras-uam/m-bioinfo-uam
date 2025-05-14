import matplotlib.pyplot as plt
import numpy as np
class Plots:

    def scree_plot(explained_variance, point, title = 'Scree Plot', size = (10,5)):
        # Crear el gráfico
        x_ticks = range(1, len(explained_variance) + 1, 2)  # Cada 2 componentes
        plt.figure(figsize=size)
        
        # Graficar la varianza explicada
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', label='Explained Variance')
        
        # Dibujar línea vertical en el punto de corte
        plt.axvline(x=point, color='r', linestyle='--', label=f'Cut-off Point: {point}', alpha=0.5)
        
        # Títulos y etiquetas
        plt.title(title)
        plt.xlabel('Componente Principal')
        plt.ylabel('Varianza Explicada')
        
        # Establecer los ticks del eje X (solo los seleccionados)
        plt.xticks(x_ticks)
        
        # Mostrar leyenda y cuadrícula
        plt.legend()
        plt.grid(True)
        
        # Mostrar el gráfico
        plt.show()
        return
    
    def boxplot_features(instants,
                         fit = "auto",
                         size=(20,6),
                         titles=["PRE","POST","SEG","CTRL","EXP"]):

        if len(instants) == 3:
            pre,post,seg = instants
            X_pre_ctrl,X_pre_exp = pre
            X_post_ctrl,X_post_exp = post
            X_seg_ctrl,X_seg_exp = seg
            
            if fit == "auto":
                fit_up = max(X_pre_ctrl+X_pre_exp+X_post_ctrl+X_post_exp+X_seg_ctrl+X_seg_exp) + 0.5
                fit_down = min(X_pre_ctrl+X_pre_exp+X_post_ctrl+X_post_exp+X_seg_ctrl+X_seg_exp) - 0.5
        elif len(instants) == 2:
            pre,post = instants
            X_pre_ctrl,X_pre_exp = pre
            X_post_ctrl,X_post_exp = post       
            if fit == "auto":
                fit_up = max(X_pre_ctrl+X_pre_exp+X_post_ctrl+X_post_exp) + 0.5
                fit_down = min(X_pre_ctrl+X_pre_exp+X_post_ctrl+X_post_exp) - 0.5


        fig, axs = plt.subplots(1, len(instants), figsize=size)  # 1 fila, 2 columnas

        axs[0].boxplot([X_pre_ctrl, X_pre_exp], labels=[titles[3], titles[4]])
        axs[0].set_title(titles[0])
        axs[0].set_ylabel("Valores")
        axs[0].set_ylim(fit_down, fit_up)
        axs[0].grid()

        axs[1].boxplot([X_post_ctrl, X_post_exp], labels=[titles[3], titles[4]])
        axs[1].set_title(titles[1])
        axs[1].set_ylabel("Valores")
        axs[1].set_ylim(fit_down, fit_up)
        axs[1].grid()
        if len(instants)==3:
            axs[2].boxplot([X_seg_ctrl, X_seg_exp], labels=[titles[3], titles[4]])
            axs[2].set_title(titles[2])
            axs[2].set_ylabel("Valores")
            axs[2].set_ylim(fit_down, fit_up)
            axs[2].grid()
        
        # Mostrar la figura
        plt.tight_layout()  
        plt.show()
        return
    

    def boxplot_groups_and_instants(instants,
                                    fit = "auto",
                                    size=(20,6),
                                    titles=["PRE","POST","SEG","CTRL","EXP"]):

        pre,post,seg = instants
        X_pre_ctrl,X_pre_exp = pre
        X_post_ctrl,X_post_exp = post
        X_seg_ctrl,X_seg_exp = seg

        if fit == "auto":
            fit = max(X_pre_ctrl+X_pre_exp+X_post_ctrl+X_post_exp+X_seg_ctrl+X_seg_exp) + 0.5

        fig, axs = plt.subplots(1, 3, figsize=size)  # 1 fila, 2 columnas

        axs[0].boxplot([X_pre_ctrl, X_pre_exp], labels=[titles[3], titles[4]])
        axs[0].set_title(titles[0])
        axs[0].set_ylabel("Valores")
        axs[0].set_ylim(0, fit)
        axs[0].grid()

        axs[1].boxplot([X_post_ctrl, X_post_exp], labels=[titles[3], titles[4]])
        axs[1].set_title(titles[1])
        axs[1].set_ylabel("Valores")
        axs[1].set_ylim(0, fit)
        axs[1].grid()

        axs[2].boxplot([X_seg_ctrl, X_seg_exp], labels=[titles[3], titles[4]])
        axs[2].set_title(titles[2])
        axs[2].set_ylabel("Valores")
        axs[2].set_ylim(0, fit)
        axs[2].grid()
        
        # Mostrar la figura
        plt.tight_layout()  
        plt.show()
        return

    def boxplot_by_groups(X_ctrl, X_exp, X_grupos, show="all", fit="auto", titles=["Comparacion de Grupos","Grupo Control","Grupo Experimento","Ambos grupos"], colours = ["blue","red"]):

        if max(X_exp) > max(X_ctrl):
            x_lim = max(X_exp)+0.5
        else:
            x_lim = max(X_ctrl)+0.5
        y_lim = x_lim

        
        if show == "all":
            fig, axs = plt.subplots(1, 4, figsize=(20, 6))  # 1 fila, 2 columnas
            # Primera gráfica: comparación de grupos
            axs[0].boxplot([X_ctrl, X_exp, X_grupos], labels=[titles[1], titles[2], "Ambos"])
            axs[0].set_title(titles[0])
            axs[0].set_ylabel("Valores")
            axs[0].grid()


            # Segunda gráfica: otra comparación (por ejemplo, con X_new)
            axs[1].scatter(X_ctrl, X_ctrl, c=colours[0], alpha = 0.5)
            axs[1].set_xlabel("X1")
            axs[1].set_ylabel("X2")
            axs[1].set_title(titles[1])
            axs[1].set_xlim(0, x_lim)
            axs[1].set_ylim(0, y_lim)

            axs[1].grid()

            axs[2].scatter(X_exp, X_exp, c=colours[1], alpha=0.5)
            axs[2].set_xlabel("X1")
            axs[2].set_ylabel("X2")
            axs[2].set_title(titles[2])
            axs[2].set_xlim(0, x_lim)
            axs[2].set_ylim(0, y_lim)

            axs[2].grid()


            for i,x in enumerate([X_ctrl, X_exp]):
                for j in range(len(x)):
                    if i == 0:
                        axs[3].scatter(x[j], x[j], c=colours[0], alpha = 0.5)
                    else:
                        axs[3].scatter(x[j], x[j], c=colours[1], alpha = 0.3)

            axs[3].set_xlabel("X1")
            axs[3].set_ylabel("X2")
            axs[3].set_title(titles[3])
            axs[3].set_xlim(0, x_lim)
            axs[3].set_ylim(0, y_lim)
            axs[3].grid()

        else:
            fig, axs = plt.subplots(1, 1, figsize=(7, 7))  # 1 fila, 2 columnas
            # Primera gráfica: comparación de grupos
            axs.boxplot([X_ctrl, X_exp], labels=[titles[1], titles[2]])
            axs.set_title(titles[0])
            axs.set_ylabel("Valores")
            if fit != "auto":
                axs.set_ylim(0, fit)
            axs.grid()

        # Mostrar la figura
        plt.tight_layout()  # Ajust
        plt.show()
        return

    def boxplot_by_instants(X_pre, X_post, X_seg, fit="auto"):


  
        fig, axs = plt.subplots(1, 1, figsize=(7, 7))  # 1 fila, 2 columnas
        # Primera gráfica: comparación de grupos
        axs.boxplot([X_pre, X_post, X_seg], labels=["PRE", "POST", "SEG"])
        axs.set_title("Comparacion de Instantes")
        axs.set_ylabel("Valores")
        if fit != "auto":
            axs.set_ylim(0, fit)
        axs.grid()

        """
        # Segunda gráfica: otra comparación (por ejemplo, con X_new)
        axs[1].scatter(X_pre, X_pre, c="green", alpha = 0.5)
        axs[1].set_xlabel("X1")
        axs[1].set_ylabel("X2")
        axs[1].set_title("Instante PRE")
        axs[1].grid()

        axs[2].scatter(X_post, X_post, c="orange", alpha=0.5)
        axs[2].set_xlabel("X1")
        axs[2].set_ylabel("X2")
        axs[2].set_title("Instante POST")
        axs[2].grid()

        axs[3].scatter(X_seg, X_seg, c="purple", alpha=0.5)
        axs[3].set_xlabel("X1")
        axs[3].set_ylabel("X2")
        axs[3].set_title("Instante SEG")
        axs[3].grid()


        for i,x in enumerate([X_ctrl, X_exp]):
            for j in range(len(x)):
                if i == 0:
                    axs[3].scatter(x[j], x[j], c="blue", alpha = 0.5)
                else:
                    axs[3].scatter(x[j], x[j], c="red", alpha = 0.3)

        axs[3].set_xlabel("X1")
        axs[3].set_ylabel("X2")
        axs[3].set_title("Ambos grupos")
        axs[3].grid()
        """
        # Mostrar la figura
        plt.tight_layout()  # Ajust
        plt.show()

        return

    def radius_plot_by_groups(df_stats_ctrl, df_stats_exp, fit = "auto", flag = "mean_std", titles=["Dispersion Control","Dispersion Experimento","Dispersion Ambos"], colours = ["blue","red"], bArrows = False):
        
        if flag == "mean_std":
            mu_exp = df_stats_exp["Mean"].iloc[0]
            std_exp = np.sqrt(df_stats_exp["Var"].iloc[0])

            mu_ctrl = df_stats_ctrl["Mean"].iloc[0]
            std_ctrl = np.sqrt(df_stats_ctrl["Var"].iloc[0])

            label_0 = "$\\sigma$"
            label_1 = "$\\mu$"

            inner_ctrl = mu_ctrl
            outter_ctrl = mu_ctrl+std_ctrl
            
            inner_exp = mu_exp
            outter_exp = mu_exp+std_exp
        else:
            median_exp = df_stats_exp["Median"].iloc[0]
            mu_exp = df_stats_exp["Mean"].iloc[0]

            median_ctrl = df_stats_ctrl["Median"].iloc[0]
            mu_ctrl = df_stats_ctrl["Mean"].iloc[0]

            label_0 = "$\\mu$"
            label_1 = "$Me$"
            
            inner_ctrl = median_ctrl
            outter_ctrl = mu_ctrl
            
            inner_exp = median_exp
            outter_exp = mu_exp

        x, y = 0, 0  # Coordenadas del centro de la circunferencia

        # Generar puntos en ángulo para la circunferencia
        theta = np.linspace(0, 2 * np.pi, 100)  # Ángulos de 0 a 2π

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 fila, 2 columnas

        axs[0].plot(x + (outter_ctrl) * np.cos(theta), y + (outter_ctrl) * np.sin(theta), label=label_0, linestyle='--', color = colours[0], alpha=0.5)
        axs[0].plot(x + inner_ctrl * np.cos(theta), y + inner_ctrl * np.sin(theta), label=label_1, color = colours[0], alpha = 0.5)
        axs[0].axhline(0, color='black',linewidth=0.5)
        axs[0].axvline(0, color='black',linewidth=0.5)
        if bArrows:
            axs[0].arrow(0, 0, 0.8*np.cos(np.pi/4), 0.8*np.sin(np.cos(np.pi/4)), head_width=0.25, head_length=0.4, fc=colours[0], ec='black')
            axs[0].arrow(0, 0, 1.0*np.cos(np.pi/4), 2.8*np.sin(np.cos(np.pi/4)), head_width=0.25, head_length=0.4, fc=colours[0], ec='black')

        if fit == "auto":
            if (outter_exp+0.5) > (outter_ctrl+0.5):
                axs[0].set_xlim(-(outter_exp+0.5), outter_exp+0.5)  
                axs[0].set_ylim(-(outter_exp+0.5), outter_exp+0.5)  
            else:
                axs[0].set_xlim(-(outter_ctrl+0.5), outter_ctrl+0.5)  
                axs[0].set_ylim(-(outter_ctrl+0.5), outter_ctrl+0.5) 
        else:
            axs[0].set_xlim(-(fit), fit)  
            axs[0].set_ylim(-(fit), fit)                

        axs[0].set_xlabel("Eje X")
        axs[0].set_ylabel("Eje Y")
        axs[0].set_title(titles[0])
        axs[0].legend()
        axs[0].grid()

        axs[1].plot(x + (outter_exp) * np.cos(theta), y + (outter_exp) * np.sin(theta), label=label_0, linestyle='--', color = colours[1], alpha=0.5)
        axs[1].plot(x + inner_exp * np.cos(theta), y + inner_exp * np.sin(theta), label=label_1, color = colours[1], alpha=0.5)
        axs[1].axhline(0, color='black',linewidth=0.5)
        axs[1].axvline(0, color='black',linewidth=0.5)
        if bArrows:
            axs[1].arrow(0, 0, -1.3*np.cos(np.pi/4), 0.8*np.sin(np.cos(np.pi/4)), head_width=0.25, head_length=0.4, fc=colours[1], ec='black')
            axs[1].arrow(0, 0, -2.5*np.cos(np.pi/4), 2.9*np.sin(np.cos(np.pi/4)), head_width=0.25, head_length=0.4, fc=colours[1], ec='black')
        if fit == "auto":
            if (outter_exp+0.5) > (outter_ctrl+0.5):
                axs[1].set_xlim(-(outter_exp+0.5), outter_exp+0.5)  
                axs[1].set_ylim(-(outter_exp+0.5), outter_exp+0.5)  
            else:
                axs[1].set_xlim(-(outter_ctrl+0.5), outter_ctrl+0.5)  
                axs[1].set_ylim(-(outter_ctrl+0.5), outter_ctrl+0.5) 
        else:
            axs[1].set_xlim(-(fit), fit)  
            axs[1].set_ylim(-(fit), fit)            

        axs[1].set_xlabel("Eje X")
        axs[1].set_ylabel("Eje Y")
        axs[1].set_title(titles[1])
        axs[1].legend()
        axs[1].grid()

        axs[2].plot(x + (outter_ctrl) * np.cos(theta), y + (outter_ctrl) * np.sin(theta), linestyle='--', color = colours[0], alpha=0.5)
        axs[2].plot(x + inner_ctrl * np.cos(theta), y + inner_ctrl * np.sin(theta), color = colours[0], alpha = 0.5)
        axs[2].plot(x + (outter_exp) * np.cos(theta), y + (outter_exp) * np.sin(theta), linestyle='--', color = colours[1], alpha=0.5)
        axs[2].plot(x + inner_exp * np.cos(theta), y + inner_exp * np.sin(theta), color = colours[1], alpha=0.5)
        axs[2].axhline(0, color='black',linewidth=0.5)
        axs[2].axvline(0, color='black',linewidth=0.5)
        if bArrows:
            axs[2].arrow(0, 0, 0.8*np.cos(np.pi/4), 0.8*np.sin(np.cos(np.pi/4)), head_width=0.25, head_length=0.4, fc=colours[0], ec='black')
            axs[2].arrow(0, 0, 1.0*np.cos(np.pi/4), 2.8*np.sin(np.cos(np.pi/4)), head_width=0.25, head_length=0.4, fc=colours[0], ec='black')
            axs[2].arrow(0, 0, -1.3*np.cos(np.pi/4), 0.8*np.sin(np.cos(np.pi/4)), head_width=0.25, head_length=0.4, fc=colours[1], ec='black')
            axs[2].arrow(0, 0, -2.5*np.cos(np.pi/4), 2.9*np.sin(np.cos(np.pi/4)), head_width=0.25, head_length=0.4, fc=colours[1], ec='black')
        if fit == "auto":
            if (outter_exp+0.5) > (outter_ctrl+0.5):
                axs[2].set_xlim(-(outter_exp+0.5), outter_exp+0.5)  
                axs[2].set_ylim(-(outter_exp+0.5), outter_exp+0.5)  
            else:
                axs[2].set_xlim(-(outter_ctrl+0.5), outter_ctrl+0.5)  
                axs[2].set_ylim(-(outter_ctrl+0.5), outter_ctrl+0.5) 
        else:
            axs[2].set_xlim(-(fit), fit)  
            axs[2].set_ylim(-(fit), fit)   

        axs[2].set_xlabel("Eje X")
        axs[2].set_ylabel("Eje Y")
        axs[2].set_title(titles[2])
        axs[2].grid()

        plt.tight_layout()
        plt.show()
        return
    
    def radius_plot_by_instants(df_stats_pre, df_stats_post, df_stats_seg, fit = "auto", bArrows = False):
        mu_pre = df_stats_pre["Mean"].iloc[0]
        std_pre = np.sqrt(df_stats_pre["Var"].iloc[0])

        mu_post = df_stats_post["Mean"].iloc[0]
        std_post = np.sqrt(df_stats_post["Var"].iloc[0])

        mu_seg = df_stats_seg["Mean"].iloc[0]
        std_seg = np.sqrt(df_stats_seg["Var"].iloc[0])

        x, y = 0, 0  # Coordenadas del centro de la circunferencia

        # Generar puntos en ángulo para la circunferencia
        theta = np.linspace(0, 2 * np.pi, 100)  # Ángulos de 0 a 2π

        fig, axs = plt.subplots(2, 2, figsize=(9, 9))  # 2 fila, 2 columnas

        axs[0,0].plot(x + (mu_pre+std_pre) * np.cos(theta), y + (mu_pre+std_pre) * np.sin(theta), label="$\\sigma$", linestyle='--', color = "green", alpha=0.5)
        axs[0,0].plot(x + mu_pre * np.cos(theta), y + mu_pre * np.sin(theta), label="$\\mu$", color = "green", alpha = 0.5)
        axs[0,0].axhline(0, color='black',linewidth=0.5)
        axs[0,0].axvline(0, color='black',linewidth=0.5)

        if fit == "auto":
            if (mu_post+std_post+0.5) > (mu_pre+std_pre+0.5) and (mu_post+std_post+0.5) > (mu_seg+std_seg+0.5):
                axs[0,0].set_xlim(-(mu_post+std_post+0.5), mu_post+std_post+0.5)  
                axs[0,0].set_ylim(-(mu_post+std_post+0.5), mu_post+std_post+0.5)  
            elif (mu_pre+std_pre+0.5) > (mu_seg+std_seg+0.5) and (mu_pre+std_pre+0.5) > (mu_post+std_post+0.5):
                axs[0,0].set_xlim(-(mu_pre+std_pre+0.5), mu_pre+std_pre+0.5)  
                axs[0,0].set_ylim(-(mu_pre+std_pre+0.5), mu_pre+std_pre+0.5) 
            else:
                axs[0,0].set_xlim(-(mu_seg+std_seg+0.5), mu_seg+std_seg+0.5)  
                axs[0,0].set_ylim(-(mu_seg+std_seg+0.5), mu_seg+std_seg+0.5) 
        else:
            axs[0,0].set_xlim(-(fit), fit)  
            axs[0,0].set_ylim(-(fit), fit) 

        axs[0,0].set_xlabel("Eje X")
        axs[0,0].set_ylabel("Eje Y")
        axs[0,0].set_title("Dispersion PRE")
        axs[0,0].legend()
        axs[0,0].grid()

        axs[0,1].plot(x + (mu_post+std_post) * np.cos(theta), y + (mu_post+std_post) * np.sin(theta), label="$\\sigma$", linestyle='--', color = "brown", alpha=0.5)
        axs[0,1].plot(x + mu_post * np.cos(theta), y + mu_post * np.sin(theta), label="$\\mu$", color = "orange", alpha=0.5)
        axs[0,1].axhline(0, color='black',linewidth=0.5)
        axs[0,1].axvline(0, color='black',linewidth=0.5)

        if fit == "auto":
            if (mu_post+std_post+0.5) > (mu_pre+std_pre+0.5) and (mu_post+std_post+0.5) > (mu_seg+std_seg+0.5):
                axs[0,1].set_xlim(-(mu_post+std_post+0.5), mu_post+std_post+0.5)  
                axs[0,1].set_ylim(-(mu_post+std_post+0.5), mu_post+std_post+0.5)  
            elif (mu_pre+std_pre+0.5) > (mu_seg+std_seg+0.5) and (mu_pre+std_pre+0.5) > (mu_post+std_post+0.5):
                axs[0,1].set_xlim(-(mu_pre+std_pre+0.5), mu_pre+std_pre+0.5)  
                axs[0,1].set_ylim(-(mu_pre+std_pre+0.5), mu_pre+std_pre+0.5) 
            else:
                axs[0,1].set_xlim(-(mu_seg+std_seg+0.5), mu_seg+std_seg+0.5)  
                axs[0,1].set_ylim(-(mu_seg+std_seg+0.5), mu_seg+std_seg+0.5) 
        else:
            axs[0,1].set_xlim(-(fit), fit)  
            axs[0,1].set_ylim(-(fit), fit) 

        axs[0,1].set_xlabel("Eje X")
        axs[0,1].set_ylabel("Eje Y")
        axs[0,1].set_title("Dispersion POST")
        axs[0,1].legend()
        axs[0,1].grid()

        axs[1,0].plot(x + (mu_seg+std_seg) * np.cos(theta), y + (mu_seg+std_seg) * np.sin(theta), label="$\\sigma$", linestyle='--', color = "purple", alpha=0.5)
        axs[1,0].plot(x + mu_seg * np.cos(theta), y + mu_seg* np.sin(theta), label="$\\mu$", color = "purple", alpha=0.5)
        axs[1,0].axhline(0, color='black',linewidth=0.5)
        axs[1,0].axvline(0, color='black',linewidth=0.5)

        if fit == "auto":
            if (mu_post+std_post+0.5) > (mu_pre+std_pre+0.5) and (mu_post+std_post+0.5) > (mu_seg+std_seg+0.5):
                axs[1,0].set_xlim(-(mu_post+std_post+0.5), mu_post+std_post+0.5)  
                axs[1,0].set_ylim(-(mu_post+std_post+0.5), mu_post+std_post+0.5)  
            elif (mu_pre+std_pre+0.5) > (mu_seg+std_seg+0.5) and (mu_pre+std_pre+0.5) > (mu_post+std_post+0.5):
                axs[1,0].set_xlim(-(mu_pre+std_pre+0.5), mu_pre+std_pre+0.5)  
                axs[1,0].set_ylim(-(mu_pre+std_pre+0.5), mu_pre+std_pre+0.5) 
            else:
                axs[1,0].set_xlim(-(mu_seg+std_seg+0.5), mu_seg+std_seg+0.5)  
                axs[1,0].set_ylim(-(mu_seg+std_seg+0.5), mu_seg+std_seg+0.5) 
        else:
            axs[1,0].set_xlim(-(fit), fit)  
            axs[1,0].set_ylim(-(fit), fit) 

        axs[1,0].set_xlabel("Eje X")
        axs[1,0].set_ylabel("Eje Y")
        axs[1,0].set_title("Dispersion SEG")
        axs[1,0].legend()
        axs[1,0].grid()

        axs[1,1].plot(x + (mu_pre+std_pre) * np.cos(theta), y + (mu_pre+std_pre) * np.sin(theta), label="$\\sigma$", linestyle='--', color = "green", alpha=0.5)
        axs[1,1].plot(x + mu_pre * np.cos(theta), y + mu_pre * np.sin(theta), label="$\\mu$", color = "green", alpha = 0.5)
        axs[1,1].plot(x + (mu_post+std_post) * np.cos(theta), y + (mu_post+std_post) * np.sin(theta), label="$\\sigma$", linestyle='--', color = "orange", alpha=0.5)
        axs[1,1].plot(x + mu_post * np.cos(theta), y + mu_post * np.sin(theta), label="$\\mu$", color = "orange", alpha=0.5)
        axs[1,1].plot(x + (mu_seg+std_seg) * np.cos(theta), y + (mu_seg+std_seg) * np.sin(theta), label="$\\sigma$", linestyle='--', color = "purple", alpha=0.5)
        axs[1,1].plot(x + mu_seg * np.cos(theta), y + mu_seg* np.sin(theta), label="$\\mu$", color = "purple", alpha=0.5)
        axs[1,1].axhline(0, color='black',linewidth=0.5)
        axs[1,1].axvline(0, color='black',linewidth=0.5)
        if fit == "auto":
            if (mu_post+std_post+0.5) > (mu_pre+std_pre+0.5) and (mu_post+std_post+0.5) > (mu_seg+std_seg+0.5):
                axs[1,1].set_xlim(-(mu_post+std_post+0.5), mu_post+std_post+0.5)  
                axs[1,1].set_ylim(-(mu_post+std_post+0.5), mu_post+std_post+0.5)  
            elif (mu_pre+std_pre+0.5) > (mu_seg+std_seg+0.5) and (mu_pre+std_pre+0.5) > (mu_post+std_post+0.5):
                axs[1,1].set_xlim(-(mu_pre+std_pre+0.5), mu_pre+std_pre+0.5)  
                axs[1,1].set_ylim(-(mu_pre+std_pre+0.5), mu_pre+std_pre+0.5) 
            else:
                axs[1,1].set_xlim(-(mu_seg+std_seg+0.5), mu_seg+std_seg+0.5)  
                axs[1,1].set_ylim(-(mu_seg+std_seg+0.5), mu_seg+std_seg+0.5) 
        else:
            axs[1,1].set_xlim(-(fit), fit)  
            axs[1,1].set_ylim(-(fit), fit)            

        axs[1,1].set_xlabel("Eje X")
        axs[1,1].set_ylabel("Eje Y")
        axs[1,1].set_title("Dispersion Total")
        axs[1,1].grid()

        plt.tight_layout()
        plt.show()
        return
    
    def plot_instant_vs_instant(X, ids, idx_pre, idx_post, params, b_arrows = False, b_id = False):
    
        X0_pre = X[idx_pre, 0]
        X1_pre = X[idx_pre, 1]
        ID_pre = ids.iloc[idx_pre]

        X0_post = X[idx_post, 0]
        X1_post = X[idx_post, 1]
        ID_post = ids.iloc[idx_post]

        list_pre = list(ID_pre)
        list_post = list(ID_post)

        # 2D-PLOT

        fig_2D = plt.figure(figsize=(10, 8))

        scatter = plt.scatter(X0_pre, X1_pre, c=params["groups_colors"][0], alpha=0.95,s=100)
        scatter = plt.scatter(X0_post, X1_post, c=params["groups_colors"][1], alpha=0.5, s=100)

        
        for i in range(len(list_pre)):
            
            if list_pre[i] in list_post:
                
                j = list_post.index(list_pre[i])

                #if b_id:
                    #plt.text(X0_pre[i], X1_pre[i], str(list_pre[i]), verticalalignment='bottom', horizontalalignment='right')
                    
                    #plt.text(X0_post[j], X1_post[j], str(list_post[j]), verticalalignment='bottom', horizontalalignment='right')

                if b_arrows:

                    if X0_post[j] - X0_pre[i] >= 0:
                        plt.plot([X0_pre[i], X0_post[j]], [X1_pre[i], X1_post[j]], color="darkblue", linestyle='-', linewidth=1)  # Línea azul

                    if X0_post[j] - X0_pre[i] < 0:
                        plt.plot([X0_pre[i], X0_post[j]], [X1_pre[i], X1_post[j]], color="green", linestyle='-', linewidth=1)  # Línea verde

        if b_id:
            for k in range(len(list_post)):
                plt.text(X0_post[k], X1_post[k], str(list_post[k]), verticalalignment='bottom', horizontalalignment='right')

            for m in range(len(list_pre)):
                plt.text(X0_pre[m], X1_pre[m], str(list_pre[m]), verticalalignment='bottom', horizontalalignment='right')
        # -------------

        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=params["groups_colors"][0], markersize=10, label=params["groups_title"][0]),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=params["groups_colors"][1], markersize=10, label=params["groups_title"][1])]

        plt.legend(handles=handles, title=params["legend_title"])

        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        plt.title(params["main_title"])
        plt.grid()

        return
    
    def plot_group_vs_group(X, ids, idx_pre, idx_post, params, b_id = False):
    
        X0_pre = X[idx_pre, 0]
        X1_pre = X[idx_pre, 1]
        ID_pre = ids.iloc[idx_pre]

        X0_post = X[idx_post, 0]
        X1_post = X[idx_post, 1]
        ID_post = ids.iloc[idx_post]

        list_pre = list(ID_pre)
        list_post = list(ID_post)

        # 2D-PLOT

        fig_2D = plt.figure(figsize=(10, 8))

        scatter = plt.scatter(X0_pre, X1_pre, c=params["groups_colors"][0], alpha=0.95,s=100)
        scatter = plt.scatter(X0_post, X1_post, c=params["groups_colors"][1], alpha=0.5, s=100)

        if b_id:
            for i in range(len(list_pre)):
                plt.text(X0_pre[i], X1_pre[i], str(list_pre[i]), verticalalignment='bottom', horizontalalignment='right')
            for j in range(len(list_post)):
                plt.text(X0_post[j], X1_post[j], str(list_post[j]), verticalalignment='bottom', horizontalalignment='right')

        #for k in range(len(list_post_exp)):
        #    plt.text(X0_post_exp[k], X1_post_exp[k], str(list_post_exp[k]), verticalalignment='bottom', horizontalalignment='right')

        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=params["groups_colors"][0], markersize=10, label=params["groups_title"][0]),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=params["groups_colors"][1], markersize=10, label=params["groups_title"][1])]

        plt.legend(handles=handles, title=params["legend_title"])

        plt.xlabel("Componente 1")
        plt.ylabel("Componente 2")
        plt.title(params["main_title"])
        plt.grid()

        return
    
    def boxplot_groups_by_components(df, cols, title, size = (10,6)):

        #cols = range(3, 22)  # index positions I want to plot from dataframef

        # Selected columns in list format for boxplot
        data_for_boxplot = [df.iloc[:, i] for i in cols]

        plt.figure(figsize=size)

        # Create boxplot
        plt.boxplot(data_for_boxplot)

        plt.xticks(range(1, len(data_for_boxplot) + 1), [f'C{i+1}' for i in range(len(data_for_boxplot))])

        # Title and labels
        plt.title(title)  
        plt.ylabel('Subjects')  
        plt.grid()
        plt.show()
        return
    
    def stem_groups_by_components(array0, array1, info = ["Title","Components","Median"], l=["label_0","label_1"], line=["r-","b-"], marker=["ro","bo"], size=(10,6)):
        plt.figure(figsize=size)  # Cambia (10, 6) a las dimensiones que prefieras

        # Crear el gráfico con líneas verticales y marcadores
        plt.stem(range(len(array0)), array0, linefmt=line[0], markerfmt=marker[0], basefmt=" ", label=l[0])  
        plt.stem(range(len(array1)), array1, linefmt=line[1], markerfmt=marker[1], basefmt=" ", label=l[1])  

        # Añadir etiquetas y título
        plt.title(info[0])
        plt.xlabel(info[1])
        plt.ylabel(info[2])

        # Mostrar leyenda
        plt.legend()

        # Activar la cuadrícula
        plt.grid(True)

        # Mostrar el gráfico
        plt.show()
        return
    
    def boxplot_paired_components(data0, data1, info=["C", "Subjects", "Title"], label=["label0", "label1"], c=["gray","blue"], size=(17,6), ylims = []):
        # Crear el gráfico
        plt.figure(figsize=size)

        # Alternar las posiciones de los boxplots entre los dos grupos
        positions = []
        for i in range(len(data0)):
            positions.append(i * 2 + 1)  # Posiciones impares para el grupo 1
            positions.append(i * 2 + 2)  # Posiciones pares para el grupo 2

        # Dibujar los boxplots de ambos grupos
        b1 = plt.boxplot(data0, positions=[i * 2 + 1 for i in range(len(data0))], widths=0.6, patch_artist=True, 
                    boxprops=dict(facecolor=c[0], color='black', alpha = 0.5),
                    capprops=dict(color='black'), whiskerprops=dict(color='black'))

        b2 = plt.boxplot(data1, positions=[i * 2 + 2 for i in range(len(data1))], widths=0.6, patch_artist=True, 
                    boxprops=dict(facecolor=c[1], color='black', alpha = 0.5),
                    capprops=dict(color='black'), whiskerprops=dict(color='black'))

        # Añadir etiquetas para las posiciones de los boxplots
        plt.xticks([i * 2 + 1.5 for i in range(len(data0))], [info[0]+str(i+1) for i in range(len(data0))])
        plt.legend([b1["boxes"][0], b2["boxes"][0]], [label[0], label[1]], loc='upper right')
        # Título y etiquetas
        plt.title(info[2])
        plt.ylabel(info[1])

        if len(ylims)!=0:
            plt.ylim(ylims)
        plt.grid(True)

        # Mostrar el gráfico
        plt.show()
        return
    
