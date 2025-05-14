import numpy as np
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftfreq

class Features:
    def extract(s):
        """
        Esta function es capaz de extraer varias caracteristicas
        a partir de una signal "s" con una frecuencia de muestreo
        "fs". 
        Caracteristicas:
            - Media y Mediana: tendencia central de la señal
            - Varianza y desviación estándar: dispersión de los datos
            - Curtosis y Skewness: forma de la señal (picos, colas, ... etc.)
            - Rango (abs(mean(>0)-mean(<0))): amplitud de las variaciones
            - Energía de la Señal: suma de los cuadrados de los valores
            - Frecuencia dominante
            - Intensidad máxima asociada a frecuencia dominante
        s: array vector (1 x n)
        fs: integer [ms]
        output: formato dict{"feat_1":float,"feat_2":float,...}
        """
        # 1. Media y Mediana
        media = np.mean(s)

        # 2. Varianza y Desviación Estándar
        varianza = np.var(s)

        # Resultado
        caracteristicas = {
            "mean": media,
            "var": varianza
        }

        return caracteristicas
    