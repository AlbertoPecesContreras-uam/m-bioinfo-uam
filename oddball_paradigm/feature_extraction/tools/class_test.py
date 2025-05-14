import numpy as np

class Test:
    def __init__(self):
        self.tipo = np.array([],dtype=str)
        self.df = []
        self.filename = np.array([],dtype=str)
        self.id = np.array([],dtype=int)
        self.grupo = np.array([],dtype=str)
        self.instante = np.array([],dtype=str)
        return
    
    def set_tipo(self, t_in):
        self.tipo = np.append(self.tipo, t_in)
        return
    def set_filename(self, f_in):
        self.filename = np.append(self.filename, f_in)
        return
    def set_instante(self, i_in):
        self.instante = np.append(self.instante, i_in)
        return
    def set_grupo(self, g_in):
        self.grupo = np.append(self.grupo, g_in)
        return
    def set_id(self, id_in):
        self.id = np.append(self.id, id_in)
        return
    def set_df(self, df_in):
        self.df.append(df_in)
        return
    
    def clear(self):
        np.delete(self.tipo, np.arange(self.tipo.size))
        np.delete(self.filename, np.arange(self.filename.size))
        np.delete(self.id, np.arange(self.id.size))
        np.delete(self.grupo, np.arange(self.grupo.size))
        np.delete(self.instante, np.arange(self.instante.size))
        self.df.clear()
        return
