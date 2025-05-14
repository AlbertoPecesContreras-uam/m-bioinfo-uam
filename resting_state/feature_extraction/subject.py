import numpy as np

class Subject:

    def __init__(self):
        self.tipo_test = np.array([],dtype=str)
        self.filename = np.array([],dtype=str)
        self.signals = []
        self.channels = []
        self.id = np.array([],dtype=int)
        self.grupo = np.array([],dtype=str)
        self.instante = np.array([],dtype=str)
        return  
    
    def set_tipo_test(self, t_in):
        self.tipo_test = np.append(self.tipo_test, t_in)
        return
    def set_filename(self, f_in):
        self.filename = np.append(self.filename, f_in)
        return
    def set_signals(self, s_in):
        self.signals.append(s_in)
        return 
    def set_channels(self, ch_in):
        self.channels.append(ch_in)
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
    
    def clear(self):
        np.delete(self.tipo_test, np.arange(self.tipo_test.size))
        np.delete(self.filename, np.arange(self.filename.size))
        self.signals.clear()
        self.channels.clear()
        np.delete(self.id, np.arange(self.id.size))
        np.delete(self.grupo, np.arange(self.grupo.size))
        np.delete(self.instante, np.arange(self.instante.size))
        return
    