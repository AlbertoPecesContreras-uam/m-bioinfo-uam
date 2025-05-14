
class Rows:
    def __init__(self):
        # Select all instants, ids, and labels from dataframe 
        self.instantes = []
        self.ids = []
        self.grupos = []
        
        # Save idx positions for PRE, POST, and SEG instants
        self.iPre = []
        self.iPost = []
        self.iSeg = []

        self.iPreCtrl = []
        self.iPostCtrl = []
        self.iSegCtrl = []

        self.iPreExp = []
        self.iPostExp = []
        self.iSegExp = []
        return
    
    def set_data(self, df_in):

        self.instantes = df_in.iloc[:,0]
        self.ids = df_in.iloc[:,1]
        self.grupos = df_in.iloc[:,2]

        self.iPre = df_in[df_in['instante']=="PRE"].index
        self.iPost = df_in[df_in['instante']=="POST"].index
        self.iSeg = df_in[df_in['instante']=="SEG"].index
        
        # Select labels from every instant
        grupos_pre = self.grupos.iloc[self.iPre]
        grupos_post = self.grupos.iloc[self.iPost]
        grupos_seg = self.grupos.iloc[self.iSeg]

        # Select Ctrl positions in every instant
        self.iPreCtrl = grupos_pre[grupos_pre=="Control"].index
        self.iPostCtrl = grupos_post[grupos_post=="Control"].index
        self.iSegCtrl = grupos_seg[grupos_seg=="Control"].index
        # Select Exp positions in every instant
        self.iPreExp = grupos_pre[grupos_pre=="Exp"].index
        self.iPostExp = grupos_post[grupos_post=="Exp"].index
        self.iSegExp = grupos_seg[grupos_seg=="Exp"].index

        return
"""
# Select all instants, ids, and labels from dataframe 
instantes = df.iloc[:,0]
ids = df.iloc[:,1]
grupos = df.iloc[:,2]

idx_pre = df[df['instante']=="PRE"].index
idx_post = df[df['instante']=="POST"].index
idx_seg = df[df['instante']=="SEG"].index

# Select labels from every instant
grupos_pre = grupos.iloc[idx_pre]
grupos_post = grupos.iloc[idx_post]
grupos_seg = grupos.iloc[idx_seg]

# Select control positions in every instant
rows_pre_ctrl = grupos_pre[grupos_pre=="Control"].index
rows_post_ctrl = grupos_post[grupos_post=="Control"].index
rows_seg_ctrl = grupos_seg[grupos_seg=="Control"].index

# Select exp positions in every instant
rows_pre_exp = grupos_pre[grupos_pre=="Exp"].index
rows_post_exp = grupos_post[grupos_post=="Exp"].index
rows_seg_exp = grupos_seg[grupos_seg=="Exp"].index
"""