import pandas as pd
import os

class HistWorker:

    currentHists = {}
    
    def get_hist_files():
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), 'histories'))
        return histFiles

    def get_data_frame(fname):
        frame = pd.read_csv('./histories/'+fname)
        return frame

    def append_coin_to_col_names(df, c_name):
        
    def get_file_symbol(f):
    f = f.split("_", 2)
    return f[1]

    def __init__(self):
        return self

    
