import pandas as pd
import os
def get_hist_files():
    histFiles = os.listdir(os.path.join(os.path.dirname(__file__), 'histories'))
    return histFiles

def get_data_frame(fname):
    frame = pd.read_csv('./histories/'+fname)
    return frame

files = get_hist_files()

def get_file_symbol(f):
    f = f.split("_", 2)
    return f[1]
