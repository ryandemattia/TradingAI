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


