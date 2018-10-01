import pickle
import time
import pandas as pd
import requests
import numpy as np
from poloniex import Poloniex
from datetime import date, timedelta, datetime 
import os

class HistWorker:

    currentHists = {}
    
    def get_hist_files():
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), 'histories'))
        return histFiles

    def get_data_frame(fname):
        frame = pd.read_csv('./histories/'+fname)
        return frame
        
    def get_file_symbol(f):
        f = f.split("_", 2)
        return f[1]

    def pull_polo():
        polo = Poloniex()
        coins = polo.returnTicker()
        tickLen = '7200'
        start = datetime.today() - timedelta(365) 
        start = str(int(start.timestamp()))
        for coin in coins:
            if coin[:3] == 'BTC':
                hist = requests.get('https://poloniex.com/public?command=returnChartData&currencyPair='+coin+'&start='+start+'&end=9999999999&period='+tickLen)
                try:
                    frame = pd.DataFrame(hist.json())
                    print(frame.head())
                    frame.to_csv(coin+"_hist.txt", encoding="utf-8")
                except:
                    print("error reading json")

    
    def combine_frames(self):
        fileNames = get_hist_files()
        for x in range(0,len(fileNames)):
            df = get_data_frame(fileNames[x])
            col_prefix = get_file_symbol(fileNames[x])
            #df.rename(columns = lambda x: col_prefix+'_'+x, inplace=True)
            self.currentHists[col_prefix] = df
        '''
        main = df_list[0]
        for i in range(1, len(df_list)):
            main = main.join(df_list[i])
        return main
        '''
    def __init__(self):
        self.combine_frames()
        return

    
