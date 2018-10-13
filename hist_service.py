import pickle
import time
import pandas as pd
import requests
import numpy as np
from poloniex import Poloniex
from datetime import date, timedelta, datetime 
import os
from ephemGravityWrapper import gravity as gbaby
'''
As can be expected by this point, you will notice that 
nothing done here has been done in the best possible way
so feel STRONGLY ENCOURAGED TO FORK AND FIX/UPDATE/REFACTOR,
also for the sake of not running computations for computations 
sake instead of calculating the actual gravitational pull we
will just tack on a column of moon distances since its porportional 
to the gravitational pull
and occurs at the same intervals
'''

'''
the properties for histworker are set for the most part in combine frames which is called from the constructor
'''
class HistWorker:
    
    def get_hist_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), 'histories'))
        return histFiles

    def get_data_frame(self, fname):
        frame = pd.read_csv('./histories/'+fname) # timestamps will but used as index    
        return frame
        
    def get_file_symbol(self, f):
        f = f.split("_", 2)
        return f[1]

    def get_data_for_astro(self):
        data = {}
        dates = []
        md = []
        l_of_frame = len(self.currentHists['DASH']['date'])
        for snoz in range(0, l_of_frame):
            new_date = datetime.utcfromtimestamp((self.currentHists['DASH']['date'][snoz])).strftime('%Y-%m-%d  %H:%M:%S')
            dates.append(self.currentHists['DASH']['date'][snoz])
            md.append(gbaby.planet_dist(new_date, 'moon'))
        
        data = {'date': dates, 'moon_dist': md}
        data = pd.DataFrame.from_dict(data)
        data.to_csv("moon_dists.txt", encoding="utf-8")
        #data = data.join(self.currentHists['DASH'].set_index('date'), on='date', how="left").drop('Unnamed: 0', 1)
        return data.head()

    def read_in_moon_data(self, df):
        moon = pd.read_csv('./moon_dists.txt')
        moon.set_index("date")
        moon.drop("Unnamed: 0", 1)
        df = df.drop('Unnamed: 0', 1).set_index("date")
        return moon.join(df, on="date")

    def pull_polo(self):
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
                    frame.to_csv("./histories/"+coin+"_hist.txt", encoding="utf-8")
                except:
                    print("error reading json")
        self.get_data_for_astro()

    
    def combine_frames(self):
        fileNames = self.get_hist_files()
        for x in range(0,len(fileNames)):
            df = self.get_data_frame(fileNames[x])
            col_prefix = self.get_file_symbol(fileNames[x])
            self.coin_dict[x] = col_prefix
            #df.drop("Unnamed: 0", 1)
            #df = self.read_in_moon_data(df)
            df = df.drop("Unnamed: 0", 1)
            #df.rename(columns = lambda x: col_prefix+'_'+x, inplace=True)
            self.currentHists[col_prefix] = df
            self.hist_shaped[x] = np.array(df)
        self.hist_shaped = pd.Series(self.hist_shaped)
        '''
        main = df_list[0]
        for i in range(1, len(df_list)):
            main = main.join(df_list[i])
        return main
        '''
    def __init__(self):
        self.currentHists = {}
        self.hist_shaped = {}
        self.coin_dict = {}
        self.combine_frames()
        return
'''
hs = HistWorker()

print(hs.currentHists['DASH']['date'][0])
print(hs.currentHists['DASH']['date'][14])
print(hs.hist_shaped[0][0][0])
print(hs.hist_shaped[0][14][0])
'''