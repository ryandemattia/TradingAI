import pickle
import time
import pandas as pd
import requests
import numpy as np
from poloniex import Poloniex
from binance.client import Client
from datetime import date, timedelta, datetime
import os
#from ephemGravityWrapper import gravity as gbaby
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
    look_back = 0
    def __init__(self):
        self.currentHists = {}
        self.hist_shaped = {}
        self.coin_dict = {}
        #self.combine_frames()
        self.look_back = 666
        self.hist_full_size = 666*12
        self.binance_client = Client("PBmYxFlOc2PSJb9KVSOUXLrsdqsG7bGTZ6suaTuTYRBCdMWo4Pn0d4Z93kp21Kzd","79uw7k4drsKFL66i8J3LB6KSq35O2W2PEydIgY0tHLwURhXemVCfsAY63XdN3G6A")
        return

    def get_hist_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), 'histories'))
        return histFiles

    def get_binance_hist_files(self):
        binanceFiles = os.listdir(os.path.join(os.path.dirname(__file__), 'binance_hist'))
        return binanceFiles

    def get_live_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), 'paper'))
        return histFiles

    def get_gdax_training_files(self):
        histFiles = os.listdir(os.path.join(os.path.dirname(__file__), '../gdax'))
        return histFiles

    def get_data_frame(self, fname):
        frame = pd.read_csv('./histories/'+fname) # timestamps will but used as index
        return frame

    def get_live_data_frame(self, fname):
        frame = pd.read_csv('./paper/'+fname)
        return frame

    def get_file_as_frame(self, fname):
        frame = pd.read_csv('../gdax/'+fname)
        return frame

    def get_file_symbol(self, f):
        f = f.split("_", 2)
        return f[1]
    '''
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
    '''
    def read_in_moon_data(self, df):
        moon = pd.read_csv('./moon_dists.txt')
        moon.set_index("date")
        moon.drop("Unnamed: 0", 1)
        df = df.drop('Unnamed: 0', 1).set_index("date")
        return moon.join(df, on="date")

    #BEGIN BINANCE METHODS

    def pull_binance_symbols(self):
        sym_list = []
        for x in self.binance_client.get_products()["data"]:
            sym_list.append(x["symbol"])
        return sym_list

    def get_binance_hist_frame(self, symbol):
        frame = hs.binance_client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE, "1 May, 2018", "1 Jan, 2019")
        for x in range(len(frame)):
            frame[x] = frame[x][:6]
        frame = pd.DataFrame(frame, columns=["date", "open", "high", "low", "close", "volume"])
        print(frame.head())
        return frame
    
    def write_binance_training_files(self, syms):
        for s in range(len(syms)):
            frame = self.get_binance_hist_frame(syms[s])
            frame['avg_vol_3'] = frame['volume'].rolling(3).mean()
            frame['avg_vol_13'] = frame['volume'].rolling(13).mean()
            frame['avg_vol_34'] = frame['volume'].rolling(34).mean()
            frame['avg_close_3'] = frame['close'].rolling(3).mean()
            frame['avg_close_13'] = frame['close'].rolling(13).mean()
            frame['avg_close_34'] = frame['close'].rolling(34).mean()
            frame.fillna(value=-99999, inplace=True)
            frame.to_csv("./binance_hist/"+syms[s]+"_hist.txt", encoding="utf-8")


    def pull_polo_live(self, lb):
        polo = Poloniex()
        coins = polo.returnTicker()
        tickLen = '7200'
        start = datetime.today() - timedelta(lb)
        start = str(int(start.timestamp()))
        for coin in coins:
            if coin[:3] == 'BTC':
                #print(coin)
                hist = requests.get('https://poloniex.com/public?command=returnChartData&currencyPair='+coin+'&start='+start+'&end=9999999999&period='+tickLen)
                h_frame = pd.DataFrame(hist.json())
                frame = h_frame.copy()
                frame['avg_vol_3'] = frame['volume'].rolling(3).mean()
                frame['avg_vol_13'] = frame['volume'].rolling(13).mean()
                frame['avg_vol_34'] = frame['volume'].rolling(34).mean()
                frame['avg_close_3'] = frame['close'].rolling(3).mean()
                frame['avg_close_13'] = frame['close'].rolling(13).mean()
                frame['avg_close_34'] = frame['close'].rolling(34).mean()
                frame.fillna(value=-99999, inplace=True)
                print(coin + " written")
                frame.to_csv("./paper/"+coin+"_hist.txt", encoding="utf-8")


    def pull_polo(self):
        polo = Poloniex()
        coins = polo.returnTicker()
        tickLen = '7200'
        start = datetime.today() - timedelta(self.look_back)
        start = str(int(start.timestamp()))
        for coin in coins:
            if coin[:3] == 'BTC':
                hist = requests.get('https://poloniex.com/public?command=returnChartData&currencyPair='+coin+'&start='+start+'&end=9999999999&period='+tickLen)
                try:
                    h_frame = pd.DataFrame(hist.json())
                    frame = h_frame.copy()
                    frame['avg_vol_3'] = frame['volume'].rolling(3).mean()
                    frame['avg_vol_13'] = frame['volume'].rolling(13).mean()
                    frame['avg_vol_34'] = frame['volume'].rolling(34).mean()
                    frame['avg_close_3'] = frame['close'].rolling(3).mean()
                    frame['avg_close_13'] = frame['close'].rolling(13).mean()
                    frame['avg_close_34'] = frame['close'].rolling(34).mean()
                    frame = frame.fillna(value=-99999, inplace=True)
                    print(frame.head())
                    frame.to_csv("./histories/"+coin+"_hist.txt", encoding="utf-8")
                except:
                    print("error reading json")
        #self.get_data_for_astro()

    def combine_frames(self):
        length = 7992
        fileNames = self.get_hist_files()
        coin_and_hist_index = 0
        for x in range(0,len(fileNames)):
            df = self.get_data_frame(fileNames[x])
            col_prefix = self.get_file_symbol(fileNames[x])
            #df.drop("Unnamed: 0", 1)
            #df = self.read_in_moon_data(df)
            df = df.drop("Unnamed: 0", 1)
            #df.rename(columns = lambda x: col_prefix+'_'+x, inplace=True)

            as_array = np.array(df)

            #print(len(as_array))
            if(len(as_array) == length):
                self.currentHists[col_prefix] = df
                df = (df - df.mean()) / (df.max() - df.min())
                as_array=np.array(df)
                self.hist_shaped[coin_and_hist_index] = as_array
                self.coin_dict[coin_and_hist_index] = col_prefix
                coin_and_hist_index += 1
        self.hist_shaped = pd.Series(self.hist_shaped)
        
    def combine_frames(self):
        length = 7992
        fileNames = self.get_hist_files()
        coin_and_hist_index = 0
        for x in range(0,len(fileNames)):
            df = self.get_data_frame(fileNames[x])
            col_prefix = self.get_file_symbol(fileNames[x])
            #df.drop("Unnamed: 0", 1)
            #df = self.read_in_moon_data(df)
            df = df.drop("Unnamed: 0", 1)
            #df.rename(columns = lambda x: col_prefix+'_'+x, inplace=True)

            as_array = np.array(df)

            #print(len(as_array))
            if(len(as_array) == length):
                self.currentHists[col_prefix] = df
                df = (df - df.mean()) / (df.max() - df.min())
                as_array=np.array(df)
                self.hist_shaped[coin_and_hist_index] = as_array
                self.coin_dict[coin_and_hist_index] = col_prefix
                coin_and_hist_index += 1
        self.hist_shaped = pd.Series(self.hist_shaped)
        '''
        main = df_list[0]
        for i in range(1, len(df_list)):
            main = main.join(df_list[i])
        return main
        '''

    def combine_live_frames(self, length):
        fileNames = self.get_live_files()
        coin_and_hist_index = 0
        for x in range(0,len(fileNames)):
            df = self.get_live_data_frame(fileNames[x])
            col_prefix = self.get_file_symbol(fileNames[x])
            #df.drop("Unnamed: 0", 1)
            #df = self.read_in_moon_data(df)
            df = df.drop("Unnamed: 0", 1)
            #df.rename(columns = lambda x: col_prefix+'_'+x, inplace=True)
            as_array = np.array(df)
            #print(len(as_array))
            if(len(as_array) > length):
                self.currentHists[col_prefix] = df
                df = (df - df.mean()) / (df.max() - df.min())
                as_array = np.array(df)
                self.hist_shaped[coin_and_hist_index] = as_array
                self.coin_dict[coin_and_hist_index] = col_prefix
                coin_and_hist_index += 1
        self.hist_shaped = pd.Series(self.hist_shaped)
        '''
        main = df_list[0]
        for i in range(1, len(df_list)):
            main = main.join(df_list[i])
        return main
        '''


    def combine_live_usd_frames(self):
        fileNames = self.get_gdax_training_files()
        coin_and_hist_index = 0
        for x in range(0,len(fileNames)):
            df = self.get_file_as_frame(fileNames[x])
            col_prefix = self.get_file_symbol(fileNames[x])
            #df.drop("Unnamed: 0", 1)
            #df = self.read_in_moon_data(df)
            #df = df[::-1]
            df = df[::-1]
            self.currentHists[col_prefix] = df
            df = df.drop('Symbol', 1)
            df = df.drop("Date", 1)
            df = (df - df.mean()) / (df.max() - df.min())
            #df.rename(columns = lambda x: col_prefix+'_'+x, inplace=True)
            as_array = np.array(df)
            #print(len(as_array))
            self.hist_shaped[coin_and_hist_index] = as_array
            self.coin_dict[coin_and_hist_index] = col_prefix
            coin_and_hist_index += 1
        self.hist_shaped = pd.Series(self.hist_shaped)
        '''
        main = df_list[0]
        for i in range(1, len(df_list)):
            main = main.join(df_list[i])
        return main
        '''
hs = HistWorker()
sym = hs.pull_binance_symbols()
hs.write_binance_training_files(sym)
