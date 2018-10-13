import pickle
import time
import pandas as pd
import requests
import numpy as np
from poloniex import Poloniex
from datetime import date, timedelta, datetime 



def pull_polo():
    polo = Poloniex()
    coins = polo.returnTicker()
    tickLen = '7200'
    start = datetime.today() - timedelta(30) 
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

pull_polo()