import pickle
import time
import pandas as pd
import urllib
import numpy as np
from poloniex import Poloniex
from datetime import date, timedelta, datetime 



def pull_polo():
    polo = Poloniex()
    coins = polo.returnTicker()
    tickLen = '3600'
    start = datetime.today() - timedelta(365) 
    start = str(time.mktime(start.timetuple()))
    with open('polopredict.txt', 'w') as f:
        f.write('date,coin,lastPrice,nextPrice,accuraccy,returns\n')
        for coin in coins:
            if coin[:3] == 'BTC':
                hist = urllib.request('https://poloniex.com/public?command=returnChartData&currencyPair='+coin+'&start='+start+'&end=9999999999&period='+tickLen)
                try:
                    frame = pd.read_json(hist)
                    print(frame)
                except:
                    print("error")

pull_polo()