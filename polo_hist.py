import pickle
import time
import pandas as pd
import urllib2
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression, Ridge, TheilSenRegressor
import numpy as np
from poloniex import Poloniex
from datetime import date, timedelta, datetime 



def pull_polo():
    coins = polo.returnTicker()
    tickLen = '3600'
    start = 230
    with open('polopredict.txt', 'w') as f:
        f.write('date,coin,lastPrice,nextPrice,accuraccy,returns\n')
        for coin in coins:
            if coin[:3] == 'BTC':
                hist = urllib2.urlopen('https://poloniex.com/public?command=returnChartData&currencyPair='+coin+'&start='+start+'&end=9999999999&period='+interval)
                try:
                    frame = pd.read_json(hist)
                except:
                    print "error"
