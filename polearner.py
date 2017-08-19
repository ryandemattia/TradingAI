# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 08:07:30 2017

@author: nick
"""
import pickle
import time
import pandas as pd
import urllib2
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression, Ridge, TheilSenRegressor
import numpy as np
from poloniex import Poloniex
from datetime import date, timedelta, datetime 







def getscore_getnext(df, days_ahead, coin):

    forecast_val = days_ahead

    forecast_col = 'close'
    df.fillna(value=-99999, inplace=True)
    df['label'] = df[forecast_col].shift(-forecast_val)

    X = np.array(df.drop(['label', 'date'], 1))
        
    X = preprocessing.scale(X)

    futureX = X[-1:]
    X = X[:-forecast_val]
    df.dropna(inplace=True)
            
    y = np.array(df['label'])
        
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.15)
    '''
    inPickle = open('%s.pickle' %(coin), 'rb')
    clf = pickle.load(inPickle)
    '''
    clf = LinearRegression(n_jobs=2)
            
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    #print "accuracy with 1.0 being perfect:", (confidence)
    futureval = clf.predict(futureX)
    print '%s model ready' %(coin)
    return (confidence, futureval)
    
def makeModels(df, days_ahead, coin):


    forecast_val = days_ahead

    forecast_col = 'close'
    df.fillna(value=-99999, inplace=True)
    df['label'] = df[forecast_col].shift(-forecast_val)


    X = np.array(df.drop(['label', 'date'], 1))
        
    X = preprocessing.scale(X)

    futureX = X[-1:]
    X = X[:-forecast_val]
    df.dropna(inplace=True)
            
    y = np.array(df['label'])
        
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.15)
        
    clf = Ridge()
            
    clf.fit(X_train, y_train)
    with open('%s.pickle' %(coin), 'wb') as f:
        pickle.dump(clf,f)

polo = Poloniex('key', 'secret')

def runit(tickerLength, forcast):
    coins = polo.returnTicker()
    stop = datetime.now()+timedelta(seconds=(tickerLength*48))
    tickerStr = str(tickerLength)
    # uncomment the next lines to creat pickles for the ml models, then (in getscore_getnext comment out the clf definition and training, and uncomment the pickle loading 
    '''
    for coin in coins:
        if coin[:3] == 'BTC':
            hist = urllib2.urlopen('https://poloniex.com/public?command=returnChartData&currencyPair='+coin+'&start=1405699200&end=9999999999&period='+tickerStr)
            frame = pd.read_json(hist)
            makeModels(frame, forcast, coin)
            '''
    poloTrader(stop, tickerLength, forcast, coins)
def write_predictions(forcastVal, interval, coins):
    predictions = {}
    interval = str(interval)
    with open('polopredict.txt', 'w') as f:
        f.write('coin,lastPrice, nextPrice, accuraccy, returns\n')
        for coin in coins:
            if coin[:3] == 'BTC':
                hist = urllib2.urlopen('https://poloniex.com/public?command=returnChartData&currencyPair='+coin+'&start=1405699200&end=9999999999&period='+interval)
                try:
                    frame = pd.read_json(hist)
                except:
                    print "error"
                last = coins[coin]['last']
                predict = getscore_getnext(frame, forcastVal, coin)
                future = predict[1][0]
                acc = predict[0]
                returns = ((future-last)/last)*100
                #print coin, last, future, acc, returns
                f.write('%s, %f, %f, %f, %f\n' %(coin, last, future, acc, returns))
                predictions[coin] = {'returns': returns,
                                    'accuracy': acc}
    return predictions
            
def closeOrders():
    orders = polo.returnOpenOrders()
    for o in orders:
        if orders[o] != []:
            try:
                ordnum = orders[o][0]['orderNumber']
                polo.cancelOrder(ordnum)
            except:
                print 'error closing'
                
                
                
def sellCoins(coinlist, currency):
    balances = polo.returnBalances()
    for b in balances:
        bal = balances[b]*.99
        pair = 'BTC_'+b
        if (bal != 0.0) and (currency[b]['delisted'] != 1 and currency[b]['disabled'] != 1):
            if pair in coinlist:
                try:
                    tick = polo.returnTicker()
                    rt = tick[pair]['highestBid']*0.999
                    print 'selling: %s' %(pair)
                    polo.sell(pair, rt, bal, postOnly=0)
                except:
                    print 'error while selling: %s' %(pair)
                

def poloTrader(end, intervalLength, lookoutVal, coins):
    closeOrders()
    curr = polo.returnCurrencies()
    #sellList = []
    sellCoins(coins, curr)
    predict = write_predictions(lookoutVal, intervalLength, coins)
    sort = sorted(predict.keys(), key=lambda x: predict[x]['returns'], reverse=True)
    print sort
    currentBal = polo.returnBalances()
    amt = (currentBal['BTC'])/3
    #time.sleep(180)
    closeOrders()
    for i in range(0,len(sort)):
        tick = polo.returnTicker()
        c = sort[i]
        print c
        if predict[c]['accuracy'] > .95 and predict[c]['returns'] > 0.0:
            try:   
                rt = tick[c]['lowestAsk']*1.0015
                print 'buying: %s, returns: %f' %(c, predict[c]['returns'])
                polo.buy(c, rt, (amt/rt), postOnly=0)
            except:
                print 'error'
    if datetime.now() >= end:
        return
    else:
        print 'buys complete'
        time.sleep(intervalLength*4)
        newBal = polo.returnBalances()
        for n in newBal:
            if newBal[n] > 0.0:
                print n, newBal[n]
    poloTrader(end, intervalLength, lookoutVal, coins)
                        
runit(1800, 3)

