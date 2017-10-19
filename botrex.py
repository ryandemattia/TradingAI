#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:03:45 2017

@author: nickwilliams
"""

import bs4 as bs
import requests
import re
import pandas as pd
import time
import urllib2
#import json
from bittrex import bittrex
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression, Ridge, TheilSenRegressor
import numpy as np
from datetime import date, timedelta, datetime 







def getscore_getnext(df, days_ahead, coin):


    forecast_val = days_ahead

    forecast_col = 'close'
    df.fillna(value=-99999, inplace=True)
    df['label'] = df[forecast_col].shift(-forecast_val)




    #X = X[:-forecast_val]



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
    clf = TheilSenRegressor()
            
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    #print "accuracy with 1.0 being perfect:", (confidence)
    futureval = clf.predict(futureX)
    return (confidence, futureval)
    
def makeModels(df, days_ahead, coin):


    forecast_val = days_ahead

    forecast_col = 'close'
    df.fillna(value=-99999, inplace=True)
    df['label'] = df[forecast_col].shift(-forecast_val)




    #X = X[:-forecast_val]



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


def get_symbols():
    resp = requests.get('https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json?date=20160912&api_key=c3xgbur-Bga4sBAL8Hbu')
    soup = str(bs.BeautifulSoup(resp.text))
    ticks = re.findall("([A-Z]{2,})", soup)
    return ticks
    
def savesyms():
    with open('wikisyms.txt', 'w') as f:
        f.write('symbols\n')
        symbols = get_symbols()
        for i in range(0, len(symbols)):
            f.write('%s\n' %(symbols[i]))
def getBittrex():
    resp = requests.get('https://coinmarketcap.com/exchanges/bittrex/')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    sevenDay = soup.find("div", {"id":"markets" })
    tab = sevenDay.find('table', {"class": "table"})
    rows = tab.findAll('tr')[1:]
    rowNum = []
    name = []
    symbol = []
    volume = []
    price = []
    volP = []
    for row in rows:            #pack lists
        col = row.findAll('td')
        col0 = col[0].text.strip()
        rowNum.append(col0)
        col1= col[1].text.strip()
        name.append(col1)
        col2 = col[2].text.strip()
        symbol.append(col2)
        col3 = col[3].text.strip()
        volume.append(col3)
        col4 = col[4].text.strip()
        price.append(col4)
        col5 = col[5].text.strip()
        volP.append(col5)   
    tableDict = {'row': rowNum, 'name': name, 'symbol': symbol, 'volume': volume, 'price': price, 'volume%': volP}
    df = pd.DataFrame(tableDict)
    return df

def getCryptoGainers():
    resp = requests.get('https://coinmarketcap.com/gainers-losers/')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    sevenDay = soup.find("div", {"id":"losers-1h" })
    tab = sevenDay.find('table')
    body = tab.find('tbody')
    rows = body.findAll('tr')
    rowNum = []
    name = []
    symbol = []
    volume = []
    price = []
    gains = []
    for row in rows:            #pack lists
        col = row.findAll('td')
        col0 = col[0].text.strip()
        rowNum.append(col0)
        col1= col[1].text.strip()
        name.append(col1)
        col2 = col[2].text.strip()
        symbol.append(col2)
        col3 = col[3].text.strip()
        volume.append(col3)
        col4 = col[4].text.strip()
        price.append(col4)
        col5 = col[5].text.strip()
        gains.append(col5)
    tableDict = {'row': rowNum, 'name': name, 'symbol': symbol, 'volume': volume, 'price': price, 'gains': gains}
    df = pd.DataFrame(tableDict)
    return df
    
    
def getCryptoHist(coin):
    url = 'https://coinmarketcap.com/currencies/'+coin+'/historical-data/'
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, "lxml")
    sevenDay = soup.find("div", {"id":"historical-data" })
    tab = sevenDay.find('table')
    body = tab.find('tbody')
    rows = body.findAll('tr')
    date = []
    openPrice = []
    high = []
    low = []
    close = []
    volume = []
    marketCap = []
    for row in rows:            #pack lists
        col = row.findAll('td')
        col0 = col[0].text.strip()
        date.append(col0)
        col1= col[1].text.strip()
        openPrice.append(float(col1))
        col2 = col[2].text.strip()
        high.append(float(col2))
        col3 = col[3].text.strip()
        low.append(float(col3))
        col4 = col[4].text.strip()
        close.append(float(col4))
        col5 = col[5].text.strip()
        col5 = col5.replace(',', '')
        volume.append(int(col5))
        col6 = col[6].text.strip()
        col6 = col6.replace(',', '')
        marketCap.append(int(col6))
    tableDict = {'date': date, 'open': openPrice, 'high': high, 'low': volume, 'close': close, 'volume': volume, 'marketCap': marketCap}
    df = pd.DataFrame(tableDict)
    return df
def getCoinMarkets(coin):
    url = 'https://coinmarketcap.com/currencies/'+coin+'/#markets'
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text, "lxml")
    tab = soup.find('table', {"id": "markets-table"})
    rows = tab.findAll('tr')[1:]
    names = []
    price = []
    for row in rows:
        col= row.findAll('td')
        col0 = col[1].text.strip()
        names.append(col0)
    return names
def getBittrexCoins():
    resp = requests.get('https://coinmarketcap.com/g')
    

bigGuys = getCryptoGainers()

def checkGainersBittrex():
    coins = {}
    for c in bigGuys['name']:
        print c
        try:
            marks = getCoinMarkets(c)
            if 'Bittrex' in marks:
                try:
                    history = getCryptoHist(c)
                    last = history.close.iloc[-1]
                    preds = getscore_getnext(history, 1, c)
                    returns = ((last-preds[1])/preds[1])*100
                    print c+':', preds[0], returns
                except:
                    print 'error'
        except:
            print 'error getting markets'

def predictBittrex():
    bittrexCoins = getBittrex()
    scores = {}
    for coin in bittrexCoins.values:
        if coin[3][-4:] == '/BTC':
            c = coin[0]
            s = coin[3][:-4]
            try:
                history = getCryptoHist(c)
                last = history.close.iloc[-1]
                avg = history.close.mean()
                preds = getscore_getnext(history, 1, c)
                returns = ((last-preds[1][0])/preds[1][0])*100
                scores[s] = {'accuracy': preds[0], 'returns': returns}
                print scores[s]
            except:
                print 'error'
    return scores
        


def sellCoins(api):
    account = api.getbalances()
    for b in account:
        if b['Currency'] != 'BTC':
            market = 'BTC-'+b['Currency']
            rate = api.getticker(market)['Bid']
            if rate*b['Balance'] > 0.0005:
                try:
                    print api.selllimit(market, b['Balance'], rate)
                except:
                    print 'error selling', market 
                    
def buyCoins(preds, api):
    amount = api.getbalance('BTC')['Available']/10
    for c in preds:
        print c
        market = 'BTC-'+c
        rate = api.getticker(market)['Ask']
        coinAmnt = amount/rate
        try:
            print api.buylimit(market, coinAmnt, rate)
        except:
            print 'error buying', market
        time.sleep(2)
def bot():
    key = '' 
    secret = '' 
    bitApi = bittrex(key,secret)
    predictions = predictBittrex()
    sortedPre = sorted(predictions.keys(), key=lambda x: predictions[x]['returns'])
    sellCoins(bitApi)
    time.sleep(180)        
    buyCoins(sortedPre[:10], bitApi)
    time.sleep(7200*4)
    bot()

bot()
  