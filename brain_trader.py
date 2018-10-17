# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 08:07:30 2017

@author: nick
"""
import pickle
import time
import pandas as pd
import urllib2
import numpy as np
from poloniex import Poloniex
from datetime import date, timedelta, datetime 

polo = Poloniex('key', 'secret')


class PaperTrader:
    
    def __init__(self, stop_ts, ticker_len, start_amount):
        self.ticker_len = ticker_len
        self.start_amount = start_amount
        
    

    def runit(self, tickerLength, epochs):
        coins = polo.returnTicker()
        self.stop = datetime.now()+timedelta(seconds=(self.ticker_len*epochs))
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

                
    def closeOrders(self):
        orders = polo.returnOpenOrders()
        for o in orders:
            if orders[o] != []:
                try:
                    ordnum = orders[o][0]['orderNumber']
                    polo.cancelOrder(ordnum)
                except:
                    print 'error closing'
                    
                    
    def sellCoins(self, currency):
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
                    
    
    def poloTrader(self, end, intervalLength, lookoutVal, coins):
        closeOrders()
        curr = polo.returnCurrencies()
        #sellList = []
        self.sellCoins(coins, curr)
        currentBal = polo.returnBalances()
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
            time.sleep(intervalLength)
            newBal = polo.returnBalances()
            for n in newBal:
                if newBal[n] > 0.0:
                    print n, newBal[n]
        poloTrader(end, intervalLength, lookoutVal, coins)
                        


bt = PaperTrader()
bt.runit(7200, 4)

