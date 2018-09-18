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
#import json
from bittrex import bittrex
import numpy as np
from datetime import date, timedelta, datetime 

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
    
def three_months_back():
    today = datetime.today().strftime('%Y%m%d') 
    backDate = datetime.today() - timedelta(days=90)
    return (today, backDate.strftime('%Y%m%d') )

def getCryptoHist(coin):
    dates = three_months_back()
    print(dates)
    url = 'https://coinmarketcap.com/currencies/'+coin+'/historical-data/?start=%s&end=%s' % (dates[1], dates[0])
    print(url)
    resp = requests.get(url)
    soup = bs.BeautifulSoup(resp.text)
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
        date.insert(0,col0)
        col1= col[1].text.strip()
        openPrice.insert(0,float(col1))
        col2 = col[2].text.strip()
        high.insert(0,float(col2))
        col3 = col[3].text.strip()
        low.insert(0,float(col3))
        col4 = col[4].text.strip()
        close.insert(0,float(col4))
        col5 = col[5].text.strip()
        col5 = col5.replace(',', '')
        volume.insert(0,int(col5))
        col6 = col[6].text.strip()
        col6 = col6.replace(',', '')
        marketCap.insert(0, int(col6))
    tableDict = {'date': date, 'open': openPrice, 'high': high, 'low': low, 'close': close, 'volume': volume, 'marketCap': marketCap}
    return tableDict

print(getCryptoHist('monero'))