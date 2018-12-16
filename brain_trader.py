# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 08:07:30 2017

@author: nick
"""
import pickle
import time
import pandas as pd
import numpy as np
from poloniex import Poloniex
from datetime import date, timedelta, datetime 
from hist_service import HistWorker
from crypto_evolution import CryptoFolio
from random import randint, shuffle
import requests
from pytorch_neat.cppn import create_cppn
# Local
import neat.nn
import _pickle as pickle
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat_torch import ESNetwork
#polo = Poloniex('key', 'secret')

key = ""
secret = ""
class LiveTrader:
    params = {"initial_depth": 3,
            "max_depth": 6,
            "variance_threshold": 0.013,
            "band_threshold": 0.013,
            "iteration_level": 3,
            "division_threshold": 0.013,
            "max_weight": 5.0,
            "activation": "tanh"}


    # Config for CPPN.
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                'config_trader')
    def __init__(self, ticker_len, target_percent):
        self.polo = Poloniex(key, secret)
        self.target_percent = target_percent
        self.currentHists = {}
        self.hist_shaped = {}
        self.coin_dict = {}
        self.ticker_len = ticker_len
        self.end_ts = datetime.now()+timedelta(seconds=(ticker_len*55))
        file = open("es_trade_god_cppn_5day.pkl",'rb')
        self.cppn = pickle.load(file)
        file.close()
        self.tickers = self.polo.returnTicker()
        self.sellCoins()
        self.bal = self.polo.returnBalances()
        self.set_target()
        self.pull_polo()
        self.inputs = self.hist_shaped.shape[0]*(self.hist_shaped[0].shape[1]-1)
        self.outputs = self.hist_shaped.shape[0]
        self.multiplier = self.inputs/self.outputs

    def make_shapes(self):
        self.in_shapes = []
        self.out_shapes = []
        sign = 1
        for ix in range(self.outputs):
            sign = sign *-1
            self.out_shapes.append((sign*ix, 1))
            for ix2 in range(len(self.hist_shaped[0][0])-1):
                self.in_shapes.append((sign*ix, (1+ix2)*.1))
        
    def pull_polo(self):
        tickLen = '7200'
        start = datetime.today() - timedelta(1) 
        start = str(int(start.timestamp()))
        ix = 0
        for coin in self.tickers:
            if coin[:3] == 'BTC':
                try:
                    hist = requests.get('https://poloniex.com/public?command=returnChartData&currencyPair='+coin+'&start='+start+'&end=9999999999&period='+tickLen)
                except:
                    self.pull_polo()
                try:
                    df = pd.DataFrame(hist.json())
                    #df.rename(columns = lambda x: col_prefix+'_'+x, inplace=True)
                    as_array = np.array(df)
                    #print(len(as_array))
                    self.currentHists[coin] = df
                    self.hist_shaped[ix] = as_array
                    self.coin_dict[ix] = coin
                    ix += 1
                except:
                    print("error reading json")
        self.hist_shaped = pd.Series(self.hist_shaped)
        self.end_idx = len(self.hist_shaped[0])-1


    def get_one_bar_input_2d(self):
        active = []
        misses = 0
        for x in range(0, self.outputs):
            try:
                sym_data = self.hist_shaped[x][self.end_idx] 
                for i in range(len(sym_data)):
                    if (i != 1):
                        active.append(sym_data[i].tolist())
            except:
                self.outputs -= 1
                self.inputs -= self.multiplier
                print('error')
        #print(active)
        self.make_shapes()
        return active


    def closeOrders(self):
        orders = self.polo.returnOpenOrders()
        for o in orders:
            if orders[o] != []:
                try:
                    ordnum = orders[o][0]['orderNumber']
                    self.polo.cancelOrder(ordnum)
                except:
                    print('error closing')
                    
                    
                    
    def sellCoins(self):
        for b in self.tickers:
            if(b[:3] == "BTC"):
                try:
                    price = self.get_price(b)
                    price = price - (price * .005)
                    self.sell_coin(b, price)
                except:
                    print("error getting price: ", b)

    def buy_coin(self, coin, price):
        amt = self.target / price
        if(self.bal['BTC'] > self.target):
            self.polo.buy(coin, price, amt, fillOrKill=1)
            print("buying: ", coin)
        return 

    def sell_coin(self, coin, price):
        amt = self.bal[coin[4:]]
        self.polo.sell(coin, price, amt, fillOrKill=1)
        print("selling this shit: ", coin)
        return 

    def reset_tickers(self):
        self.tickers = self.polo.returnTicker()
        return 
    def get_price(self, coin):
        return self.tickers[coin]['last']
    
    def set_target(self):
        total = 0
        full_bal = self.polo.returnCompleteBalances()
        for x in full_bal:
            total += full_bal[x]["btcValue"]
        self.target = total*self.target_percent

    def poloTrader(self):
        end_prices = {}
        active = self.get_one_bar_input_2d()
        sub = Substrate(self.in_shapes, self.out_shapes)
        network = ESNetwork(sub, self.cppn, self.params)
        net = network.create_phenotype_network()
        net.reset()
        for n in range(network.activations):
            out = net.activate(active)
        #print(len(out))
        rng = len(out)
        #rng = iter(shuffle(rng))
        self.reset_tickers()
        for x in np.random.permutation(rng):
            sym = self.coin_dict[x]
            #print(out[x])
            try:
                if(out[x] < -.5):
                    print("selling: ", sym)
                    p = self.get_price(sym)
                    price = p -(p*.01)
                    self.sell_coin(sym, price)
                elif(out[x] > .5):
                    print("buying: ", sym)
                    p = self.get_price(sym)
                    price = p*1.01
                    self.buy_coin(sym, price)
            except:
                print('error', sym)
            #skip the hold case because we just dont buy or sell hehe
            end_prices[sym] = self.get_price(sym)
        
        if datetime.now() >= self.end_ts:
            return
        else:
            time.sleep(self.ticker_len)
        self.pull_polo()
        self.poloTrader()

class PaperTrader:
    params = {"initial_depth": 2,
            "max_depth": 4,
            "variance_threshold": 0.0000013,
            "band_threshold": 0.0000013,
            "iteration_level": 3,
            "division_threshold": 0.0000013,
            "max_weight": 5.0,
            "activation": "tanh"}

    # Config for CPPN.
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                'config_trader')
    def __init__(self, ticker_len, start_amount, histdepth):
        self.polo = Poloniex()
        self.currentHists = {}
        self.hist_shaped = {}
        self.coin_dict = {}
        self.hist_depth = histdepth
        self.ticker_len = ticker_len
        self.end_ts = datetime.now()+timedelta(seconds=(ticker_len*24))
        self.start_amount = start_amount
        self.pull_polo()
        self.inputs = self.hist_shaped.shape[0]*(self.hist_shaped[0].shape[1])
        self.outputs = self.hist_shaped.shape[0]
        self.make_shapes()
        self.folio = CryptoFolio(start_amount, self.coin_dict)
        self.leaf_names = []
        for l in range(len(self.in_shapes[0])):
            self.leaf_names.append('leaf_one_'+str(l))
            self.leaf_names.append('leaf_two_'+str(l))
        self.load_net()

    def load_net(self):
        file = open("perpetual_champion.pkl",'rb')
        g = pickle.load(file)
        file.close()
        [the_cppn] = create_cppn(g, self.config, self.leaf_names, ['cppn_out'])
        self.cppn = the_cppn

    def make_shapes(self):
        self.in_shapes = []
        self.out_shapes = []
        sign = 1
        for ix in range(1,self.outputs+1):
            sign = sign *-1
            self.out_shapes.append((0.0-(sign*.005*ix), -1.0, -1.0))
            for ix2 in range(1,(self.inputs//self.outputs)+1):
                self.in_shapes.append((0.0+(sign*.01*ix2), 0.0-(sign*.01*ix2), 1.0))
        
    def pull_polo(self):
        try:
            self.coins = self.polo.returnTicker()
        except:
            time.sleep(10)
            self.pull_polo()
        tickLen = '7200'
        start = datetime.today() - timedelta(7) 
        start = str(int(start.timestamp()))
        ix = 0
        for coin in self.coins:
            if coin[:3] == 'BTC':
                hist = requests.get('https://poloniex.com/public?command=returnChartData&currencyPair='+coin+'&start='+start+'&end=9999999999&period='+tickLen)
                try:
                    df = pd.DataFrame(hist.json())

                    #df.rename(columns = lambda x: col_prefix+'_'+x, inplace=True)
                    as_array = np.array(df)
                    #print(len(as_array))
                    self.currentHists[coin] = df
                    self.hist_shaped[ix] = as_array
                    self.coin_dict[ix] = coin
                    ix += 1
                except:
                    print("error reading json")
        self.hist_shaped = pd.Series(self.hist_shaped)
        self.end_idx = len(self.currentHists[self.coin_dict[0]]) - 34


    def get_current_balance(self):
        self.pull_polo()
        c_prices = {}
        for s in self.folio.ledger.keys():
            if s != 'BTC':
                c_prices[s] = self.currentHists[s]['close'][len(self.currentHists[s]['close'])-1]
        return self.folio.get_total_btc_value_no_sell(c_prices)
        
    def get_one_bar_input_2d(self,end_idx=10):
        master_active = []
        for x in range(0, self.hist_depth):
            active = []
            #print(self.outputs)
            for y in range(0, self.outputs):
                sym_data = self.hist_shaped[y][self.hist_depth-x]
                #print(len(sym_data))
                active += sym_data.tolist()
            master_active.append(active)
        #print(active)
        return master_active
        
    def poloTrader(self):
        end_prices = {}
        active = self.get_one_bar_input_2d()
        self.load_net()
        sub = Substrate(self.in_shapes, self.out_shapes)
        network = ESNetwork(sub, self.cppn, self.params)
        net = network.create_phenotype_network_nd('paper_net.png')
        net.reset()
        for n in range(1, self.hist_depth+1):
            out = net.activate(active[self.hist_depth-n])
        #print(len(out))
        rng = len(out)
        #rng = iter(shuffle(rng))
        for x in np.random.permutation(rng):
            sym = self.coin_dict[x]
            #print(out[x])
            try:
                if(out[x] < -.5):
                    print("selling: ", sym)
                    self.folio.sell_coin(sym, self.currentHists[sym]['close'][self.end_idx])
                elif(out[x] > .5):
                    print("buying: ", sym)
                    self.folio.buy_coin(sym, self.currentHists[sym]['close'][self.end_idx])
            except:
                print('error', sym)
            #skip the hold case because we just dont buy or sell hehe
            end_prices[sym] = self.hist_shaped[x][len(self.hist_shaped[x])-1][2]
        
        if datetime.now() >= self.end_ts:
            port_info = self.folio.get_total_btc_value(end_prices)
            print("total val: ", port_info[0], "btc balance: ", port_info[1])
            return
        else:
            print(self.get_current_balance())
            for t in range(3):
                time.sleep(self.ticker_len/4)
                p_vals = self.get_current_balance()
                print("current value: ", p_vals[0], "current btc holdings: ", p_vals[1])
                #print(self.folio.ledger)
        time.sleep(self.ticker_len/4)
        self.pull_polo()
        self.poloTrader()
                        


live = PaperTrader(7200, 1.0, 10)
live.poloTrader()

