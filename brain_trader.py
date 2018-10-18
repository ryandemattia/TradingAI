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
# Local
import neat.nn
import _pickle as pickle
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork
#polo = Poloniex('key', 'secret')


class PaperTrader:
    params = {"initial_depth": 0, 
            "max_depth": 4, 
            "variance_threshold": 0.03, 
            "band_threshold": 0.3, 
            "iteration_level": 1,
            "division_threshold": 0.3, 
            "max_weight": 5.0, 
            "activation": "tanh"}


    # Config for CPPN.
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                'config_trader')
    def __init__(self, ticker_len, start_amount):
        self.polo = Poloniex()
        self.currentHists = {}
        self.hist_shaped = {}
        self.coin_dict = {}
        self.ticker_len = ticker_len
        self.end_ts = datetime.now()+timedelta(seconds=(ticker_len*24))
        self.start_amount = start_amount
        file = open("es_trade_god_cppn.pkl",'rb')
        self.cppn = pickle.load(file)
        file.close()
        self.pull_polo()
        self.inputs = self.hist_shaped.shape[0]*(self.hist_shaped[0].shape[1]-1)
        self.outputs = self.hist_shaped.shape[0]
        self.multiplier = self.inputs/self.outputs
        self.folio = CryptoFolio(start_amount, self.coin_dict)
        
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
        self.coins = self.polo.returnTicker()
        tickLen = '7200'
        start = datetime.today() - timedelta(1) 
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
            print(self.folio.get_total_btc_value(end_prices))
            return
        else:
            print(self.folio.get_total_btc_value_no_sell(end_prices))
            time.sleep(self.ticker_len)
        self.pull_polo()
        self.poloTrader()
                        

pt = PaperTrader(7200, .5)
pt.poloTrader()
