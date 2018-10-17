import hist_service as hs
import datetime, time
import pandas as pd
import numpy as np


class CryptoFolio:
    
    #assume we 
    fees = .002
    buys = 0
    sells = 0
    target_amount = 0.1
    ledger = {}
    start = 0
    def __init__(self, start_amount, coins):
        self.ledger['BTC'] = start_amount
        for ix in coins:
            self.ledger[coins[ix]] = 0
        self.start = start_amount
        self.hs = hs.HistWorker()


    def buy_coin(self, c_name, price):
        amount = self.start * self.target_amount
        if(amount > self.ledger['BTC']):
            return
        else:
            coin_amount = amount/(price* 0.01)
            the_fee = self.fees * amount
            self.ledger['BTC'] -= (amount + the_fee)
            self.ledger[c_name] += coin_amount
            self.buys += 1
            return


    def sell_coin(self, c_name, price):
        price = price * .01
        if self.ledger[c_name] != 0:
            amount = self.ledger[c_name]
            self.ledger['BTC'] += ((amount*price) - ((amount * price)*self.fees))
            self.ledger[c_name] = 0
            self.sells += 1
            return
        else:
            return

    
    def get_total_btc_value(self, e_prices):
        
        for c in self.ledger.keys():
            if self.ledger[c] != 0 and c != "BTC":
                current_price = e_prices[c]
                self.sell_coin(c, current_price)
        return self.ledger['BTC'], self.buys, self.sells

    def evaluate_output(self, out, coin, price):
        if (out == 1.0):
            self.buy_coin(coin, price)
        elif(out==.5):
            return
        else:
            self.sell_coin(price,coin)


class EvoSim:
    count = 0
    starting_btc = 1000
    bestNets = []
    lastGen = []
    numNets = 0
    coins = []
    market = {}
    nextGens = []
    def __init__(self, numberOfNets, coins, gens):
        self.count += 1
        self.numNets = numberOfNets
        self.coins = coins
        self.lastGen = gens
        
    def read_hist(self, coin):
        try:
            df = pd.DataFrame.read_csv(coin+'_hist.txt')
            self.market[coin] = df
            return
        except:
            print("no history file found")
            return
    
    def read_all_hists(self):
        for c in self.coins:
            self.read_hist(c)
            return
        
        
    def feedNet(self, nextGens):
        for ix in range(0, len(nextGens)):
            print(ix)
            