import hist_service as hs
import datetime, time
import pandas as pd
import numpy as np



class Cryptolution:

    def __init__(self, generations): #pass number of generations
        self.histworker = hs.HistWorker()
        self.num_gens = generations
    
    def one_gen(self):
        for x in range(len(self.histworker.currentHists[0])):
            for symbol in self.histworker.currentHists.keys():
                dataIn = self.histworker.currentHists[symbol]
                substrate_in = len(dataIn.columns.values)


    def data_loop(self):
        for i in self.num_gens:
            return i


class CryptoFolio:
    
    #assume we 
    fees = .002
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
            coin_amount = amount/price
            the_fee = self.fees * amount
            self.ledger['BTC'] -= (amount + self.fees)
            self.ledger[c_name] += coin_amount
            return


    def sell_coin(self, c_name, price):
        if self.ledger[c_name] > 0:
            amount = self.ledger[c_name]
            self.ledger['BTC'] += ((amount*price) - ((amount * price)*self.fees))
            self.ledger[c_name] = 0
            return
        else:
            return

    
    def get_total_btc_value(self, e_prices):
        
        for c in self.ledger:
            if self.ledger[c] != 0 and c != "BTC":
                current_price = e_prices[c]
                val = (self.ledger[c] * current_price)
                val = val - val*self.fees
                self.ledger['BTC'] = self.ledger['BTC']+val
                self.ledger[c] = 0
        return self.ledger['BTC']

    def evaluate_output(self, out, coin, price):
        if (out == 1.0):
            self.buy_coin(coin, price)
        elif(out==.5):
            return
        else:
            self.sell_coin(price,coin)

class CryptoEval:

    def __init__(self, start_btc, population):
        self.port = CryptoFolio(start_btc)
        self.start_amnt = start_btc
        self.pop = population

    def evaluate(self, date):
        self.end_amnt = self.port.get_total_btc_value(date)
        perf = self.start_amnt - self.end_amnt
        return perf

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
            


class EvoNode:
    fitnessScore = 0
    lastScore = 0
    numInNodes = 0
    def __init__(self, parent1, parent2, startAmt, numNodes):
        buyPower = startAmt
        numInNodes = inNodes
        numOutNodes = numOutNodes
        generation = 0 
        fitness = 0
        weights = {}

        #def learn(self, data):
            #for i in range(0, len(data)):