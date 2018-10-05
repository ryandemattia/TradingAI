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
    
    
    fees = {
        'sell': .01,
        'buy': .01
    }
    
    
    ledger = {}
    start = 0
    def __init__(self, start_amount):
        self.ledger['BTC'] = start_amount
        self.start = start_amount
        self.hs = hs.HistWorker()
    def buy_coin(self, c_name, amount, price, fee):
        if(amount*price > self.ledger['BTC']):
            return
        else:
            self.ledger['BTC'] -= (amount * price) + fee
            self.ledger[c_name] += amount
            return
    def sell_coin(self, price, amount, c_name):
        self.ledger[c_name] -= amount
        self.ledger['BTC'] += (amount * price) - ((amount * price)*self.fees['sell'])
    
    def get_total_btc_value(self, date):
        
        for c in self.ledger:
            if self.ledger[c] != 0:
                current_price = self.hs.currentHists[c][date]['Close']
                self.ledger['BTC'] += self.ledger[c] * current_price
                self.ledger[c] = 0
        return self.ledger['BTC']

    def evaluate_output(self, out, ):
        if (out == 1):
            self.buy_coin

class CryptoEval:

    def __init__(self, start_btc, population):
        self.port = CryptoFolio(start_btc)
        self.start_amnt = start_btc
        self.pop = population

    def evaluate(self, date):
        self.end_amnt = self.port.get_total_btc_value(date)
        perf = self.start_amnt - self.end_amnt
        return perf



class CryptoIndividual:

    def __init__(self, substrate, genotype, phenotype):
        self.sub = substrate
        self.geno = genotype
        self.pheno = phenotype

class EvoLayer:

    def __init__(self):
        return

class EvoGene:

    def __init__(self):
        return

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
            