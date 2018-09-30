import datetime, time
import pandas
import numpy as np
import tensorflow as tf

class CryptoIndividual:

    def __init__(self, substrate, genotype, phenotype):
        self.sub = substrate
        self.geno = genotype
        self.pheno = phenotype

class CryptoFolio:
    buys = {}
    sells = {}
    ledger = {}

    def __init__(self, start_amount)
        ledger['BTC'] = start_amount

    def buy_coin(self, c_name, amount):
        if(amount > self.ledger['BTC']):
            return
        else:
            ledger['BTC']
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

        def learn(self, data):
            for i in range(0, len(data)):

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
        sefl.lastGen = gens
    def read_hist(coin):
        try:
            df = pandas.DataFrame.read_csv(coin+_'_hist.txt')
            self.market[coin] = df
            return
        except:
            print("no history file found")
            return
    
    def read_all_hists(self):
        for c in self.coins:
            read_hist(c)
            return
    def feedNet(self, nextGens):
        for ix in range(0, len(nextGens)):

        

