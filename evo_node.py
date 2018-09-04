import datetime, time
import pandas
import numpy as np
import tensorflow as tf
class EvoNet:
    gl = {}
    buyPower = 0
    sells = {}
    buys = {}
    fitnessScore = 0
    lastScore = 0
    numNodes = 0
    def __init__(self, parent1, parent2, startAmt):
        buyPower = startAmt
        numNodes = (parent1.numNodes + parent2.numNodes)/2
    
    

    def but_alt(self, coin, amount, price):
        btc_amnt = (amount * price)
        if (self.buyPower > btc_amnt):
            self.buyPower = self.buyPower - btc_amnt
            self.gl[coin]['amount'] += amount
            #avg entry price motherfucker
            self.gl[coin]['price'] = (self.gl[coin]['price'] + price)/2
        else:
            print("not enough btc silly")

    def sell_alt(self, )
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

        

