
### IMPORTS ###
import random
import sys, os
from functools import partial
from itertools import product

# Libs
import numpy as np
# Libraries
import numpy as np
from hist_service import HistWorker
from crypto_evolution import CryptoFolio
# Local
from peas.peas.methods.hyperneat import HyperNEATDeveloper, Substrate
import peas.peas.networks.rnn
from peas.peas.methods.neat import NEATPopulation, NEATGenotype
from peas.peas.methods.evolution import SimplePopulation

'''
will come back to this, trying to get something up and running
tonight so gonna make a quick n dirty purples experiment
''' 
#SQUAD
class TradingTask:

    #EPSILON = 1e-100

    start_idx = 0
    highest_returns = 0
    portfolio_list = []

    def __init__(self):
        self.hs = HistWorker()
        self.end_idx = len(self.hs.currentHists["DASH"])
        self.but_target = .1
        self.inputs = self.hs.hist_shaped.shape[0]*self.hs.hist_shaped[0].shape[1]
        self.outputs = self.hs.hist_shaped.shape[0] # times by three for buy | sell | hodl(pass)
        #self.sub = Substrate(self.inputs, self.outputs)
        #self.port = CryptoFolio(1)

    
    def set_portfolio_keys(self, folio):
        for k in self.hs.currentHists.keys():
            folio.ledger[k] = 0

    def get_one_bar_input_2d(self, end_idx):
        active = []
        for x in range(0, self.outputs):
            try:
                sym_data = self.hs.hist_shaped[x][end_idx] 
                for i in range(len(sym_data)):
                    if (i != 1):
                        active.append(sym_data[i].tolist())
            except:
                print('error')
        return active

    def evaluate(self, network, verbose=False):
        portfolio = CryptoFolio(1, self.hs.coin_dict)
        active = []
        results = {}
        end_prices = {}
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        rand_start = randint(0, self.hs.hist_full_size - 89) #get random start point with a week of padding from end
        for z in range(rand_start, rand_start+89):
            '''
            if(z == 0):
                old_idx = 1
            else:
                old_idx = z * 5
            new_idx = (z + 1) * 5
            '''
            active = self.get_one_bar_input_2d(z)
            results[z] = network.feed(active)

        for i in range(rand_start, rand_start+89):
            out = results[i]
            for x in range(0, self.outputs):
                sym = self.hs.coin_dict[x]
                if(out[x] == 1.0):
                    portfolio.buy_coin(sym, self.hs.currentHists[sym][x]['close'])
                elif(out[x] == 0.0):
                    portfolio.sell_coin(sym, self.hs.currentHists[sym][x]['close'])
        for y in range(len(out)):
            end_prices[self.hs.coin_dict[y]] = self.hs.hist_shaped[y][89][2]
        result_val = portfolio.get_total_btc_value(end_prices)
        print(results)
        if(results > self.highest_returns):
            self.highest_returns = results
        return results

    def solve(self, network):
        return self.evaluate(network) >= 5

    def run(self, generations=100, popsize=100):
                
        substrate = Substrate()
        substrate.add_nodes((self.hs.hist_shaped.shape[0],), 'l')
        substrate.add_connections('l', 'l')
        geno = lambda: NEATGenotype(feedforward=True, inputs=self.inputs, weight_range=(-3.0, 3.0), 
                                       prob_add_conn=0.3, prob_add_node=0.03,
                                       types=['sin', 'ident', 'gauss', 'sigmoid', 'abs'])
        pop = NEATPopulation(geno, popsize=popsize, target_species=8)
        developer = HyperNEATDeveloper(substrate=substrate, add_deltas=False, sandwich=False, node_type="sigmoid")
        results = pop.epoch(generations=generations, evaluator=partial(self.evaluate, task=self, developer=developer), solution=partial(self.solve, task=self, developer=developer))
        return results



if __name__ == '__main__':
    do_it = TradingTask().run()

