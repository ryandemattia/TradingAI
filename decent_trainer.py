
### IMPORTS ###
import random
import sys, os
from functools import partial
from itertools import product

# Libs
import numpy as np
from hist_service import HistWorker
from crypto_evolution import CryptoFolio
from random import randint, shuffle
# Local
import neat.nn
import _pickle as pickle
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork
# Local
class DTrader(pkl="local_champ"):
    
    #needs to be initialized so as to allow for 62 outputs that return a coordinate

    # ES-HyperNEAT specific parameters.
    params = {"initial_depth": 3, 
            "max_depth": 4, 
            "variance_threshold": 0.03, 
            "band_threshold": 0.03, 
            "iteration_level": 5,
            "division_threshold": 0.003, 
            "max_weight": 5.0, 
            "activation": "tanh"}


    # Config for CPPN.
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                'config_trader')
                                
    start_idx = 0
    highest_returns = 0
    portfolio_list = []


    in_shapes = []
    out_shapes = []
    def __init__(self, hist_depth):
        self.hs = HistWorker()
        self.hd = hist_depth
        self.end_idx = len(self.hs.currentHists["XCP"])
        self.but_target = .1
        self.inputs = self.hs.hist_shaped.shape[0]*(self.hs.hist_shaped[0].shape[1]-1)
        self.outputs = self.hs.hist_shaped.shape[0]
        sign = 1
        for ix in range(1,self.outputs+1):
            sign = sign *-1
            self.out_shapes.append((sign/ix, .0, 1.0*sign))
            for ix2 in range(1,len(self.hs.hist_shaped[0][0])):
                self.in_shapes.append((-sign/ix, (sign/ix2), 1.0*sign))
        self.subStrate = Substrate(self.in_shapes, self.out_shapes)
        self.epoch_len = 360
        
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
        #print(active)
        return active

    def evaluate(self, network, es, rand_start, verbose=False):
        portfolio = CryptoFolio(.05, self.hs.coin_dict)
        end_prices = {}
        buys = 0
        sells = 0 
        for z in range(rand_start, rand_start+self.epoch_len):
            '''
            if(z == 0):
                old_idx = 1
            else:
                old_idx = z * 5
            new_idx = (z + 1) * 5
            '''
            active = self.get_one_bar_input_2d(z)
            network.reset()
            for n in range(es.activations):
                out = network.activate(active)
            #print(len(out))
            rng = len(out)
            #rng = iter(shuffle(rng))
            for x in np.random.permutation(rng):
                sym = self.hs.coin_dict[x]
                #print(out[x])
                try:
                    if(out[x] < -.5):
                        #print("selling")
                        portfolio.sell_coin(sym, self.hs.currentHists[sym]['close'][z])
                    elif(out[x] > .5):
                        #print("buying")
                        portfolio.buy_coin(sym, self.hs.currentHists[sym]['close'][z])
                except:
                    print('error', sym)
                #skip the hold case because we just dont buy or sell hehe
                end_prices[sym] = self.hs.hist_shaped[x][self.epoch_len][2]
        result_val = portfolio.get_total_btc_value(end_prices)
        print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2])
        return result_val[0]

    def solve(self, network):
        return self.evaluate(network) >= self.highest_returns
        

    def eval_fitness(self, genomes, config):
        r_start = randint(0, self.hs.hist_full_size - self.epoch_len)    
        for idx, g in genomes:

            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            network = ESNetwork(self.subStrate, cppn, self.params)
            net = network.create_phenotype_network_nd()
            g.fitness = self.evaluate(net, network, r_start)

    def eval_fitness_single(self, genome, config):
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = ESNetwork(self.subStrate, cppn, self.params)
        net = network.create_phenotype_network_nd()
        g.fitness = self.evaluate(net, network, r_start)
        return g.fitness