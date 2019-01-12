### IMPORTS ###
import random
import sys, os
from functools import partial
from itertools import product
from pytorch_neat.cppn import create_cppn
import pandas as pd
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
from pureples.es_hyperneat.es_hyperneat_torch import ESNetwork
# Local
class PurpleTrader:

    #needs to be initialized so as to allow for 62 outputs that return a coordinate

    # ES-HyperNEAT specific parameters.
    params = {"initial_depth": 2,
            "max_depth": 3,
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

    start_idx = 0
    highest_returns = 0
    portfolio_list = []



    in_shapes = []
    out_shapes = []
    def __init__(self, hist_depth):
        self.hs = HistWorker()
        self.hs.build_vix_frame()
        self.hd = hist_depth
        print(self.hs.currentHists.keys())
        self.end_idx = len(self.hs.currentHists)
        self.but_target = .25
        self.inputs = self.hs.hist_shaped.shape[1]
        self.outputs = 1
        sign = 1
        for ix in range(1,self.inputs + 1):
            sign = sign *-1
            self.in_shapes.append((0.0-(sign*.005*ix), 0.0-(sign*.005*ix), 1.0))
        self.out_shapes.append((0.0, -1.0, -1.0))
        self.subStrate = Substrate(self.in_shapes, self.out_shapes)
        self.epoch_len = 144
        #self.node_names = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'weight']
        self.leaf_names = []
        #num_leafs = 2**(len(self.node_names)-1)//2
        for l in range(len(self.in_shapes[0])):
            self.leaf_names.append('leaf_one_'+str(l))
            self.leaf_names.append('leaf_two_'+str(l))
        #self.leaf_names.append('bias')
    def set_portfolio_keys(self, folio):
        for k in self.hs.currentHists.keys():
            folio.ledger[k] = 0

    def get_one_epoch_input(self,end_idx):
        master_active = []
        for x in range(0, self.hd):
            try:
                sym_data = self.hs.hist_shaped[end_idx-x]
                #print(len(sym_data))
                master_active.append(sym_data.tolist())
            except:
                print('error')
        #print(active)
        return master_active


    def get_single_symbol_epoch(self, end_idx, symbol_idx):
        master_active = []
        for x in range(0, self.hd):
            try:
                sym_data = self.hs.hist_shaped[symbol_idx][end_idx-x]
                #print(len(sym_data))
                master_active.append(sym_data.tolist())
            except:
                print('error')
        return master_active
    def load_net(self, fname):
        f = open(fname,'rb')
        g = pickle.load(f)
        f.close()
        [the_cppn] = create_cppn(g, self.config, self.leaf_names, ['cppn_out'])
        self.cppn = the_cppn

    def run_champs(self):
        genomes = os.listdir(os.path.join(os.path.dirname(__file__), 'champs'))
        fitness_data = {}
        best_fitness = 0.0
        for g_ix in range(len(genomes)):
            genome = self.load_net('./champs/'+genomes[g_ix])
            start = self.hs.hist_full_size - self.epoch_len
            network = ESNetwork(self.subStrate, self.cppn, self.params)
            net = network.create_phenotype_network_nd('./champs_vis/genome_'+str(g_ix))
            fitness = self.evaluate(net, network, start, g_ix, genomes[g_ix])
            if fitness > best_fitness:
                best_genome = genome

    def evaluate(self, network, es, rand_start, g, p_name):
        portfolio_start = 100000
        portfolio = CryptoFolio(portfolio_start, self.hs.coin_dict)
        portfolio.ledger['vix'] = 0.0
        end_prices = {}
        buys = 0
        sells = 0
        th = []
        with open('./champs_hist/trade_hist'+p_name + '.txt', 'w') as ft:
            ft.write('Date,symbol,type,amnt,price,current_balance \n')
            for z in range(self.hd, self.hs.hist_full_size -1):
                sym = 'vix'
                active = self.get_one_epoch_input(z)
                network.reset()
                for n in range(0, self.hd):
                    n += 1
                    out = network.activate(active[self.hd-n])
                end_prices[sym] = self.hs.currentHists['VIX Close'][self.hs.hist_full_size-1]
                if(out[0] < -.5):
                    #print("selling")
                    did_sell = portfolio.sell_coin(sym, self.hs.currentHists['VIX Close'][z])
                    if did_sell:
                        ft.write(str(self.hs.currentHists['Date'][z]) + ",")
                        ft.write(sym +",")
                        ft.write('sell,')
                        ft.write(str(portfolio.ledger[sym])+",")
                        ft.write(str(self.hs.currentHists['VIX Close'][z])+",")
                        ft.write(str(portfolio.get_total_btc_value_no_sell(end_prices)[0])+ " \n")
                    #print("bought ", sym)
                elif(out[0] > .5):
                    did_buy = portfolio.buy_coin(sym, self.hs.currentHists['VIX Close'][z])
                    if did_buy:
                        ft.write(str(self.hs.currentHists['Date'][z]) + ",")
                        ft.write(sym +",")
                        ft.write('buy,')
                        ft.write(str(portfolio.target_amount)+",")
                        ft.write(str(self.hs.currentHists['VIX Close'][z])+",")
                        ft.write(str(portfolio.get_total_btc_value_no_sell(end_prices)[0])+ " \n")
                        #print("sold ", sym)
                else:
                    ft.write(str(self.hs.currentHists['Date'][z]) + ",")
                    ft.write(sym +",")
                    ft.write('none,')
                    ft.write(str(-1)+",")
                    ft.write(str(self.hs.currentHists['VIX Close'][z])+",")
                    ft.write(str(portfolio.get_total_btc_value_no_sell(end_prices)[0])+ " \n")
                    #skip the hold case because we just dont buy or sell heh
        result_val = portfolio.get_total_btc_value(end_prices)
        print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2], p_name)
        ft = result_val[0]
        return ft

    def solve(self, network):
        return self.evaluate(network) >= self.highest_returns

    def report_back(self, portfolio, prices):
        print(portfolio.get_total_btc_value(prices))


# Create the population and run the XOR task by providing the above fitness function.


pt = PurpleTrader(144)
pt.run_champs()