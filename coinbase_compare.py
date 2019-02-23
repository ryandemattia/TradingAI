
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
class CoinBaseComparer:

    #needs to be initialized so as to allow for 62 outputs that return a coordinate

    # ES-HyperNEAT specific parameters.
    params = {"initial_depth": 3,
            "max_depth": 4,
            "variance_threshold": 0.00013,
            "band_threshold": 0.00013,
            "iteration_level": 3,
            "division_threshold": 0.00013,
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
        self.hs.combine_live_usd_frames()
        self.hd = hist_depth
        print(self.hs.currentHists.keys())
        self.end_idx = len(self.hs.currentHists["ETHUSD"])
        self.but_target = .1
        self.inputs = self.hs.hist_shaped.shape[0]*(self.hs.hist_shaped[0].shape[1])
        self.outputs = self.hs.hist_shaped.shape[0]
        sign = 1
        for ix in range(1,self.outputs+1):
            sign = sign *-1
            self.out_shapes.append((0.0-(sign*.005*ix), -1.0, -1.0))
            for ix2 in range(1,(self.inputs//self.outputs)+1):
                self.in_shapes.append((0.0+(sign*.01*ix2), 0.0-(sign*.01*ix2), 1.0))
        self.subStrate = Substrate(self.in_shapes, self.out_shapes)
        self.epoch_len = 89
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
            active = []
            #print(self.outputs)
            for y in range(0, self.outputs):
                try:
                    sym_data = self.hs.hist_shaped[y][end_idx-x]
                    #print(len(sym_data))
                    active += sym_data.tolist()
                except:
                    print('error')
            master_active.append(active)
        #print(active)
        return master_active

    def load_net(self, fname):
        f = open(fname,'rb')
        g = pickle.load(f)
        f.close()
        [the_cppn] = create_cppn(g, self.config, self.leaf_names, ['cppn_out'])
        self.cppn = the_cppn

    def run_champs(self):
        genomes = os.listdir(os.path.join(os.path.dirname(__file__), 'champs_cppn_7_hidden'))
        for g_ix in range(len(genomes)):
            genome = self.load_net('./champs_cppn_7_hidden/'+genomes[g_ix])
            start = self.hs.hist_full_size - self.epoch_len
            network = ESNetwork(self.subStrate, self.cppn, self.params)
            net = network.create_phenotype_network_nd('./champs_visualized/genome_'+str(g_ix))
            fitness = self.evaluate(net, network, start, g_ix)

    def evaluate(self, network, es, rand_start, g):
        portfolio_start = 100000
        portfolio = CryptoFolio(portfolio_start, self.hs.coin_dict)
        end_prices = {}
        buys = 0
        sells = 0
        th = []
        with open('./champs_hist/trade_hist'+str(g)+'.txt', 'w') as ft:
            ft.write('date,symbol,type,amnt,price,current_balance \n')
            for z in range(self.hd, self.hs.hist_full_size -1):
                active = self.get_one_epoch_input(z)
                network.reset()
                for n in range(1, self.hd+1):
                    out = network.activate(active[self.hd-n])
                #print(len(out))
                rng = len(out)
                for x in range(rng):
                    sym2 = self.hs.coin_dict[x]
                    end_prices[sym2] = self.hs.currentHists[sym2]['Close'][self.hs.hist_full_size-1]
                #rng = iter(shuffle(rng))
                for x in np.random.permutation(rng):
                    sym = self.hs.coin_dict[x]
                    #print(out[x])
                    #try:
                    if(out[x] < -.5):
                        #print("selling")
                        did_sell = portfolio.sell_coin(sym, self.hs.currentHists[sym]['Close'][z])
                        if did_sell:
                            ft.write(self.hs.currentHists[sym]['Date'][z] + ",")
                            ft.write(sym +",")
                            ft.write('sell,')
                            ft.write(str(portfolio.ledger[sym])+",")
                            ft.write(str(self.hs.currentHists[sym]['Close'][z])+",")
                            ft.write(str(portfolio.get_total_btc_value_no_sell(end_prices)[0])+ " \n")
                        #print("bought ", sym)
                    elif(out[x] > .5):
                        did_buy = portfolio.buy_coin(sym, self.hs.currentHists[sym]['Close'][z])
                        if did_buy:
                            ft.write(self.hs.currentHists[sym]['Date'][z] + ",")
                            ft.write(sym +",")
                            ft.write('buy,')
                            ft.write(str(portfolio.target_amount)+",")
                            ft.write(str(self.hs.currentHists[sym]['Close'][z])+",")
                            ft.write(str(portfolio.get_total_btc_value_no_sell(end_prices)[0])+ " \n")
                        #print("sold ", sym)
                    #skip the hold case because we just dont buy or sell heh
        result_val = portfolio.get_total_btc_value(end_prices)
        print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2])
        ft = result_val[0]
        return ft

    def solve(self, network):
        return self.evaluate(network) >= self.highest_returns

    def trial_run(self):
        r_start = 0
        file = open("es_trade_god_cppn_3d.pkl",'rb')
        [cppn] = pickle.load(file)
        network = ESNetwork(self.subStrate, cppn, self.params)
        net = network.create_phenotype_network_nd()
        fitness = self.evaluate(net, network, r_start)
        return fitness

    def eval_fitness(self, genomes, config):
        r_start = randint(0+self.hd, self.hs.hist_full_size - self.epoch_len)
        fitter = genomes[0]
        fitter_val = 0.0 
        for idx, g in genomes:
            [cppn] = create_cppn(g, config, self.leaf_names, ['cppn_out'])
            network = ESNetwork(self.subStrate, cppn, self.params)
            net = network.create_phenotype_network_nd('current_net.png')
            g.fitness = self.evaluate(net, network, r_start)
            if(g.fitness > fitter_val):
                fitter = g
                fitter_val = g.fitness
        with open('./champs/perpetual_champion_'+str(fitter.key)+'.pkl', 'wb') as output:
            pickle.dump(fitter, output)
        print("latest_saved")
# Create the population and run the XOR task by providing the above fitness function.


pt = CoinBaseComparer(89)
pt.run_champs()