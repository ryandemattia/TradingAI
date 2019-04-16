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
import neat
import _pickle as pickle
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat_torch import ESNetwork
from NTree import nDimensionTree
# Local
class PurpleTrader:

    #needs to be initialized so as to allow for 62 outputs that return a coordinate

    # ES-HyperNEAT specific parameters.
    params = {"initial_depth": 3,
            "max_depth": 4,
            "variance_threshold": 0.00013,
            "band_threshold": 0.00013,
            "iteration_level": 3,
            "division_threshold": 0.00013,
            "max_weight": 3.0,
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
        self.hs.combine_polo_frames_vol_sorted()
        self.hd = hist_depth
        print(self.hs.currentHists.keys())
        self.end_idx = len(self.hs.hist_shaped[0])
        self.but_target = .1
        self.inputs = self.hs.hist_shaped.shape[0]*(self.hs.hist_shaped[0].shape[1])
        self.outputs = self.hs.hist_shaped.shape[0]
        self.leaf_names = []
        #num_leafs = 2**(len(self.node_names)-1)//2
        self.tree = nDimensionTree((0.0, 0.0, 0.0), 1.0, 1)
        self.tree.divide_childrens()
        self.set_substrate()
        self.set_leaf_names()
        self.epoch_len = hist_depth


    def set_leaf_names(self):
        for l in range(len(self.in_shapes[0])):
            self.leaf_names.append('leaf_one_'+str(l))
            self.leaf_names.append('leaf_two_'+str(l))
        #self.leaf_names.append('bias')
    def set_substrate(self):
        sign = 1
        x_increment = 1.0 / self.outputs
        y_increment = 1.0 / len(self.hs.hist_shaped[0][0])
        for ix in range(self.outputs):
            self.out_shapes.append((1.0-(ix*x_increment), 0.0, -1.0))
            for ix2 in range(self.inputs//self.outputs):
                if(ix2 >= len(self.tree.cs)-1):
                    treex = ix2 - len(self.tree.cs)-1
                else:
                    treex = ix2
                center = self.tree.cs[treex]
                self.in_shapes.append((center.coord[0]+(ix*x_increment), center.coord[1] - (ix2*y_increment), center.coord[2]+.5))
        self.subStrate = Substrate(self.in_shapes, self.out_shapes)
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
        g = neat.Checkpointer().restore_checkpoint(fname)
        bestFit = 0.0
        for gx in g.population:
            if g.population[gx].fitness != None:
                if g.population[gx].fitness > bestFit:
                    bestg = g.population[gx]
        g = bestg
        f.close()
        [the_cppn] = create_cppn(g, self.config, self.leaf_names, ['cppn_out'])
        self.cppn = the_cppn

    def load_net_easy(self, g):
        [the_cppn] = create_cppn(g, self.config, self.leaf_names, ['cppn_out'])
        self.cppn = the_cppn

    def run_champs(self):
        genomes = neat.Checkpointer.restore_checkpoint("tradegod-checkpoint-60").population
        fitness_data = {}
        best_fitness = 0.0
        for g_ix in genomes:
            self.load_net_easy(genomes[g_ix])
            start = self.hs.hist_full_size - self.epoch_len
            network = ESNetwork(self.subStrate, self.cppn, self.params)
            net = network.create_phenotype_network_nd('./champs_visualizedd-3/genome_'+str(g_ix))
            fitness = self.evaluate(net, network, start, genomes[g_ix], g_ix)

    def run_champ(self, g_ix):
        genomes = neat.Checkpointer.restore_checkpoint("./tradegod-checkpoint-28").population
        self.load_net_easy(genomes[g_ix])
        start = self.hs.hist_full_size - self.epoch_len
        network = ESNetwork(self.subStrate, self.cppn, self.params)
        net = network.create_phenotype_network_nd('./champs_visualizedd3/genome_'+str(g_ix))
        fitness = self.evaluate(net, network, start, genomes[g_ix], g_ix)

    def evaluate(self, network, es, rand_start, g, p_name):
        portfolio_start = 1.0
        portfolio = CryptoFolio(portfolio_start, list(self.hs.currentHists.keys()))
        end_prices = {}
        port_ref = portfolio_start
        with open('./champs_histd3/trade_hist'+ str(p_name) + '.txt', 'w') as ft:
            ft.write('date,symbol,type,amnt,price,current_balance \n')
            for z in range(self.hs.hist_full_size-377, self.hs.hist_full_size -1):
                active = self.get_one_epoch_input(z)
                signals = []
                network.reset()
                for n in range(1, self.hd+1):
                    out = network.activate(active[self.hd-n])
                for x in range(len(out)):
                    signals.append(out[x])
                    sym2 = list(self.hs.currentHists.keys())[x]
                    end_prices[sym2] = self.hs.currentHists[sym2]['close'][self.hs.hist_full_size-1]
                sorted_shit = np.argsort(signals)[::-1]
                #rng = iter(shuffle(rng))
                for x in sorted_shit:
                    sym = list(self.hs.currentHists.keys())[x]
                    #print(out[x])
                    #try:
                    if(out[x] < -.5):
                        #print("selling")
                        did_sell = portfolio.sell_coin(sym, self.hs.currentHists[sym]['close'][z])
                        if did_sell:
                            ft.write(str(self.hs.currentHists[sym]['date'][z]) + ",")
                            ft.write(sym +",")
                            ft.write('sell,')
                            ft.write(str(portfolio.ledger[sym])+",")
                            ft.write(str(self.hs.currentHists[sym]['close'][z])+",")
                            ft.write(str(portfolio.get_total_btc_value_no_sell(end_prices)[0])+ " \n")
                        #print("bought ", sym)
                    elif(out[x] > .5):
                        did_buy = portfolio.buy_coin(sym, self.hs.currentHists[sym]['close'][z])
                        if did_buy:
                            portfolio.target_amount = .1 + (out[x] * .1)
                            ft.write(str(self.hs.currentHists[sym]['date'][z]) + ",")
                            ft.write(sym +",")
                            ft.write('buy,')
                            ft.write(str(portfolio.target_amount)+",")
                            ft.write(str(self.hs.currentHists[sym]['close'][z])+",")
                            ft.write(str(portfolio.get_total_btc_value_no_sell(end_prices)[0])+ " \n")
                    else:
                        ft.write(str(self.hs.currentHists[sym]['date'][z]) + ",")
                        ft.write(sym +",")
                        ft.write('none,')
                        ft.write("0.0,")
                        ft.write(str(self.hs.currentHists[sym]['close'][z])+",")
                        ft.write(str(portfolio.get_total_btc_value_no_sell(end_prices)[0])+ " \n")
                        #print("sold ", sym)
                new_ref = portfolio.get_total_btc_value_no_sell(end_prices)[0]
                if(new_ref > 1.05 * port_ref):
                    port_ref = portfolio.get_total_btc_value_no_sell(end_prices)[0]
                    portfolio.start = port_ref
                    #skip the hold case because we just dont buy or sell heh
        result_val = portfolio.get_total_btc_value(end_prices)
        print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2], p_name)
        ft = result_val[0]
        return ft

    def solve(self, network):
        return self.evaluate(network) >= self.highest_returns

    def report_back(self, portfolio, prices):
        print(portfolio.get_total_btc_value(prices))

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


pt = PurpleTrader(8)
pt.run_champ(3844)
