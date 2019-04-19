
### IMPORTS ###
import random
import sys, os
from functools import partial
from itertools import product
from pytorch_neat.cppn import create_cppn
# Libs
import numpy as np
from hist_service import HistWorker
from crypto_evolution import CryptoFolio
from random import randint, shuffle
# Local
import neat.ctrnn
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
    rand_start = 0


    in_shapes = []
    out_shapes = []
    def __init__(self, hist_depth):
        self.hs = HistWorker()
        self.hs.combine_binance_frames_vol_sorted(8)
        self.hd = hist_depth
        print(self.hs.currentHists.keys())
        self.end_idx = len(self.hs.hist_shaped[0])
        self.but_target = .1
        self.inputs = self.hs.hist_shaped.shape[0]*(self.hs.hist_shaped[0].shape[1])
        self.outputs = len(self.hs.coin_dict)
        print(self.inputs, self.outputs)
        self.epoch_len = 144
        #self.node_names = ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'weight']
        self.leaf_names = []
        #num_leafs = 2**(len(self.node_names)-1)//2
        self.tree = nDimensionTree((0.0, 0.0, 0.0), 1.0, 1)
        self.tree.divide_childrens()
        self.set_substrate()
        self.set_leaf_names()


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

    def set_portfolio_keys(self, folio):
        for k in self.hs.currentHists.keys():
            folio.ledger[k] = 0

    def get_one_epoch_input(self,end_idx):
        master_active = []
        for x in range(0, self.hd):
            active = []
            
            for y in range(0, self.outputs):
                try:
                    sym_data = self.hs.hist_shaped[y][end_idx-x]
                    
                    active += sym_data.tolist()
                except:
                    print('error')
            master_active.append(active)
        
        return master_active

    def evaluate(self, g, config):
        rand_start = self.rand_start
        [cppn] = create_cppn(g, config, self.leaf_names, ['cppn_out'])
        net = ESNetwork(self.subStrate, cppn, self.params)
        network = net.create_phenotype_network_nd()
        portfolio_start = 1.0
        key_list = list(self.hs.currentHists.keys())
        portfolio = CryptoFolio(portfolio_start, self.hs.coin_dict)
        end_prices = {}
        buys = 0
        sells = 0
        if(len(g.connections) > 0.0):
            for z in range(rand_start, rand_start+self.epoch_len):
                active = self.get_one_epoch_input(z)
                signals = []
                network.reset()
                for n in range(1, self.hd+1):
                    out = network.activate(active[self.hd-n])
                for x in range(len(out)):
                    signals.append(out[x])
                #rng = iter(shuffle(rng))
                sorted_shit = np.argsort(signals)[::-1]
                #print(sorted_shit, len(sorted_shit))
                #print(len(sorted_shit), len(key_list))
                for x in sorted_shit:
                    sym = self.hs.coin_dict[x]
                    #print(out[x])
                    #try:
                    if(out[x] < -.5):
                        #print("selling")
                        portfolio.sell_coin(sym, self.hs.currentHists[sym]['close'][z])
                        #print("bought ", sym)
                    if(out[x] > .5):
                        #print("buying")
                        portfolio.target_amount = .1 + (out[x] * .1)
                        portfolio.buy_coin(sym, self.hs.currentHists[sym]['close'][z])
                        #print("sold ", sym)
                    #skip the hold case because we just dont buy or sell hehe
                    if(z > self.epoch_len+rand_start-2):
                        end_prices[sym] = self.hs.currentHists[sym]['close'][z]
            result_val = portfolio.get_total_btc_value(end_prices)
            print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2])
            ft = result_val[0]
        else:
            ft = 0.0
        return ft


    def eval_fitness(self, genomes, config):
        self.rand_start = randint(0+self.hd, self.hs.hist_full_size - self.epoch_len)
        runner = neat.ParallelEvaluator(4, self.evaluate)
        runner.evaluate(genomes, config)

# Create the population and run the XOR task by providing the above fitness function.
def run_pop(task, gens):
    pop = neat.population.Population(task.config)
    checkpoints = neat.Checkpointer(generation_interval=1, time_interval_seconds=None, filename_prefix='tradegod-checkpoint-')
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(checkpoints)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    winner = pop.run(task.eval_fitness, gens)
    print("es trade god summoned")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    task = PurpleTrader(8)
    #print(task.trial_run())
    winner = run_pop(task, 89)[0]
    print('\nBest genome:\n{!s}'.format(winner))

    # Verify network output against training data.
    print('\nOutput:')
    [cppn] = create_cppn(winner, task.config, task.leaf_names, ['cppn_out'])
    network = ESNetwork(task.subStrate, cppn, task.params)
    with open('es_trade_god_cppn_3d.pkl', 'wb') as output:
        pickle.dump(winner, output)
    #draw_net(cppn, filename="es_trade_god")
    winner_net = network.create_phenotype_network_nd('dabestest.png')  # This will also draw winner_net.

    # Save CPPN if wished reused and draw it to file.
    draw_net(cppn, filename="es_trade_god")


    '''
    for x in range(len(task.hs.hist_shaped[0])):
        print(task.hs.hist_shaped[1][x][3],task.hs.hist_shaped[0][x][3])
    '''
