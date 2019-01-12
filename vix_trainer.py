
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
        self.generation_index = 0
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
        self.epoch_len = 55
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

    def evaluate(self, network, es, rand_start, verbose=False):
        portfolio_start = 100000
        portfolio = CryptoFolio(portfolio_start, self.hs.coin_dict)
        portfolio.ledger['vix'] = 0.0
        end_prices = {}
        buys = 0
        sells = 0
        for z in range(rand_start, rand_start+self.epoch_len):
            active = self.get_one_epoch_input(z)
            network.reset()
            #print(len(active))
            for n in range(0, self.hd):
                n+=1
                out = network.activate(active[self.hd-n])
            #print(len(out))
            rng = len(out)
            #rng = iter(shuffle(rng))
            sym = "vix"
                #print(out[x])
                #try:
            if(out[0] < -.5):
                #print("selling")
                portfolio.sell_coin(sym, self.hs.currentHists['VIX Close'][z])
                #print("bought ", sym)
            elif(out[0] > .5):
                #print("buying")
                portfolio.buy_coin(sym, self.hs.currentHists['VIX Close'][z])
                #print("sold ", sym)
            #skip the hold case because we just dont buy or sell hehe
            end_prices[sym] = self.hs.currentHists['VIX Close'][self.epoch_len+rand_start]
        result_val = portfolio.get_total_btc_value(end_prices)
        print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2])
        ft = result_val[0]
        if(ft == portfolio_start):
            ft = portfolio_start/2
        return ft

    def solve(self, network):
        return self.evaluate(network) >= self.highest_returns


    def eval_fitness(self, genomes, config):
        r_start = randint(0+self.hd, self.hs.hist_full_size - self.epoch_len)
        self.generation_index += 1
        fitter = genomes[0]
        fitter_val = 0.0 
        for idx, g in genomes:
            [cppn] = create_cppn(g, config, self.leaf_names, ['cppn_out'])
            network = ESNetwork(self.subStrate, cppn, self.params)
            net = network.create_phenotype_network_nd()
            new_fit = self.evaluate(net, network, r_start, g)
            if(new_fit > fitter_val):
                fitter = g
                fitter_val = new_fit
                with open('./champs/perpetual_champion_'+str(self.generation_index)+'.pkl', 'wb') as output:
                    pickle.dump(fitter, output)
                print("latest_saved")
            g.fitness = new_fit



# Create the population and run the XOR task by providing the above fitness function.
def run_pop(task, gens):
    pop = neat.population.Population(task.config)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    winner = pop.run(task.eval_fitness, gens)
    print("es trade god summoned")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    task = PurpleTrader(34)
    winner = run_pop(task, 21)[0]
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
    #draw_net(cppn, filename="es_trade_god")

    '''
    for x in range(len(task.hs.hist_shaped[0])):
        print(task.hs.hist_shaped[1][x][3],task.hs.hist_shaped[0][x][3])
    '''
