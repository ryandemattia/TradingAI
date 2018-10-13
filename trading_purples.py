
### IMPORTS ###
import random
import sys, os
from functools import partial
from itertools import product

# Libs
import numpy as np
from hist_service import HistWorker
from crypto_evolution import CryptoFolio
# Local
import neat.nn
import _pickle as pickle
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork
# Local
class PurpleTrader:
    
    #needs to be initialized so as to allow for 62 outputs that return a coordinate

    # ES-HyperNEAT specific parameters.
    params = {"initial_depth": 0, 
            "max_depth": 1, 
            "variance_threshold": 0.03, 
            "band_threshold": 0.3, 
            "iteration_level": 1,
            "division_threshold": 0.5, 
            "max_weight": 5.0, 
            "activation": "sigmoid"}

    # Config for CPPN.
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                'config_trader')
                                
    start_idx = 0
    highest_returns = 0
    portfolio_list = []

    in_shapes = []
    out_shapes = []
    def __init__(self):
        self.hs = HistWorker()
        self.end_idx = len(self.hs.currentHists["DASH"])
        self.but_target = .1
        self.inputs = self.hs.hist_shaped.shape[0]*self.hs.hist_shaped[0].shape[1]
        self.outputs = self.hs.hist_shaped.shape[0]
        for ix in range(self.outputs):
            self.out_shapes.append((0, ix))
            for ix2 in range(len(self.hs.hist_shaped[0][0])):
                self.in_shapes.append((ix, ix2))
        self.subStrate = Substrate(self.in_shapes, self.out_shapes)
        
        
    def set_portfolio_keys(self, folio):
        for k in self.hs.currentHists.keys():
            folio.ledger[k] = 0

    def get_one_bar_input_2d(self, end_idx):
        active = []
        for x in range(0, self.outputs):
            sym_data = self.hs.hist_shaped[x][end_idx] 
            for i in range(len(sym_data)):
                active.append((x, sym_data[i]))
        return np.asarray(active)

    def evaluate(self, network, verbose=False):
        portfolio = CryptoFolio(1, self.hs.coin_dict)
        active = {}
        results = {}
        for z in range(0, 14):
            '''
            if(z == 0):
                old_idx = 1
            else:
                old_idx = z * 5
            new_idx = (z + 1) * 5
            '''
            active = self.get_one_bar_input_2d(z)
            results[z] = network.activate(active)
        #first loop sets up buy sell hold signal result from the net,
        #we want to gather all 14 days of 
        for i in range(0, 14):
            out = results[i]
            print(len(out))
            for x in range(len(out)):
                sym = self.hs.coin_dict[x]
                #print(out[x])
                if isinstance(out[x], float):
                    try:
                        if(out[x] > .6):
                            portfolio.buy_coin(sym, self.hs.currentHists[sym]['close'][i])
                        elif(out[x] < 0.3):
                            portfolio.sell_coin(sym, self.hs.currentHists[sym]['close'][i])
                    except:
                        print('error', sym, i)
                #skip the hold case because we just dont buy or sell hehe
        end_ts = self.hs.hist_shaped[0][14][1]
        result_val = portfolio.get_total_btc_value(end_ts)
        print(result_val)
        return result_val

    def solve(self, network):
        return self.evaluate(network) >= self.highest_returns
        

    def eval_fitness(self, genomes, config):
    
        for idx, g in genomes:

            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            network = ESNetwork(self.subStrate, cppn, self.params)
            net = network.create_phenotype_network()
            g.fitness = self.evaluate(net)
        


# Create the population and run the XOR task by providing the above fitness function.
def run(task, gens):
    pop = neat.population.Population(task.config)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    winner = pop.run(task.eval_fitness, gens)
    print("es_hyperneat_xor_small done")
    return winner, stats


# If run as script.
if __name__ == '__main__':
    task = PurpleTrader()
    winner = run(task, 300)[0]
    print('\nBest genome:\n{!s}'.format(winner))

    # Verify network output against training data.
    print('\nOutput:')
    cppn = neat.nn.FeedForwardNetwork.create(winner, task.config)
    network = ESNetwork(task.subStrate, cppn, task.params)
    winner_net = network.create_phenotype_network(filename='es_hyperneat_xor_small_winner.png')  # This will also draw winner_net.

    # Save CPPN if wished reused and draw it to file.
    draw_net(cppn, filename="es_hyperneat_xor_small_cppn")
    with open('es_hyperneat_xor_small_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output, pickle.HIGHEST_PROTOCOL)

    '''
    for x in range(len(task.hs.hist_shaped[0])):
        print(task.hs.hist_shaped[1][x][3],task.hs.hist_shaped[0][x][3])
    '''
    