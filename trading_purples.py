
### IMPORTS ###
import random
import sys, os
from functools import partial
from itertools import product

# Libs
import numpy as np
from hist_service import HistWorker
from crypto_evolution import CryptoFolio
from random import randint
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
        self.inputs = self.hs.hist_shaped.shape[0]*(self.hs.hist_shaped[0].shape[1] - 1)
        self.outputs = self.hs.hist_shaped.shape[0]
        for ix in range(self.outputs):
            self.out_shapes.append((ix,1))
            for ix2 in range(len(self.hs.hist_shaped[0][0])-1):
                self.in_shapes.append((ix, ix2))
        self.subStrate = Substrate(self.in_shapes, self.out_shapes)
        
        
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
        portfolio = CryptoFolio(1, self.hs.coin_dict)
        end_prices = {}
        buys = 0
        sells = 0 
        for z in range(rand_start, rand_start+89):
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
            for x in range(len(out)):
                sym = self.hs.coin_dict[x]
                print(out[x])
                try:
                    if(out[x] > .7):
                        #print("buying")
                        portfolio.buy_coin(sym, self.hs.currentHists[sym]['close'][z])
                    elif(out[x] < 0.3):
                        #print("selling")
                        portfolio.sell_coin(sym, self.hs.currentHists[sym]['close'][z])
                except:
                    print('error', sym)
                #skip the hold case because we just dont buy or sell hehe
        for y in range(len(out)):
            end_prices[self.hs.coin_dict[y]] = self.hs.hist_shaped[y][89][2]
        result_val = portfolio.get_total_btc_value(end_prices)
        print(result_val, "buys: ", portfolio.buys, "sells: ", portfolio.sells)
        return result_val

    def solve(self, network):
        return self.evaluate(network) >= self.highest_returns
        

    def eval_fitness(self, genomes, config):
        r_start = randint(0, self.hs.hist_full_size - 89)    
        for idx, g in genomes:

            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            network = ESNetwork(self.subStrate, cppn, self.params)
            net = network.create_phenotype_network()
            g.fitness = self.evaluate(net, network, r_start)
        


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
    task = PurpleTrader()
    winner = run_pop(task, 34)[0]
    print('\nBest genome:\n{!s}'.format(winner))

    # Verify network output against training data.
    print('\nOutput:')
    cppn = neat.nn.FeedForwardNetwork.create(winner, task.config)
    network = ESNetwork(task.subStrate, cppn, task.params)
    winner_net = network.create_phenotype_network(filename='es_god_trader_winner.png')  # This will also draw winner_net.

    # Save CPPN if wished reused and draw it to file.
    #draw_net(cppn, filename="es_trade_god")
    with open('es_trade_god_cppn.pkl', 'wb') as output:
        pickle.dump(cppn, output)

    '''
    for x in range(len(task.hs.hist_shaped[0])):
        print(task.hs.hist_shaped[1][x][3],task.hs.hist_shaped[0][x][3])
    '''
    