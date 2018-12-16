
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
from pureples.es_hyperneat.es_hyperneat_torch import ESNetwork
# Local

class LiqMaster2000:
    app = Flask(__name__, static_url_path='champ')
    local_ip = ''
    net_data = {
        'peers': ['localhost:8080, localhost:5050'],
        'peer_pkls': {},
        'best_pkl': '',
        'local_pkl': '',
        'global_gens': 0,
        'local_gens': 0,
        'training': False,
    }
    peer_data = {}
    config = {}
    
    
    def get_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP
    
    @app.route("/best_local")
    def best_local_pkl(request):c
        return send_from_directory('champ', 'best_local.pkl')
        
    @app.route("/new_peer_net/<str:peer_address>")
    def add_new_peer(request):
        self.net_data.peers.append(peer_address)
        return self.net_data
        
    @app.route("/best_pickle/<str:peer_address>/<str:pkl_name")
    def peer_posting_best(request):
        p_ep = peer_address+'/'+pkl_name
        p_file = peer_address+'_'+pkl_name
        self.net_data.peer_pkls[peer_address] = pkl_name
        self.get_peer_pkl(p_ep, p_file)
        
    def get_peer_pkl(self, pkl_add, file_p):
        try:
            p_pickle = urllib.request.urlretrieve(pkl_add, 'champ/'+file_p)
        except:
            print('error retrieving peers pickle')
        return p_pickle
    
    # Create the population and run the XOR task by providing the above fitness function.
    def run_pop(self, task, gens):
        pop = neat.population.Population(task.config)
        stats = neat.statistics.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    
        winner = pop.run(task.eval_fitness, gens)
        print("es trade god summoned")
        return winner, stats


    def __init__(self, seed_peer=''):
        self.local_ip = self.get_ip()
        return
    
    def peer_v_peer(self, g1, g2):
        cppn = neat.nn.FeedForwardNetwork.create(g1, config)
        network = ESNetwork(self.subStrate, cppn, self.params)
        net = network.create_phenotype_network_nd()
        g1.fitness = self.evaluate(net, network, r_start)
        cppn2 = neat.nn.FeedForwardNetwork.create(g2, config)
        network2 = ESNetwork(self.subStrate, cppn, self.params)
        net2 = network.create_phenotype_network_nd()
        g2.fitness2 = self.evaluate(net, network, r_start)
        if g1 > g2:
            return g1
        else:
            return g2
            






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
        r_start = self.epoch_len
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = ESNetwork(self.subStrate, cppn, self.params)
        net = network.create_phenotype_network_nd()
        g.fitness = self.evaluate(net, network, r_start)
        return g.fitness
        
        
        
# If run as script.
def main_loopies():
    cs = LiqMaster2000()
    print(cs.local_ip)
    task = PurpleTrader(13)
    winner = run_pop(task, 21)[0]
    print('\nBest genome:\n{!s}'.format(winner))

    # Verify network output against training data.
    print('\nOutput:')
    cppn = neat.nn.FeedForwardNetwork.create(winner, task.config)
    network = ESNetwork(task.subStrate, cppn, task.params)
    with open('local_winner.pkl', 'wb') as output:
        pickle.dump(cppn, output)
    #draw_net(cppn, filename="es_trade_god")
    winner_net = network.create_phenotype_network_nd('dabestest.png')  # This will also draw winner_net.
    
    # Save CPPN if wished reused and draw it to file.
    #draw_net(cppn, filename="es_trade_god")

