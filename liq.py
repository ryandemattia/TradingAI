from flask import request
from flask import Flask, url_for
import random
import urllib.request 
import sys, os
from functools import partial
from itertools import product
import socket
from trading_purples import PurpleTrader
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
    
    def give_mad_props(self, to_whom):
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
            
# If run as script.

if __name__ == '__main__':
    cs = LiqMaster2000()
    print(cs.local_ip)
    '''
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


'''

