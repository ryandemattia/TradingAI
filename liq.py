from flask import request
from flask import Flask, url_for
import random
import requests
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
    app = Flask(__name__)
    local_ip = ''
    net_data = {
        'peers': ['localhost:8080, localhost:5050'],
        'best_pkl': '',
        'local_pkl': '',
        'global_gens': 0,
        'local_gens': 0,
    }
    
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
    
    @app.route("/")
    def best_pickle(request):
        return "heres your pickle"
        
    @app.route("/best_pickle/<str:pkl_address>/<str:pkl_name>")
    def peer_posting_best(request):
        
    
    
    def get_endpoint(self, ep):
        r = requests.get(url='')
        print(r.json())
    #used by get_global_best to retrieve best pkl eheheh
    def retrieve_pkl(pkl_address, pkl_name):
        
    
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
# If run as script.

if __name__ == '__main__':
    cs = LiqMaster2000()
    print(cs.local_ip)
    '''
    task = PurpleTrader(5)
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

