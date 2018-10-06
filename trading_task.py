
### IMPORTS ###
import random
import sys, os
from functools import partial
from itertools import product

# Libs
import numpy as np
# Libraries
import numpy as np
from hist_service import HistWorker
from crypto_evolution import CryptoFolio
# Local
from peas.peas.networks.rnn import NeuralNetwork
from peas.peas.methods import hyperneat

class Trading_Task:

    EPSILON = 1e-100

    start_idx = 0
    
    task_output_actions = {
        'BUY': 1,
        'SELL': 0.0,
        'HODL': 0.5,
    }
    
    portfolio_list = []
    

    def __init__(self):
        self.hs = HistWorker()
        self.end_idx = len(self.hs.currentHists["DASH"])
        self.inputs = len(self.hs.currentHists)*len(self.currentHists["DASH"].keys())
        self.outputs = self.end_idx * 3 # times by three for buy | sell | hodl(pass)
        self.strt_amnt = self.port.start

    
    def set_portfolio_keys(folio):
        for k in self.hs.currentHists.keys:
            folio.ledger[k] = 0
    
    def evaluate(pop, d):
        p_ordered = []
        for each p in pop:
            inserted = False
            g = p.genotype
            w = p.weights
            pf = self.portfolio_list[pop.port_index]
            p_score = pf.get_total_btc_value(d)
            for x in range(0, len(p_ordered)):
                if(p_ordered[x] < p_score):
                    p_ordered.insert(x, p_score)
            if(inserted == false):
                p_ordered.append(p_score)
        return p_ordered


    def evaluate(self, network, verbose=False):
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        
        network.feed()

    

    def run(method, setup, generations=100, popsize=100):    