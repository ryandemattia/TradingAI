
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
from peas.peas.methods import hyperneat.HyperNEATDeveloper, hyperneat.Substrate

from peas.peas.methods.neat import NEATPopulation, NEATGenotype
from peas.peas.methods.evolution import SimplePopulation

class Trading_Task:

    EPSILON = 1e-100

    start_idx = 0
    
    portfolio_list = []
    

    def __init__(self):
        self.hs = HistWorker()
        self.end_idx = len(self.hs.currentHists["DASH"])
        self.inputs = self.hs.hist_shaped.shape[0]*self.hist_shaped[0].shape[1]
        self.outputs = self.end_idx * 3 # times by three for buy | sell | hodl(pass)
        #self.port = CryptoFolio(1)

    
    def set_portfolio_keys(folio):
        for k in self.hs.currentHists.keys:
            folio.ledger[k] = 0


    def evaluate(self, network, verbose=False):
        portfolio = CryptoFolio(1)
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        for x in range()
        network.feed()


    def run(generations=100, popsize=100):
                
        substrate = Substrate()
        substrate.add_nodes((self.hs.hist_shape.shape[0],), 'l')
        substrate.add_connections('l', 'l')
        geno = lambda: NEATGenotype(feedforward=True, inputs=self.inputs, weight_range=(-3.0, 3.0), 
                                       prob_add_conn=0.3, prob_add_node=0.03,
                                       types=['sin', 'ident', 'gauss', 'sigmoid', 'abs'])
        pop = NEATPopulation(geno, popsize=popsize, target_species=8)
        developer = HyperNEATDeveloper(substrate=substrate, add_deltas=False, sandwich=False)