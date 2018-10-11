
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
from pureples import neat
import neat.nn
import cPickle as pickle
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork
# Local
class PurpleTrader:
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
                                'config_cppn_xor')
                                
    def __init__(self, inputCords, outputCords):
        