
### IMPORTS ###
import random
import sys, os
from functools import partial
from itertools import product

# Libs
import numpy as np
# Local
from pureples import neat
import neat.nn
import cPickle as pickle
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork

'''
the aim of this class is to abstract what is left to the developer
to configure, namely the input and out put cords, this abstraction
would be significant for obvious reasons namely ease of 
'''


class AWholeNewPurp:
    
    def __init__(self, train_data):