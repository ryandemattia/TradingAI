
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
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat_torch import ESNetwork
# Local



class DeepThotNodeMaster(object):

    participants = []

    current_trainers = {}

    current_validations_open = {}

    current_states = {}

    token_table_name = ""

    def __init__(
        clusterName,
        clusterDescription,
        clusterPeers,
        from_checkpoint
    ):
    self.clusterName = clusterName
    self.clusterDescription = clusterDescription
    self.clusterPeers = clusterPeers
    self.from_checkpoint = from_checkpoint

