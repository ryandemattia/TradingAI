from flask import request
from flask import Flask, url_for
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
from pureples.es_hyperneat.es_hyperneat import ESNetwork


app = Flask(__name__)

net_data = {
    'peers': ['localhost:8080, localhost:5050'],
    'best_pkl': '',
    'local_pkl': '',
    'global_gens': 0,
    'local_gens': 0,
}

config = {}


@route("/")
def best_pickle(request):
    