import numpy as np
from math import log10
import json
import itertools
from copy import copy


# X = np.logspace(log10(0.25), log10(1.18), 16)
X = np.logspace(log10(0.05), log10(1000), 64)
Xname = "A"
# Y = np.logspace(log10(9e-6), log10(0.0003), 16)
Y = np.logspace(log10(1e-7), log10(1e5), 64)
Yname = "gamma"

other_params = dict(
    D=10e6,
    adjustment_step=0.0004,
    fee_gamma=1.25e-3,
    ma_half_time=600,
    mid_fee=4e-4,
    out_fee=4e-3,
    n=2,
    log=0,
    price_threshold=0.0004,
    gamma=8e-4,
    ext_fee=0.0,
    A=10.0)

config = {
    'configuration': [],
    'datafile': [
        'eurusd'],
    'debug': 0}

for x, y in itertools.product(X, Y):
    params = copy(other_params)
    params[Xname] = x
    params[Yname] = y
    config['configuration'].append(params)

with open('configuration.json', 'w') as f:
    json.dump(config, f)
