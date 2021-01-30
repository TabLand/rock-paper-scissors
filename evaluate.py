#!/usr/bin/python3

import numpy as np

from kaggle_environments import make, evaluate

env = make("rps", debug = True, configuration={ "epsideSteps": 200})

env.run(
    ["markov_agent.py", "neural_network.py"]
)
