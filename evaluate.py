#!/usr/bin/python3

import numpy as np

from kaggle_environments import make, evaluate

env = make("rps", debug = True)
results = env.run(["rock.py", "neural_network.py"])

print(str(results))

