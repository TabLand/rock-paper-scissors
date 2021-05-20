#!/usr/bin/python3



import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from kaggle_environments import make, evaluate

env = make(
    "rps", 
    configuration={
        "episodeSteps": 1000,

    }
)
