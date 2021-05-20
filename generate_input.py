#!/usr/bin/python3

import numpy as np
from kaggle_environments import make, evaluate
from kaggle_environments.envs.rps.utils import get_score

episode_steps = 100

list_names = [
    "rock", 
    "paper", 
    "scissors",
    "hit_the_last_own_action",  
    "copy_opponent", 
    "reactionary", 
    "counter_reactionary", 
    "statistical", 
    "nash_equilibrium",
    "markov_agent", 
    "memory_patterns", 
    "multi_armed_bandit",
    "opponent_transition_matrix",
    "decision_tree_classifier",
    "statistical_prediction",
    "high_performance_rps_dojo",
    "geometry",
]

list_agents = [agent_name + ".py" for agent_name in list_names]

env = make("rps", configuration={"episodeSteps": episode_steps } )

print("Simulation of battles. It can take some time...")

for i in range(len(list_names)):
    for j in range(i + 1, len(list_names)):
        left_strategy = list_names[i]
        right_strategy = list_names[j]
        
        file_name = "input/" + left_strategy + "_vs_" + right_strategy + ".txt"
        f = open(file_name, "w")
         
        print("writing to " + file_name + "....")

        results = env.run([list_agents[i], list_agents[j]])
 

        for k in range(0, len(results)):
       
            left  = results[k][0].action
            right = results[k][1].action

            f.write(str(left) + "\t" + str(right) + "\n")

        f.close()
