#!/usr/bin/python3

import neural_network as nn

class Observation:
    def __init__(self, move):
        self.lastOpponentMove = move

def get_input():
    result = input("Your next move please (rock, paper, scissors): ")
    while result not in classes:
        result = input("Please pick between rock, paper or scissors: ")

    return result

classes = ["rock", "paper", "scissors"]
configuration = None
observation = Observation(0)

while True:
    current_nn_move    = nn.neural_network(observation, configuration)
    current_human_move = classes.index(get_input())

    observation = Observation(current_human_move)
    
    print("You: " + classes[current_human_move] + "\t Computer:" + classes[current_nn_move])


