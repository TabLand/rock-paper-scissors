#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd

NN = None

class NeuralNetwork:
    history_size = 10
    debug_window = 5
    my_moves     = np.zeros(history_size, dtype=int)
    op_moves     = np.zeros(history_size, dtype=int)
    debug        = True
    classes      = ["rock", "paper", "scissors"]
    train_runs   = 60
    my_score     = 0
    op_score     = 0 
    model        = None

    def debug_history():
        print("Actual history kept for OP:    " + str(self.op_moves.size))
        print("Actual history kept for NN: " + str(self.my_moves.size))
        for i in range(self.debug_window, 0, -1):
            rounds_ago = i
            print(str(self.op_moves))
            op_move    = classes[self.op_moves[self.op_moves.size - i ]]
            my_move    = classes[self.my_moves[self.my_moves.size - i ]]
            print(str(rounds_ago) + " round(s) ago, OP played: " + str(op_move) + ", and NN played: " + str(my_move))

    def debug_prediction():
        i = 0
        total = np.sum(chances)
        chances = chances.T
        for i in range(0, len(chances)):
            move   = classes[i]
            chance = chances[i]
            print("NN predict's OP chance of " + move + ": " + str(chance/total))

    def debug_next_move():
        print("NN will play: " + self.classes[self.my_next_move])

    def calculate_scores():
        self.my_last_move = self.my_moves[-1]
        self.op_last_move = self.op_moves[-1]

        if (my_last_move - 1) % 3 == op_last_move:
            my_score = my_score + 1
        else:
            op_score = op_score + 1
        print("NN: " + str(my_score) + "\tOP: " + str(op_score))


    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(self.history_size),
            tf.keras.layers.Dense(30 , activation='sigmoid'),
            tf.keras.layers.Dense(20 , activation='sigmoid'),
            tf.keras.layers.Dense(10 , activation='sigmoid'),
            tf.keras.layers.Dense(8  , activation='relu'),
            tf.keras.layers.Dense(len(self.classes))
        ])

        self.model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    def __init__(self):
        self.build_model()

    def step(self, observation, configuration):
        X = np.array([self.op_moves])
        if hasattr(observation,'lastOpponentMove'):
            last_opponent_move = observation.lastOpponentMove
        else:
            last_opponent_move = np.random.randint(3)

        Y = np.array([last_opponent_move])
        self.model.fit(X, Y, epochs=self.train_runs, verbose=0)

        self.op_moves = np.append(self.op_moves[1-self.history_size:], last_opponent_move)
        X = np.array([self.op_moves])
        self.chances = self.model.predict(X)
        self.my_next_move = ((np.argmax(self.chances) + 1) % 3).item()

        self.my_moves = np.append(self.my_moves[1-self.history_size:], self.my_next_move)

        if debug:
            debug_history()
            debug_prediction()
            calculate_scores()
            debug_next_move()
       
        return self.my_next_move

def neural_network(observation, configuration):
    global NN
    if NN == None:
        NN = NeuralNetwork()
    NN.step(observation, configuration)

agents = { 
    "neural_network": neural_network
}
