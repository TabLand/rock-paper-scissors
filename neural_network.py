#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd

NN = None

class NeuralNetwork:
    debug_window = 5
    num_features = 10
    debug        = True
    classes      = ["rock", "paper", "scissors"]
    train_runs   = 60

    def train(self, observation):
        if hasattr(observation,'lastOpponentAction'):
            last_opponent_move = observation.lastOpponentAction
        else:
            last_opponent_move = np.random.randint(3)

        latest_inputs = np.array([np.concatenate((self.op_moves, self.my_moves))])

        self.train_X = np.append(self.train_X, latest_inputs)
        self.train_Y = np.append(self.train_Y, last_opponent_move)
        
        num_examples = self.train_Y.size

        train_X_reshaped = self.train_X.reshape(num_examples, self.num_features * 2)
        train_Y_reshaped = self.train_Y.reshape(num_examples, 1)
        
        if self.debug:
            self.debug_train(train_X_reshaped.shape, train_Y_reshaped.shape, num_examples)
        
        self.model.fit(train_X_reshaped, train_Y_reshaped, epochs=self.train_runs, verbose=0)
        self.op_moves = np.append(self.op_moves[1-self.num_features:], last_opponent_move)
        
    def predict(self):
        X = np.array([np.concatenate((self.op_moves, self.my_moves))])
        self.chances = self.model.predict(X)
        self.my_next_move = ((np.argmax(self.chances) + 1) % 3).item()

        if self.debug:
            self.debug_combo()

        self.my_moves = np.append(self.my_moves[1-self.num_features:], self.my_next_move)

    def save_input(self):
        return 0

    def save_model(self):
        return 0

    def load_model(self):
        return 0

    def debug_combo(self):
        self.debug_history()
        self.debug_prediction()
        self.debug_next_move()
        self.calculate_scores()

    def debug_train(self, shape_X, shape_Y, num_examples):
        print("Shape train X: " + str(shape_X))
        print("Shape train Y: " + str(shape_Y))
        print("Num Examples: " + str(num_examples))

    def debug_history(self):
        print("Actual history kept for OP:    " + str(self.op_moves.size))
        print("Actual history kept for NN: " + str(self.my_moves.size))
        for i in range(self.debug_window, 0, -1):
            rounds_ago = i
            self.op_move  = self.classes[self.op_moves[self.op_moves.size - i ]]
            self.my_move  = self.classes[self.my_moves[self.my_moves.size - i ]]
            print("OP History:" + str(self.op_moves))
            print("NN History:" + str(self.my_moves))
            print(str(rounds_ago) + " round(s) ago, OP played: " + str(self.op_move) + ", and NN played: " + str(self.my_move))

    def debug_prediction(self):
        i = 0
        total = np.sum(self.chances)
        self.chances = self.chances.T
        for i in range(0, len(self.chances)):
            move   = self.classes[i]
            chance = self.chances[i]
            print("NN predict's OP chance of " + move + ": " + str(chance/total))

    def debug_next_move(self):
        print("NN will play: " + self.classes[self.my_next_move])

    def calculate_scores(self):
        self.my_last_move = self.my_moves[-1]
        self.op_last_move = self.op_moves[-1]

        if self.my_last_move == self.op_last_move:
            print("DRAW!!")
        elif (self.my_last_move - 1) % 3 == self.op_last_move:
            self.my_score = self.my_score + 1
        else:
            self.op_score = self.op_score + 1
        print("NN: " + str(self.my_score) + "\tOP: " + str(self.op_score))


    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(self.num_features * 2),
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
        self.my_moves     = np.zeros(self.num_features, dtype=int)
        self.op_moves     = np.zeros(self.num_features, dtype=int)
        self.my_score     = 0
        self.op_score     = 0
        self.train_X      = np.random.randint(2, size=self.num_features * 2)
        self.train_Y      = np.array([np.random.randint(2)])
        self.build_model()

    def step(self, observation, configuration):

        self.train(observation)
        self.predict()
       
        return self.my_next_move

def neural_network(observation, configuration):
    global NN
    if NN == None:
        NN = NeuralNetwork()
    NN.step(observation, configuration)

agents = { 
    "neural_network": neural_network
}
