#####################
#####################
##                 ##
##  Copyright (C)  ##
##   Brandon Cho   ##
##    2017-2018    ##
##                 ##
#####################
#####################

"""
Here are the notes referenced by the code.

1.  To retrieve the original score, use the logit (inverse sigmoid) function and multiply by DIV_SCORE.

2.  This is the data generator for the model -- it takes data from a PGN file and converts it into
    flattened arrays using the following process:
        a. Read a game from the PGN file.
        b. Take each individual position:
            i. Convert it to a 768-long one-hot array.
            ii. Run an evaluation function on each position.
        c. Return a batch of BATCH_SIZE positions to be fed into the model.
"""

import sys
import random
import keras
import numpy as np
import chess
import chess.pgn
import math
import time
import evalp # Local PST/value function

# Specific imports
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import backend as K

# Configuration
# Defaults: -, 10k, 100k, 1k, 1
PGN_FILE_PATH = "" # PGN file reference
#PGN_FILE_PATH = "/Users/bracho/Desktop/cgd/data.pgn"
PREVIOUS_MODEL = input("Previous model filepath (Enter to skip): ")
name_components = ["$TIME_SPENT", "$POS_EVALUATED", "W"] # The components in the name (use $VAR to access VAR)
BATCH_SIZE = 10000 # In different positions (not necessarily unique)
MAX_GENERATOR_SIZE = 100000
EPOCHS = 1000
DATA_UPDATES = 100
# Custom parameters (for naming -- changeable, but make sure to add code in loop to change value)
TIME_SPENT = int(input("Previous time spent on model (0 is default): "))
DATA_OFFSET = int(input("Data offset (default 0): ")) # Start at the Nth batch of BATCH_SIZE.
# Optimizer parameters
LEARNING_RATE = 1e-2
DECAY = 5e-6
MOMENTUM = 0.9
USE_NESTEROV = True
# Model.compile() parameters
LOSS_TYPE = "mse"
### --------- ###
pgn_file = open(PGN_FILE_PATH)
if(PREVIOUS_MODEL != ""):
    model = load_model(PREVIOUS_MODEL)
PIECE_MAP = {None:-1,"K":0,"Q":1,"R":2,"B":3,"N":4,"P":5,"k":6,"q":7,"r":8,"b":9,"n":10,"p":11}
### --------- ###
def K_logit(x): # Not used yet
    return K.log(x/(1-x))

def sigmoid(x):
    return 1/(1+math.exp(-x)) # Watch out for an overflow error!

def data_generator(): # See (2).
    print("Calculating offset of", str(DATA_OFFSET) + ".")
    POS_EVALUATED = 0
    while(POS_EVALUATED < DATA_OFFSET):
        game = chess.pgn.read_game(pgn_file)
        for move in game.main_line():
            POS_EVALUATED += 1
        sys.stdout.write("\r" + str(round((POS_EVALUATED/DATA_OFFSET)*100,3)) + "% generated.")
        sys.stdout.flush()

    positions = []
    while(True):
        while(len(positions) < MAX_GENERATOR_SIZE):
            sys.stdout.write("\r" + str(round((len(positions)/MAX_GENERATOR_SIZE)*100,3)) + "% generated.")
            sys.stdout.flush()
            game = chess.pgn.read_game(pgn_file)
            board = game.board()
            for move in game.main_line():
                board.push(move)

                # Here, we have to convert the chess.Board() object into a 768-long array.
                position = []
                for square in chess.SQUARES: # len(chess.SQUARES) == range(0,64) -- more descriptive
                    try:
                        piece_type = board.piece_at(square).symbol()
                    except AttributeError: # piece.symbol() doesn't exist for None type.
                        piece_type = None
                    one = PIECE_MAP[piece_type]
                    one_hot = [0 for _ in range(0,12)]
                    if(one != -1):
                        one_hot[one] = 1
                    position.append(one_hot)
                flattened_position = []
                for oh in position:
                    for v in oh:
                        flattened_position.append(v) # Flatten the 64*12 matrix into a 1-D array (size 768).
                positions.append((flattened_position, board))
                POS_EVALUATED += 1

        data = positions[:MAX_GENERATOR_SIZE]
        labels = [sigmoid(evalp.evaluate(pos[0], pos[1])) for pos in data] # Use a sigmoid function for data regularization.
        data = [pos[0] for pos in data]
        yield data, labels, POS_EVALUATED
        positions = []

if(PREVIOUS_MODEL == ""):
    model = Sequential([
        Dense(8*8*12, input_shape=(8*8*12,)), # 8*8 for the positions, 12 for the one-hot array
        Dropout(0.5),
        Dense(8*8),
        Dropout(0.5),
        Dense(8),
        Dropout(0.5),
        Dense(1), # Should return a scalar value for the score.
        Activation("sigmoid")
    ])

sgd = SGD(lr=LEARNING_RATE, decay=DECAY, momentum=MOMENTUM, nesterov=USE_NESTEROV)

model.compile(optimizer=sgd, loss=LOSS_TYPE)

# Generate training data from PGN file in config.
generator = data_generator()

for _ in range(0,DATA_UPDATES):
    data, labels, POS_EVALUATED = generator.__next__()
    score = model.evaluate(data, labels, batch_size=BATCH_SIZE) # Loss, accuracy (see (1))
    print(score) # Test the model before training to test for overfitting.
    # Generate labels using these positions and a basic evaluation function.
    # See (1)
    # Fit the model!
    print(np.array(data).shape, "->", np.array(labels).shape)
    timer = time.time()
    model.fit(data, labels, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.2)
    TIME_SPENT += time.time()-timer
    TIME_SPENT = round(TIME_SPENT)
    tnc = name_components[:]

    for i in range(0,len(tnc)):
        if(tnc[i][0] == "$"):
            tnc[i] = globals()[tnc[i][1:]]
    model.save("../models/" + "__".join([str(comp) for comp in tnc]))
    print("Saved model to file.")
