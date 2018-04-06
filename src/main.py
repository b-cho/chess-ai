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

1.  This is Forsynth-Edwards Notation (FEN), used to represent chess positions.
    Its format is as follows:
        a. This part is used to denote the board layout. a number means that many empty spaces.
        b. This is "w" or "b" depending on whose turn it is.
        c. This denotes castling availability (KQ for White, kq for Black).
        d. This denotes "en passant" availability -- the space directly behind a moving pawn that moved two spaces at first move.
        e. This is a half-move (i.e., ply) clock -- this is the clock to count the half-moves since the last capture or pawn advance.
        f. The number of full moves played in the game (Black and White).

 2. A transposition table is a form of dynamic programming used to store positions for later use (effectively "memoization").
    It stores the evaluation results for certain nodes so that the computer doesn't have to reiterate over already computed nodes.
    However, because the number of positions in chess is at least 10^40, it is impossible to store all positions. Therefore, we use
    a kind of cache system to only store the most relevant node values.
    In this program, we use a FIFO-type OrderedDict to store the data in order so that it never exceeds the maximum size.

 3. In chess engines, a tree-searching algorithm is required to determine the best move.
    For this, we use the negamax algorith with alpha-beta pruning and a transposition table (see (4)).
    See https://wikipedia.org/wiki/Negamax for more information.

 4. We use Zobrist hashing to efficiently store already-evaluated positions so we don't have to re-evaluate already searched nodes.
    See (2) for more information. In addition, see https://wikipedia.org/wiki/Zobrist_hashing for more information.

 5. Quiescence search is used to mitigate the horizon effect, or the effect that you can't see over the "horizon" because the depth doesn't permit it.
    This only applies to depth-limited search algorithms such as the Negamax algorithm we are using in this program.
    It looks for noisy board states (those with captures, large movements, etc) and searches them deeper to find any challenges beyond the depth limit.

    "W" is white with layout 768I-64-8-1, "B" is black with DIV_SCORE = 100 LR = 1e-2 DEC = 5e-6 E/BS = 1000/10000 MM/NESTEROV = 0.9/True
    "W_N" is white with layout 768I-768-768-768-768-64-1 with DIV_SCORE = 100 LR = 1e-1 DEC = 5e-6 E/BS = 1000/10000 MM/NESTEROV = 0.9/True
    "W_S" is for the model trained by train_stockfish.py with DIV_SCORE = 100 LR = 1e-2 DEC = 5e-6 E/BS = 400/10000 VS = 20%  MM/NESTEROV = 0.9/True
    "W_R" is for the model trained by 768I-768-8-1 train_random.py with DIV_SCORE = 1000 LR = 1e-1 DEC = 5e-6 E/BS = 1000/10000 VS = 20%  MM/NESTEROV = 0.9/True
"""

import chess # For easy board manipulation
import itertools
import sys
import os
import math
import random
import time
import keras
import gc
import socket
import json
import tensorflow as tf
import numpy as np
import urllib.parse
import threading
from collections import OrderedDict
from keras.models import load_model
from keras import backend as K
from http.server import BaseHTTPRequestHandler, HTTPServer

### Configuration information ###
TRANSPOSITION_MAX_ELEMENTS = 1e8 # See (2)
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" # See (1)
NORMAL_DEPTH = 100 # In plies (half-moves)
ENDGAME_DEPTH = 100 # In plies (half-moves)
PLUS_ONE_ED = False # Add one ply for every piece less than 7
MAX_TIME = 30 # Seconds for max time
QUIESCENCE_DEPTH = float("inf") # See (5)
SORT_MOVES = False # Sort moves before negamax-ing?
INITIAL_ALPHA, INITIAL_BETA = -float("inf"), float("inf") # Initial alpha-beta values.
DIV_SCORE = 1 # Set to 1 for default, multiplies <score> variable in Searcher._evaluate().
AI_PLAYS = True # What side the AI plays -- True for white and False for black.

CPU_CORES = 4 # Set to number of CPU cores
NUM_CPU = 1 # Number of CPUs you want to use
NUM_GPU = 0 # Number of GPUs you want to use

HOST_IP = "127.0.0.1" # Server host IP
HOST_PORT = 19265 # Server host port

### General Definitions ###
CHECKMATE_SCORE = 1e6
evaluation_model = None # Defined under if(__name__ == "__main__") below
PIECE_MAP = {None:-1,"K":0,"Q":1,"R":2,"B":3,"N":4,"P":5,"k":6,"q":7,"r":8,"b":9,"n":10,"p":11}

table = np.random.randint(1, (2**63)-1, size=(64,12), dtype="int64") # Generate Zobrist hashing pseudo-random table and 64-bit long integers (see (4))

class ZobristHash(): # See (4)
    def __init__(self, depth, flag, value):
        self.depth = 0
        self.flag = 0
        self.value = 0

    @staticmethod
    def generateHash(board):
        h = 0
        for square in chess.SQUARES:
            if(board.piece_type_at(square) != None):
                j = board.piece_type_at(square)
                h = h ^ table[square][j]
        return h

class FIFOCache(): # See (2)
    def __init__(self, size):
        self.cache = OrderedDict()
        self.size = size

    def get(self, key, default=None):
        try:
            self.cache.move_to_end(key)
        except KeyError:
            return default
        return self.cache[key]

    def __setitem__(self, key, value, prune=True):
        try:
            del self.cache[key]
        except KeyError:
            while(len(self.cache) >= self.size):
                self.cache.popitem(last=False)
        self.cache[key] = value

class Searcher():
    def __init__(self):
        self.cache = FIFOCache(TRANSPOSITION_MAX_ELEMENTS)
        self.moves_evaluated = 0 # Just a counter.

    def _evaluate(self, board):
        if(board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition()):
            return -70 # Bad, but not really bad
        if(board.is_checkmate()):
            if(board.turn):
                return -CHECKMATE_SCORE
            else:
                return CHECKMATE_SCORE

        score = evaluation_model.predict(flatten_board(board))
        return logit(score[0][0])*DIV_SCORE # We need a scalar integer.

    def _is_quiet(self, board, move=None):
        original_move = move
        if(move == None):
            move = board.pop()

        is_quiet = [
            #board.is_check(), # Check if the board is in check -- thus it is not quiet. WARNING FOR INFINITE RECURSION!
            board.is_capture(move) # Check if the last move was a capture.
        ]

        if(original_move == None):
            board.push(move)

        return not any(is_quiet) # If any of the above are true, it is not quiet.

    def _quiescence_search(self, board, depth, alpha, beta, player): # This is quiescence search using negamax with alpha-beta pruning. (see (5))
        if(self._is_quiet(board) or board.legal_moves.count() <= 0 or depth <= 0):
            if(player == True):
                return -self._evaluate(board) # Swapped on 3/31/2018
            else:
                return self._evaluate(board)

        else:
            best_value = -CHECKMATE_SCORE
            best_move = chess.Move.null() # Zugzwang?
            for move in board.legal_moves:
                if(self._is_quiet(board, move)):
                    self.moves_evaluated += 1
                    board.push(move)
                    value = -self._quiescence_search(board, depth-1, -beta, -alpha, not player)
                    if(value >= 0.99*CHECKMATE_SCORE):
                        value -= 10
                    if(value <= -0.99*CHECKMATE_SCORE):
                        value += 10
                    board.pop()
                    prev_best_value = best_value
                    best_value = max(value, best_value)
                    alpha = max(alpha, value)
                    if(alpha >= beta):
                        break

            return best_value

    def _negamax(self, board, depth, alpha, beta, player, orig_call=True): # Use Negamax with alpha-beta pruning and transposition table (see (3))
        if(depth <= 0):
            self.moves_evaluated += 1
            return -self._quiescence_search(board, QUIESCENCE_DEPTH, -beta, -alpha, player)

        # Neither of the above statements are true, so we can check moves.
        original_alpha = alpha

        board_hash = ZobristHash.generateHash(board)
        if(not orig_call): # ERROR! Watch for #001
            transposition_entry = self.cache.get(board_hash)
            if(transposition_entry != None and transposition_entry.depth >= depth):
                if(transposition_entry.flag == "EXACT"):
                    return transposition_entry.value
                elif(transposition_entry.flag == "LOWERBOUND"):
                    alpha = max(alpha, transposition_entry.value)
                elif(transposition_entry.flag == "UPPERBOUND"):
                    beta = min(beta, transposition_entry.value)
                if(alpha >= beta):
                    return transposition_entry.value

        if(SORT_MOVES == True):
            moves = []
            for m in board.legal_moves:
                new_board = board.copy()
                new_board.push(m)
                moves.append((board.is_capture(m), (2*board.turn+1)*self._evaluate(new_board), m))

            moves.sort(key=lambda x: x[1]) # potentially O(2N^2), which could be too slow. (NOT as of 3/21/18)
            moves.sort(key=lambda y: y[0]) # Is actually O(2*N log N) because Timsort
            moves = [m[2] for m in moves]
        else:
            moves = [m for m in board.legal_moves]

        best_value = -CHECKMATE_SCORE
        best_move = chess.Move.null() # Zugzwang?

        for mv in range(0,len(moves)):
            self.moves_evaluated += 1
            board.push(moves[mv])
            if(mv == 0):
                try:
                    value = -self._negamax(board, depth-1, -beta, -alpha, not player, False)[0]
                except: # Usually TypeError
                    value = -self._negamax(board, depth-1, -beta, -alpha, not player, False)
            else:
                try:
                    value = -self._negamax(board, depth-1, -alpha-1, -alpha, not player, False)[0]
                except: # Usually TypeError
                    value = -self._negamax(board, depth-1, -alpha-1, -alpha, not player, False)
                if(alpha < value and value < beta):
                    try:
                        value = -self._negamax(board, depth-1, -beta, -alpha, not player, False)[0]
                    except: # Usually TypeError
                        value = -self._negamax(board, depth-1, -beta, -alpha, not player, False)

            if(value >= 0.99*CHECKMATE_SCORE):
                value -= 10
            if(value <= -0.99*CHECKMATE_SCORE):
                value += 10

            board.pop()
            prev_best_value = best_value
            best_value = max(value, best_value)
            if(prev_best_value < best_value):
                best_move = moves[mv]
            alpha = max(alpha, value)
            if(alpha >= beta):
                break

            #print(depth, best_move, value)

        new_transposition_entry = ZobristHash(None, None, None) # We'll initialize these values below.
        new_transposition_entry.value = best_value
        if(best_value <= original_alpha):
            new_transposition_entry.flag = "UPPERBOUND"
        elif(best_value >= beta):
            new_transposition_entry.flag = "LOWERBOUND"
        else:
            new_transposition_entry.flag = "EXACT"
        new_transposition_entry.depth = depth
        self.cache[board_hash] = new_transposition_entry

        return best_value, best_move

searcher = Searcher()

def meta_search(m, b):
    for d in range(1, m+1):
        print("Searching depth", d, "of", m)
        s = searcher._negamax(b, d, INITIAL_ALPHA, INITIAL_BETA, b.turn)
        best_value, best_move = s[0], s[1]
        yield best_value, best_move

class NetworkServer(BaseHTTPRequestHandler):
    def do_POST(self): # LOOP
        global searcher

        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        print(post_data)
        post_data = urllib.parse.unquote(str(post_data)).replace("+", " ")
        post_data = post_data[8:len(post_data)-1]
        print(post_data)

        board = chess.Board()
        board.set_fen(post_data)

        ttm = 0
        if(len(board.piece_map()) > 7): # Search deeper on average for end-game
            ms = meta_search(NORMAL_DEPTH, board)
            while(True):
                try:
                    int_time = time.time()
                    best_value, best_move = ms.__next__()
                    ttm += (time.time()-int_time)
                    print("Searched in", (time.time()-int_time), "seconds.", MAX_TIME-ttm, "seconds left.")
                    if((time.time()-int_time)*12 > MAX_TIME-ttm):
                        break
                except StopIteration:
                    break
        else:
            if(PLUS_ONE_ED):
                ms = meta_search(ENDGAME_DEPTH+7-len(board.piece_map()), board)
                while(True):
                    try:
                        int_time = time.time()
                        best_value, best_move = ms.__next__()
                        ttm += (time.time()-int_time)
                        print("Searched in", (time.time()-int_time), "seconds.", MAX_TIME-ttm, "seconds left.")
                        if((time.time()-int_time)*20 > MAX_TIME-ttm):
                            break
                    except StopIteration:
                        break
            else:
                ms = meta_search(ENDGAME_DEPTH, board)
                while(True):
                    try:
                        int_time = time.time()
                        best_value, best_move = ms.__next__()
                        ttm += (time.time()-int_time)
                        print("Searched in", (time.time()-int_time), "seconds.", MAX_TIME-ttm, "seconds left.")
                        if((time.time()-int_time)*20 > MAX_TIME-ttm):
                            break
                    except StopIteration:
                        break

        #Previous error here?

        best_value = str(best_value)
        best_move = board.san(best_move)
        board.push_san(best_move)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b"{\"value\":"+bytes(best_value, "utf-8")+b",\"move\":\""+bytes(best_move, "utf-8")+b"\"}")
        return


def flatten_board(board):
    # Here, we have to convert the chess.Board() object into a 768-long array.
    pieces = board.piece_map()
    position = np.zeros(shape=(64,12))
    for square, piece in pieces.items(): # Position is a dictionary, so we get (key, value) pairs.
        piece = PIECE_MAP[piece.symbol()]
        position[square][piece] = 1

    return np.array([position.flatten()]) # Flatten the 64*12 matrix into a 1-D array (size 768).

def logit(x):
    try:
        return math.log(x) - math.log(1-x)
    except ValueError: # log(0) isn't defined.
        return math.log1p(x) - math.log1p(1-x) # Maybe return -inf?

def execute(filepath=None):
    global evaluation_model
    global INITIAL_FEN

    config = tf.ConfigProto(intra_op_parallelism_threads=CPU_CORES,\
            inter_op_parallelism_threads=CPU_CORES, allow_soft_placement=True,\
            device_count = {'CPU' : NUM_CPU, 'GPU' : NUM_GPU})
    session = tf.Session(config=config)
    K.set_session(session)

    try:
        evaluation_model = load_model(sys.argv[1])
    except:
        if(filepath == None):
            evaluation_model = load_model(input("Evaluation model filepath: "))
        else:
            evaluation_model = load_model(filepath)

    server_address = ('127.0.0.1', 8081)
    httpd = HTTPServer(server_address, NetworkServer)
    print('Running server...')
    httpd.serve_forever()
    return

    NEW_FEN = input("Input an initial FEN (leave blank for standard start): ")
    if(NEW_FEN != ""):
        INITIAL_FEN = NEW_FEN
    board = chess.Board(INITIAL_FEN)

    #import cProfile #DEBUG
    #cProfile.run("Searcher()._negamax(chess.Board(INITIAL_FEN), NORMAL_DEPTH, INITIAL_ALPHA, INITIAL_BETA, True)") #DEBUG
    print("=--------------------------=")
    print(str(board))
    print("Initial State")
    while(True):
        temp_board = chess.Board() # For SAN notation print-out
        if(board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition()):
            print("Draw.")
            break
        elif(board.turn == AI_PLAYS):
            if(board.is_checkmate()):
                print("Black wins.")
                break

            start = time.time() # TIME
            searcher.moves_evaluated = 0
            ttm = 0
            if(len(board.piece_map()) > 7): # Search deeper on average for end-game
                ms = meta_search(NORMAL_DEPTH, board)
                while(True):
                    try:
                        int_time = time.time()
                        best_value, best_move = ms.__next__()
                        ttm += (time.time()-int_time)
                        if((time.time()-int_time)*20 > MAX_TIME-ttm):
                            break
                        d_s += 1
                    except:
                        break
            else:
                if(PLUS_ONE_ED):
                    ms = meta_search(ENDGAME_DEPTH+7-len(board.piece_map()), board)
                    while(True):
                        try:
                            int_time = time.time()
                            best_value, best_move = ms.__next__()
                            ttm += (time.time()-int_time)
                            if((time.time()-int_time)*20 > MAX_TIME-ttm):
                                break
                            d_s += 1
                        except:
                            break
                else:
                    ms = meta_search(ENDGAME_DEPTH, board)
                    while(True):
                        try:
                            int_time = time.time()
                            best_value, best_move = ms.__next__()
                            ttm += (time.time()-int_time)
                            if((time.time()-int_time)*20 > MAX_TIME-ttm):
                                break
                            d_s += 1
                        except:
                            break

            board.push(best_move)
            print("=--------------------------=")
            print(str(board))
            print(str(time.time()-start), "seconds searching.") # TIME
            if(best_value >= 0.99*CHECKMATE_SCORE): # Mate found in 1000 moves
                print("M" + str(int((CHECKMATE_SCORE-best_value)/20)+1), "was the best value.")
            else:
                print(str(best_value), "was the best value.")
            print(str(best_move), "was the best move.")
            print(str(searcher.moves_evaluated), "moves evaluated.")
            print(str(len(searcher.cache.cache)), "is the TT size.")
            #print(temp_board.variation_san([chess.Move.from_uci(m) for m in [k.uci() for k in board.move_stack]]))
            print("=--------------------------=")
            gc.collect()

        elif(board.turn != AI_PLAYS):
            if(board.is_checkmate()):
                print("White wins.")
                break

            while(True):
                move = input("Move: ")
                try:
                    move = chess.Move.from_uci(move)
                    if(move in board.legal_moves):
                        board.push(move)
                    else:
                        raise ValueError
                    break
                except ValueError:
                    print("Invalid move.")
                    continue
            print("=--------------------------=")
            print(str(board))
            #print(str(time.time()-start), "seconds searching.") # TIME
            #print(str(best_value), "was the best value.")
            #print(str(best_move), "was the best move.")
            #print(str(searcher.moves_evaluated), "moves evaluated.")
            #print(str(len(searcher.cache.cache)), "is the TT size.")
            #print(temp_board.variation_san([chess.Move.from_uci(m) for m in [k.uci() for k in board.move_stack]]))
            print("=--------------------------=")
            gc.collect()

if(__name__ == "__main__"):
    try:
        execute()
    except KeyboardInterrupt:
        sys.exit(0) # Exit safely.
