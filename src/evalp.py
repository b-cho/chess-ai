import pystockfish
# Configuration
DIV_SCORE = 100 # A DIV_SCORE of 10 works on an ASUS G752VS-XB72K without an OverflowError
# Remember to change this!
# Normally 100

PLAYER_MODIFIER = 4
OPPONENT_MODIFIER = 4
piece_map = {0:"K",1:"Q",2:"R",3:"B",4:"N",5:"P",6:"K",7:"Q",8:"R",9:"B",10:"N",11:"P"}
pieces = {None: 0, 'P': 100*PLAYER_MODIFIER, 'N': 280*PLAYER_MODIFIER, 'B': 320*PLAYER_MODIFIER, 'R': 479*PLAYER_MODIFIER, 'Q': 929*PLAYER_MODIFIER, 'K': 30000}
opp_pieces = {None: 0, 'P': 100*OPPONENT_MODIFIER, 'N': 280*OPPONENT_MODIFIER, 'B': 320*OPPONENT_MODIFIER, 'R': 479*OPPONENT_MODIFIER, 'Q': 929*OPPONENT_MODIFIER, 'K': 30000}
pst = {
    'Pw': (   0,   0,   0,   0,   0,   0,   0,  0,
            78,  83,  86,  73, 102,  82,  85,  90,
             7,  29,  21,  44,  40,  31,  44,   7,
           -17,  16,  -2,  15,  14,   0,  15, -13,
           -26,   3,  15,   21,  22,   6,   2, -23,
           -22,   9,   5, -11, -10,  -2,   3, -19,
           -31,   8,  -7, -37, -36, -14,   3, -31,
             0,   0,   0,   0,   0,   0,   0,   0),
    'Nw': ( -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6, 100, -36,   4,  62,  -4, -14,
            10,  67,   1,  74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,   5,  31,  21,  22,  35,   2,   0,
           -18,  10,  13,  22,  18,  15,  11, -14,
           -23, -15,   2,   0,   2,   0, -23, -20,
           -74, -23, -26, -24, -19, -35, -22, -69),
    'Bw': ( -59, -78, -82, -76, -23,-107, -37, -50,
           -11,  20,  35, -42, -39,  31,   2, -22,
            -9,  39, -32,  41,  52, -10,  28, -14,
            25,  17,  20,  34,  26,  25,  15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,   8,  25,  20,  15,
            19,  20,  11,   6,   7,   6,  20,  16,
            -7,   2, -15, -12, -14, -15, -10, -10),
    'Rw': (  35,  29,  33,   4,  37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
             0,   5,  16,  13,  18,  -4,  -9,  -6,
           -28, -35, -16, -21, -13, -29, -46, -30,
           -42, -28, -42, -25, -25, -35, -26, -46,
           -53, -38, -31, -26, -29, -43, -44, -53,
           -30, -32, -21,   5,  -2, -21, -32, -32), # Rooks should not make pointless moves
    'Qw': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -10,  20,  76,  57,  24,
            -2,  43,  32,  60,  72,  63,  43,   2,
             1, -16,  22,  17,  25,  20, -13,  -6,
           -14, -15,  -2,  -5,  -1, -10, -20, -22,
           -30,  -6, -13, -11, -16, -11, -16, -27,
           -36, -18,   0, -19, -15, -15, -21, -38,
           -39, -30, -31, -13, -31, -36, -34, -42),
    'qw': (   6,   1,  -8,-104,  69,  24,  88,  26,
            14,  32,  60, -60,  20,  76,  57,  24,
            -2,  43,  32, -60,  72,  63,  43,   2,
             1, -16,  22, -60,  25,  20, -13,  -90,
           -60, -15,  -2, -60,  -1, -10, -80, -22,
           -30, -60, -13, -60, -16, -60, -16, -27,
           -36, -18, -60, -60, -60, -15, -21, -38,
           -60, -60, -60, -13, -60, -60, -60, -60),
    'Kw': (   4,  54,  47, -99, -99,  60,  83, -62, # Uppercase for midgame
           -32,  10,  55,  56,  56,  55,  10,   3,
           -62,  12, -57,  44, -67,  28,  37, -31,
           -55,  50,  11,  -4, -19,  13,   0, -49,
           -55, -43, -52, -28, -51, -47,  -8, -50,
           -47, -42, -43, -79, -64, -32, -29, -32,
            -4,   3, -14, -50, -57, -18,  13,   4,
            17,  30,  -3, -14,   6,  -1,  40,  18),
    'kw': (-50, -40, -30, -20, -20, -30, -40, -50, # Lowercase for endgame
           -30, -20, -10,   0,   0, -10, -20, -30,
           -30, -10,  20,  30,  30,  20, -10, -30,
           -30, -10,  30,  40,  40,  30, -10, -30,
           -30, -10,  30,  40,  40,  30, -10, -30,
           -30, -10,  20,  30,  30,  20, -10, -30,
           -30, -30,   0,   0,   0,   0, -30, -30,
           -50, -30, -30, -30, -30, -30, -30, -50)
}

for pt in "KkQRBNP":
    t = []
    for i in range(0,64):
        t.append(pst[pt + "w"][63-i]%63)
    pst[pt + "b"] = t

def evaluate(board, board_obj): # This is a 8*8*12-long array converted into a scalar using PSTs and piece values.
    score = 0 # Initialize score
    num_pieces = {"K":0,"Q":0,"R":0,"B":0,"N":0,"P":0,"k":0,"q":0,"r":0,"b":0,"n":0,"p":0}

    legal_moves = [(move.from_square, move.to_square) for move in board_obj.legal_moves]
    legal_moves = sorted(legal_moves)

    for i in range(0,64): # For every piece, (each one-hot is of length 12)
        # We have to start at bottom left (i.e., A1), so we convert the positions.
        square = 63-i-(7-(2*i)%8)
        piece_one_hot = board[i*12:(i+1)*12] # len(piece_one_hot) == 12
        piece_type = None
        wb_mod = None
        wb = None
        for j in range(0,len(piece_one_hot)):
            if(piece_one_hot[j] == 1):
                piece_type = piece_map[j]
                wb_mod = j <= 5 # True is White, False is Black
                if(wb_mod):
                    wb = "w"
                    num_pieces[piece_type.upper()] += 1
                else:
                    wb = "b"
                    num_pieces[piece_type.lower()] += 1

        if(wb_mod == True):
            wb_mod = 1
        else:
            wb_mod = -1

        # >>>>>> Piece Value <<<<<< #
        if(wb == "w"):
            score += pieces[piece_type]*wb_mod # Add PIECE VALUE to score.
        else:
            score += opp_pieces[piece_type]*wb_mod # Add PIECE VALUE to score.

        # >>>>>> Piece Positioning <<<<<< #
        if(piece_type != None and piece_type != "K"): # Add PIECE POSITION to score.
            score += pst[piece_type + wb][square]*wb_mod
        elif(piece_type == "K"):
            # Determine if early/mid or end-game. Check if 7 or less "1"s exist.
            ones = 0
            for v in board:
                if(v == 1):
                    ones += 1
            if(ones <= 7):
                score += pst[piece_type.lower()+wb][square]*wb_mod
            else:
                score += pst[piece_type.upper()+wb][square]*wb_mod

        elif(piece_type == "K"):
            # Determine if early/mid or end-game. Check if 7 or less "1"s exist.
            ones = 0
            for v in board:
                if(v == 1):
                    ones += 1
            if(ones >=28):
                score += pst[piece_type.lower()+wb][square]*wb_mod
            else:
                score += pst[piece_type.upper()+wb][square]*wb_mod

        # >>>>>> Special Piece Modifications <<<<<< #
        if(piece_type == "R"): # Less pawns is better (cumulative) so we add more for # of pawns below.
            score += (16-num_pieces["P"]-num_pieces["p"])*10*wb_mod

        if(piece_type == "N"): # More pawns is better (cumulative) so we subtract more for # of pawns below.
            score -= (16-num_pieces["P"]-num_pieces["p"])*10*wb_mod

        # >>>>>> Mobility <<<<<< #
        for move in legal_moves:
            if(move[0] == i):
                score += 10*wb_mod

    return score/DIV_SCORE
