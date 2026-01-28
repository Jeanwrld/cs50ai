"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    Takes a state and tells us whose turn it is.
    If its x first move then player function returns x. 
    If board has a move by x then it returns O.
    """
    countX = 0
    countO = 0

    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == X:
                countX += 1
            if board[row][col] == O:
                countO += 1

    if countX > countO:
        return O
    else:
        return X
    
def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    allPossibleActions=set()

    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == EMPTY:
                allPossibleActions.add((row,col))

    return allPossibleActions



def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if action not in actions(board):
        raise Exception("Not valid action")
    
    row, col = action
    board_copy = copy.deepcopy(board)
    board_copy[row][col] = player(board)

    return board_copy

def check_row(board, player):
    for row in board:
        if row[0] == row[1] == row[2] == player:
            return True
        return False
    
def check_col(board, player):
    for col in range(len(board)):
        if board[col][0] == player and board[col][1] == player and board[col][2] == player:
            return True
    return False
    
'''def checkFirstDig(board, player):
    count = 0
    for row in range(len(board)):
        for col in range(len(board[row])):
            if row == col and board[row][col] == player:
                count += 1

    if count == 3:
        return True
    else:
        return False
    
def checkSecondDig(board, player):
    count = 0
    for row in range(len(board)):
        for col in range(len(board[row])):
            pass

    if count == 3:
        return True
    else:
        return False'''
def check_diag(board, player):
    return (
        board[0][0] == board[1][1] == board[2][2] == player or
        board[0][2] == board[1][1] == board[2][0] == player
    )


def winner(board):
    if check_row(board, X) or check_col(board, X) or check_diag(board, X):
        return X
    if check_row(board, O) or check_col(board, O) or check_diag(board, O):
        return O
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    for row in board:
        if EMPTY in row:
            return False
    return True
    


def utility(board):
    #final position
    w = winner(board)
    if w == X:
        return 1
    if w == O:
        return -1
    return 0



def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    If game is not over, eg Os turn by PLAYER(s) = O.(O tries to minimize)
    considers what actions and next states.
    Recursively call itself in X shoes,and make a decision where the oppponent has either 0 or -1

    
    def max_value(state):
        if terminal(board):
            return utility(board)
        #value = -math.inf# as low as possible
        value = float('-inf')
        for action in actions(board):
            value = max(value, min_value(result(state, action)))
        return value
    
    def min_value(state):
        if terminal(board) == True:
            return utility()
        value = math.inf
        for action in actions(board):
            value = min(value, max_value(result(state, action)))

        return value"""
    
    

    if terminal(board):
        return None

    current_player = player(board)

    def max_value(state):
        if terminal(state):
            return utility(state)

        value = float('-inf')
        for action in actions(state):
            value = max(value, min_value(result(state, action)))
        return value

    def min_value(state):
        if terminal(state):
            return utility(state)

        value = float('inf')
        for action in actions(state):
            value = min(value, max_value(result(state, action)))
        return value

    best_action = None

    if current_player == X:
        best_score = float('-inf')
        for action in actions(board):
            score = min_value(result(board, action))
            if score > best_score:
                best_score = score
                best_action = action
    else:
        best_score = float('inf')
        for action in actions(board):
            score = max_value(result(board, action))
            if score < best_score:
                best_score = score
                best_action = action

    return best_action

    
