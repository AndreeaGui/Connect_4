from agents.common import PlayerAction, BoardPiece, SavedState, GenMove, PLAYER1, PLAYER2, NO_PLAYER, GameState
from agents.common import connected_four, apply_player_action, pretty_print_board, check_end_state
from agents.common import generate_main_diagonals, generate_second_diagnals
import numpy as np
from typing import Optional, Callable, Tuple
import math

POSITIVE_INF = math.inf
NEGATIVE_INF = -math.inf


def find_opponent(player: BoardPiece) -> BoardPiece:
    """
    This method returns the opponent for player.
    :param player: the player we want to find the opponent for
    :return: the opponent
    """
    if player == PLAYER1:
        return PLAYER2
    return PLAYER1


def compute_score(board: np.ndarray, player: BoardPiece) -> float:
    """
    This method assigns the score needed in minimax to each board state given as input.
    The scores are between 100 (for winning) and -100 (for loosing)
    :param board: the board state that needs computing the score
    :param player: the player for whom is the score computed
    :return: the score, an int
    """
    if connected_four(board, player):
        return 100

    opponent = find_opponent(player)
    if connected_four(board, opponent):
        return -100

    return 0


def find_line_score(line: np.ndarray, player: BoardPiece) -> int:
    line_score = 0
    for first_element in range(0, len(line) - 3):
        possible_pattern = line[first_element:first_element + 4]
        counts = np.zeros(3)
        counts[0] = np.sum([possible_pattern == PLAYER1])
        counts[1] = np.sum([possible_pattern == PLAYER2])
        counts[2] = np.sum([possible_pattern == NO_PLAYER])
        assert (np.sum(counts)) == 4
        if player == PLAYER1:
            if np.all(counts == np.array([4, 0, 0])):
                line_score += 100
            elif np.all(counts == np.array([3, 0, 1])):
                line_score += 50
            elif np.all(counts == np.array([2, 0, 2])):
                line_score += 10
            elif np.all(counts == np.array([1, 0, 3])):
                line_score += 1
            elif np.all(counts == np.array([0, 4, 0])):
                line_score += -100
            elif np.all(counts == np.array([0, 3, 1])):
                line_score += -50
            elif np.all(counts == np.array([0, 2, 2])):
                line_score += -10
            elif np.all(counts == np.array([0, 1, 3])):
                line_score += -1
        else:
            if np.all(counts == np.array([4, 0, 0])):
                line_score += -100
            elif np.all(counts == np.array([3, 0, 1])):
                line_score += -50
            elif np.all(counts == np.array([2, 0, 2])):
                line_score += -10
            elif np.all(counts == np.array([1, 0, 3])):
                line_score += -1
            elif np.all(counts == np.array([0, 4, 0])):
                line_score += 100
            elif np.all(counts == np.array([0, 3, 1])):
                line_score += 50
            elif np.all(counts == np.array([0, 2, 2])):
                line_score += 10
            elif np.all(counts == np.array([0, 1, 3])):
                line_score += 1
    return line_score


def compute_score_2(board: np.ndarray, player: BoardPiece) -> int:
    """
    This method assigns the score needed in minimax to each board state given as input.
    The scores are between 100 (for winning) and -100 (for loosing)
    :param board: the board state that needs computing the score
    :param player: the player for whom is the score computed
    :return: the score, an int
    """

    score = 0
    for i in range(5):
        score += find_line_score(board[i, :], player)
    for i in range(6):
        score += find_line_score(board[:, i], player)
    main_diagonals = generate_main_diagonals(board)
    for d in main_diagonals:
        score += find_line_score(np.ndarray(d), player)
    second_diagonals = generate_second_diagnals(board)
    for d in second_diagonals:
        score += find_line_score(np.ndarray(d), player)

    return score


def generate_board_tree(board: np.array, player: BoardPiece):
    """
    The input board is the current state of the game.
    The child nodes returned are possible boards in the future, depending on the selected action.
    The first level is 1 step into the future, 2nd level 2 steps, etc. This method goes 4 levels depth.

    :param board: the root board from which the minimax algorithm starts.
    :param player: the current player
    :return:
    """

    level1 = []
    level2 = []
    level3 = []
    # 1st move - done by the current player
    for move1 in range(7):
        level1.append(apply_player_action(board, np.int8(move1), player))
    # 2nd move - done by the opponent
    for board1 in level1:
        for move2 in range(7):
            level2.append(apply_player_action(board1, np.int8(move2), find_opponent(player)))
    # 3rd move - done by the opponent
    for board2 in level2:
        for move3 in range(7):
            level3.append(apply_player_action(board2, np.int8(move3), player))

    print(len(level1))
    print(len(level2))
    print(len(level3))


def generate_child_boards(board: np.array, player: BoardPiece) -> [np.array]:
    """
    This method creates the children of the current root board.
    :param board: the root boards
    :param player: the current player
    :return: a list of 7 child boards
    """
    children_boards = []

    for move in range(7):
        board_copy = board.copy()
        children_boards.append(apply_player_action(board_copy, np.int8(move), player))

    return children_boards


def propagate_min(boards: [np.array], player: BoardPiece) -> (int, int):
    scores = [compute_score(b, player) for b in boards]
    return np.min(scores), np.argmin(scores)


def propagate_max(boards: [np.array], player: BoardPiece) -> (int, int):
    scores = [compute_score(b, player) for b in boards]
    return np.max(scores), np.argmax(scores)


# def generate_move_minimax(board: np.array, root_player: BoardPiece, current_player: BoardPiece, depth=2) -> (int, int):
#     """
#     The function implementing the minimax algorithm.
#     :param board: the current board
#     :param root_player: the player that makes the move on the root board
#     :param current_player: the player who makes the move on the current board
#     :param depth: the number of moves in the future there are left to compute
#     :return: a tuple with the score and the action move for obtaining that score
#     """
#
#     # TODO: define this method recurrently such that I have dynamical nr of levels
#     # i can use PLAYER%2 to see which player is now at this level
#     # i compare the level player to the original player and based on that I decide if i have a max or a min
#     # i return 2 things: both the min/max value and the column played
#
#     if depth == 0:
#         children = generate_child_boards(board, current_player)
#         final_score = 0
#         next_move = -1
#         if root_player == current_player:
#             final_score, next_move = propagate_max(children, current_player)
#         else:
#             final_score, next_move = propagate_min(children, current_player)
#
#         return final_score, next_move
#
#     children = generate_child_boards(board, current_player)
#     final_scores = np.zeros(7)
#     next_moves = np.zeros(7)
#     for i in range(len(children)):
#         final_scores[i], next_moves[i] = generate_move_minimax(children[i], root_player, find_opponent(current_player),
#                                                                depth - 1)
#     # final_score = np.


def recursive_minimax(board, root_player, current_player, depth=2):
    if depth == 0:
        score = compute_score(board, root_player)
        # print(pretty_print_board(board))
        print("depth 0 score: %.1f" % score)
        return score

    children = generate_child_boards(board, current_player)
    scores = np.zeros(7)
    for i in range(len(children)):
        print("child board %d at depth %d" % (i, depth))
        print(pretty_print_board(children[i]))
        scores[i] = recursive_minimax(children[i], root_player, find_opponent(current_player), depth - 1)

    if current_player == root_player:
        final_score = np.max(scores)
    else:
        final_score = np.min(scores)

    print("depth %d score: %.1f" % (depth, final_score))

    return final_score


## the working method
# def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], depth=3
#                           ) -> Tuple[PlayerAction, Optional[SavedState]]:
#     children = generate_child_boards(board, player)
#     scores = np.zeros(7)
#     for i in range(len(children)):
#         scores[i] = recursive_minimax(children[i], player, find_opponent(player), depth - 1)
#     # final_score = np.max(scores)
#     next_move = np.argmax(scores)
#
#     return np.int8(next_move), saved_state


### the new method
def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], depth=3
                          ) -> Tuple[PlayerAction, Optional[SavedState]]:
    children = generate_child_boards(board, player)
    scores = np.zeros(7)

    for i in range(len(children)):
        scores[i] = minimax(children[i], player, find_opponent(player), depth - 1, NEGATIVE_INF, POSITIVE_INF)
    next_move = np.argmax(scores)

    return np.int8(next_move), saved_state


## the code implemented frome the pseudcode with alpha-beta pruning
def minimax(board: np.ndarray, root_player: BoardPiece, current_player: BoardPiece,
            depth: int = 4, alpha=NEGATIVE_INF, beta=POSITIVE_INF) -> float:
    if depth == 0 or check_end_state(board, current_player) != GameState.STILL_PLAYING:
        # score = compute_score(board, root_player)
        score = compute_score_2(board, root_player)
        return score

    children = generate_child_boards(board, current_player)

    if current_player == root_player:
        max_eval = NEGATIVE_INF
        for i in range(len(children)):
            score = minimax(children[i], root_player, find_opponent(current_player), depth - 1, alpha, beta)
            max_eval = np.maximum(max_eval, score)
            alpha = np.maximum(alpha, score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = POSITIVE_INF
        for i in range(len(children)):
            score = minimax(children[i], root_player, find_opponent(current_player), depth - 1, alpha, beta)
            min_eval = np.minimum(min_eval, score)
            beta = np.minimum(beta, score)
            if beta <= alpha:
                break
        return min_eval


b1 = np.empty((6, 7), dtype=BoardPiece)
b1.fill(NO_PLAYER)
b1[0, 1] = PLAYER2
b1[1, 1] = PLAYER2
b1[0, 2] = PLAYER2
b1[2, 2] = PLAYER2
b1[1, 3] = PLAYER2
b1[1, 4] = PLAYER2
b1[1, 2] = PLAYER1
b1[3, 2] = PLAYER1
b1[0, 3] = PLAYER1
b1[2, 3] = PLAYER1
b1[3, 3] = PLAYER1
b1[0, 4] = PLAYER1
b1[2, 4] = PLAYER1
# print(pretty_print_board(b1))
#
# children = generate_child_boards(b1, PLAYER1)
# for i in range(7):
#     print(pretty_print_board(children[i]))

# print(compute_score(b1, PLAYER2))

b2 = np.empty((6, 7), dtype=BoardPiece)
b2.fill(NO_PLAYER)
b2[0, 1] = PLAYER2
b2[1, 1] = PLAYER2
b2[0, 2] = PLAYER2
b2[2, 2] = PLAYER2
# b2[1, 3] = PLAYER2
# b2[1, 4] = PLAYER2
b2[1, 2] = PLAYER2
# b2[3, 2] = PLAYER1
# b2[0, 3] = PLAYER1
# b2[2, 3] = PLAYER1
# b2[3, 3] = PLAYER1
# b2[0, 4] = PLAYER1
# b2[2, 4] = PLAYER1
'''|==============|
|              |
|              |
|    X X       |
|    O X X     |
|  O O O O     |
|  O O X X     |
|==============|
|0 1 2 3 4 5 6 |'''
# print(pretty_print_board(b2))
# print(compute_score(b2, PLAYER1))
# print(compute_score(b2, PLAYER2))
#
# print(propagate_max([b1, b2], PLAYER2))

# recursive_minimax(b2, PLAYER2, PLAYER1, 2)
# print(generate_move_minimax(b2, PLAYER2, 3))

# ce = np.minimum(0.0, -10000)
# print(ce)
#
# minimax(b2, PLAYER1, PLAYER2, 2)
