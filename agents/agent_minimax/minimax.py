from agents.common import PlayerAction, BoardPiece, SavedState, GenMove, PLAYER1, PLAYER2, NO_PLAYER
from agents.common import connected_four, apply_player_action, pretty_print_board
import numpy as np


def find_opponent(player: BoardPiece) -> BoardPiece:
    """
    This method returns the opponent for player.
    :param player: the player we want to find the opponent for
    :return: the opponent
    """
    if player == PLAYER1:
        return PLAYER2
    return PLAYER1


def compute_score(board: np.ndarray, player: BoardPiece) -> int:
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


def propagate_min(boards: [np.array], player: BoardPiece) -> int:
    scores = [compute_score(b, player) for b in boards]
    return np.min(scores)


def propagate_max(boards: [np.array], player: BoardPiece) -> int:
    scores = [compute_score(b, player) for b in boards]
    return np.max(scores)


def generate_move_minimax(board: np.array, player: BoardPiece, depth=2) -> (int, int):
    # TODO: define this method recurrently such that I have dynamical nr of levels
    # i can use PLAYER%2 to see which player is now at this level
    # i compare the level player to the original player and based on that I decide if i have a max or a min
    # i return 2 things: both the min/max value and the column played
    if depth == 0:
        children = generate_child_boards(board, player)
        compute_score(children)

    # generate_move_minimax(b, player, depth-1)


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

print(compute_score(b1, PLAYER2))

b2 = np.empty((6, 7), dtype=BoardPiece)
b2.fill(NO_PLAYER)
b2[0, 1] = PLAYER2
b2[1, 1] = PLAYER2
b2[0, 2] = PLAYER2
b2[2, 2] = PLAYER2
b2[1, 3] = PLAYER2
b2[1, 4] = PLAYER2
b2[1, 2] = PLAYER2
b2[3, 2] = PLAYER1
b2[0, 3] = PLAYER1
b2[2, 3] = PLAYER1
b2[3, 3] = PLAYER1
b2[0, 4] = PLAYER1
b2[2, 4] = PLAYER1
'''|==============|
|              |
|              |
|    X X       |
|    O X X     |
|  O O O O     |
|  O O X X     |
|==============|
|0 1 2 3 4 5 6 |'''
print(pretty_print_board(b2))
print(compute_score(b2, PLAYER1))
print(compute_score(b2, PLAYER2))

print(propagate_max([b1, b2], PLAYER2))
