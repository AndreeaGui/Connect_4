from enum import Enum
from typing import Optional
import numpy as np

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiece_Print = str  # dtype for string representation of BoardPiece
NO_PLAYER_Print = str(' ')
PLAYER1_Print = str('X')
PLAYER2_Print = str('O')

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1  # remiza
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.zeros((6, 7), BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """

    board_str = "|==============|\n"

    for i in range(5, -1, -1):
        board_str = board_str + '|'
        for j in range(7):
            if board[i, j] == PLAYER1:
                board_str = board_str + PLAYER1_Print + ' '
            elif board[i, j] == PLAYER2:
                board_str = board_str + PLAYER2_Print + ' '
            else:
                board_str = board_str + NO_PLAYER_Print + ' '
        board_str = board_str + '|\n'
    board_str = board_str + "|==============|\n"
    board_str = board_str + "|0 1 2 3 4 5 6 |"

    return board_str


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """

    start_index = pp_board.find('=|\n') + len('=|\n')
    elements_nr = 6 * (2 * 7 + 2)
    board_str = pp_board[start_index: start_index + elements_nr + 4]
    board_str = board_str.replace("|", "")
    board_str = board_str.replace("\n", "")

    board = initialize_game_state()
    string_index = 0
    for i in range(5, -1, -1):
        for j in range(7):
            if board_str[string_index] == PLAYER1_Print:
                board[i, j] = PLAYER1
            elif board_str[string_index] == PLAYER2_Print:
                board[i, j] = PLAYER2
            else:
                board[i, j] = NO_PLAYER
            string_index += 2

    return board


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    if np.int(action) < 0 or np.int(action) > 6:
        raise ValueError
    i = 0
    while i <= 5 and board[i, action] != 0:
        i += 1
    if i <= 5:
        if copy:
            initial_board = np.copy(board)  # what do we do with this copy?
        board[i, action] = player

    return board


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """

    for i in range(6):
        if connected_four_line(board[i, :], player):
            return True

    for j in range(7):
        if connected_four_line(board[:, j], player):
            return True

    # define the diagonals
    # TODO: automatic generation of diagonals
    # rows = 6
    # cols = 7
    # diags = [[board[sum_ - k][k] for k in range(sum_ + 1) if (sum_ - k) < rows and k < cols]
    #          for sum_ in range(rows + cols - 1)]
    # diags = [[board[i, j] for i in range(rows)]]
    # print(diags)

    diag1 = np.array([board[3, 0], board[2, 1], board[1, 2], board[0, 3]])
    diag2 = np.array([board[4, 0], board[3, 1], board[2, 2], board[1, 3], board[0, 4]])
    diag3 = np.array([board[5, 0], board[4, 1], board[3, 2], board[2, 3], board[1, 4], board[0, 5]])
    diag4 = np.array([board[5, 1], board[4, 2], board[3, 3], board[2, 4], board[1, 5], board[0, 6]])
    diag5 = np.array([board[5, 2], board[4, 3], board[3, 4], board[2, 5], board[1, 6]])
    diag6 = np.array([board[5, 3], board[4, 4], board[3, 5], board[2, 6]])

    diag7 = np.array([board[2, 0], board[3, 1], board[4, 2], board[5, 3]])
    diag8 = np.array([board[1, 0], board[2, 1], board[3, 2], board[4, 3], board[5, 4]])
    diag9 = np.array([board[0, 0], board[1, 1], board[2, 2], board[3, 3], board[4, 4], board[5, 5]])
    diag10 = np.array([board[0, 1], board[1, 2], board[2, 3], board[3, 4], board[4, 5], board[5, 6]])
    diag11 = np.array([board[0, 2], board[1, 3], board[2, 4], board[3, 5], board[4, 6]])
    diag12 = np.array([board[0, 3], board[1, 4], board[2, 5], board[3, 6]])

    if connected_four_line(diag1, player):
        return True
    if connected_four_line(diag2, player):
        return True
    if connected_four_line(diag3, player):
        return True
    if connected_four_line(diag4, player):
        return True
    if connected_four_line(diag5, player):
        return True
    if connected_four_line(diag6, player):
        return True
    if connected_four_line(diag7, player):
        return True
    if connected_four_line(diag8, player):
        return True
    if connected_four_line(diag9, player):
        return True
    if connected_four_line(diag10, player):
        return True
    if connected_four_line(diag11, player):
        return True
    if connected_four_line(diag12, player):
        return True

    return False


def connected_four_line(line: np.ndarray, player: BoardPiece) -> bool:
    winning_sequence = np.full(4, player)
    line_boolean = (line == player)
    player_on_line = np.count_nonzero(line_boolean)
    if player_on_line >= 4:
        # print("Nice row")
        for first in range(0, len(line) - 3):
            # print(line[first: first + 4])
            if np.all(line[first: first + 4] == winning_sequence):
                return True
    return False


def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """

    if connected_four(board, player):
        return GameState.IS_WIN
    if NO_PLAYER in board:
        return GameState.STILL_PLAYING
    return GameState.IS_DRAW


# b = initialize_game_state()
# b[:, 0] = PLAYER1
# b[:, 1] = PLAYER2
# b_str = pretty_print_board(b)
# print(b_str)
#
# # pretty_print_board(b)
# new_b = apply_player_action(b, np.int8(6), PLAYER1)
# new_b = apply_player_action(b, np.int8(6), PLAYER2)
# new_b = apply_player_action(b, np.int8(6), PLAYER2)
# new_b = apply_player_action(b, np.int8(6), PLAYER1)
#
# new_b = apply_player_action(b, np.int8(2), PLAYER2)
# new_b = apply_player_action(b, np.int8(3), PLAYER2)
# new_b = apply_player_action(b, np.int8(4), PLAYER2)
#
# new_str = pretty_print_board(new_b)
# print(new_str)
#
# print(new_str.find('=|\n|'))
# print(new_str[15])
#
# back_board = string_to_board(new_str)
# print(back_board)
#
# cf = connected_four(new_b, PLAYER1)
# print(cf)
#
# rows = 6
# cols = 7
# diags = [[new_b[sum_ - k][k]
#           for k in range(sum_ + 1)
#           if (sum_ - k) < rows and k < cols]
#          for sum_ in range(rows + cols - 1)]
# print(diags)
#
# a = initialize_game_state()
# for i in range(0, 3):
#     a = apply_player_action(a, np.int8(0), PLAYER1)
# for i in range(0, 2):
#     a = apply_player_action(a, np.int8(1), PLAYER1)
# a = apply_player_action(a, np.int8(2), PLAYER1)
# a = apply_player_action(a, np.int(0), PLAYER2)
# a = apply_player_action(a, np.int(1), PLAYER2)
# a = apply_player_action(a, np.int(2), PLAYER2)
# # a = apply_player_action(a, np.int(3), PLAYER2)
# print(pretty_print_board(a))
# cf = connected_four(a, PLAYER2)
# print(cf)
#
# print(check_end_state(a, PLAYER2))
