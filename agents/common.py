from enum import Enum
from typing import Optional, Callable, Tuple
import numpy as np

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiece_Print = str  # dtype for string representation of BoardPiece
NO_PLAYER_PRINT = str(' ')
PLAYER1_PRINT = str('X')
PLAYER2_PRINT = str('O')

PlayerAction = np.int8  # The column to be played


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


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
                board_str = board_str + PLAYER1_PRINT + ' '
            elif board[i, j] == PLAYER2:
                board_str = board_str + PLAYER2_PRINT + ' '
            else:
                board_str = board_str + NO_PLAYER_PRINT + ' '
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
            if board_str[string_index] == PLAYER1_PRINT:
                board[i, j] = PLAYER1
            elif board_str[string_index] == PLAYER2_PRINT:
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
    action = np.int(action)
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

    # main diagonals orientation
    main_diagonals = []
    main_diagonals.append([board[i, i] for i in range(0, 6)])
    main_diagonals.append([board[i, i - 1] for i in range(1, 6)])
    main_diagonals.append([board[i, i - 2] for i in range(2, 6)])
    main_diagonals.append([board[i, i + 1] for i in range(0, 6)])
    main_diagonals.append([board[i, i + 2] for i in range(0, 5)])
    main_diagonals.append([board[i, i + 3] for i in range(0, 4)])

    for d in main_diagonals:
        if connected_four_line(np.array(d), player):
            return True

    # second diagonal orientation
    second_diagonals = []
    second_diagonals.append([board[i, 3 - i] for i in range(0, 4)])
    second_diagonals.append([board[i, 4 - i] for i in range(0, 5)])
    second_diagonals.append([board[i, 5 - i] for i in range(0, 6)])
    second_diagonals.append([board[i, 6 - i] for i in range(0, 6)])
    second_diagonals.append([board[i, 7 - i] for i in range(1, 6)])
    second_diagonals.append([board[i, 8 - i] for i in range(2, 6)])

    for d in second_diagonals:
        if connected_four_line(np.array(d), player):
            return True

    return False


def connected_four_line(line: np.ndarray, player: BoardPiece) -> bool:
    """
    This method is the boilerplate code necessary in @method connected_four for checking each possible line
    :param line: a row, column or diagonal from the play board
    :param player: the player for which 4 connected points are searched
    :return: True if player had 4 connected points in the line; False otherwise
    """

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
