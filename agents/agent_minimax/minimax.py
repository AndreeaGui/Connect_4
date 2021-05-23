from agents.common import PlayerAction, BoardPiece, SavedState, GenMove, PLAYER1, PLAYER2, NO_PLAYER, GameState
from agents.common import connected_four, apply_player_action, check_end_state
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
    if player == PLAYER2:
        return PLAYER1
    return PLAYER2


def compute_score(board: np.ndarray, player: BoardPiece) -> float:
    """
    This method is a dummy heuristic in minimax.
    The scores returned are 100 (for winning) and -100 (for loosing). ) 0 score for any other case.
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
    """
    Computes the score in one possible line of the board (line = row, column or diagonal)
    :param line: the board line whose score is computed
    :param player: the current player making the next move
    :return: the line score for the minimax heuristic
    """
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
                line_score += 1000
            elif np.all(counts == np.array([3, 0, 1])):
                line_score += 50
            elif np.all(counts == np.array([2, 0, 2])):
                line_score += 10
            elif np.all(counts == np.array([1, 0, 3])):
                line_score += 1
            elif np.all(counts == np.array([0, 4, 0])):
                line_score += -1000
            elif np.all(counts == np.array([0, 3, 1])):
                line_score += -50
            elif np.all(counts == np.array([0, 2, 2])):
                line_score += -10
            elif np.all(counts == np.array([0, 1, 3])):
                line_score += -1
        else:
            if np.all(counts == np.array([4, 0, 0])):
                line_score += -1000
            elif np.all(counts == np.array([3, 0, 1])):
                line_score += -50
            elif np.all(counts == np.array([2, 0, 2])):
                line_score += -10
            elif np.all(counts == np.array([1, 0, 3])):
                line_score += -1
            elif np.all(counts == np.array([0, 4, 0])):
                line_score += 1000
            elif np.all(counts == np.array([0, 3, 1])):
                line_score += 50
            elif np.all(counts == np.array([0, 2, 2])):
                line_score += 10
            elif np.all(counts == np.array([0, 1, 3])):
                line_score += 1
    return line_score


def compute_score_2(board: np.ndarray, player: BoardPiece) -> int:
    """
    This method is a smart heuristic for minimax. It associates a score to each board state.
    :param board: the board state that needs computing the score
    :param player: the player for whom the score is computed
    :return: the final and total score of the minimax heuristic
    """

    score = 0
    for i in range(6):
        score += find_line_score(board[i, :], player)
    for i in range(7):
        score += find_line_score(board[:, i], player)
    main_diagonals = generate_main_diagonals(board)
    for d in main_diagonals:
        score += find_line_score(np.ndarray(d), player)
    second_diagonals = generate_second_diagnals(board)
    for d in second_diagonals:
        score += find_line_score(np.ndarray(d), player)

    return score


def generate_child_boards(board: np.array, player: BoardPiece) -> [np.array]:
    """
    This method creates the children of the current root board.
    :param board: the root board
    :param player: the current player, making the next move
    :return: a list of 7 child boards
    """
    children_boards = []

    for move in range(7):
        board_copy = board.copy()
        children_boards.append(apply_player_action(board_copy, np.int8(move), player))

    return children_boards


def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState], depth=4
                          ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generate the next move for the minimax agent.
    :param board: the current board state
    :param player: the current player who should make the next move
    :param saved_state: the last saved state
    :param depth: the number of future moves to be considered by the minimax search
    :return: the next action, the new saved state
    """
    children = generate_child_boards(board, player)
    scores = np.zeros(7)

    for i in range(len(children)):
        scores[i] = minimax_algorithm(children[i], player, find_opponent(player), depth - 1, NEGATIVE_INF, POSITIVE_INF)
    next_move = np.argmax(scores)

    return np.int8(next_move), saved_state


def minimax_algorithm(board: np.ndarray, root_player: BoardPiece, current_player: BoardPiece,
                      depth: int = 4, alpha=NEGATIVE_INF, beta=POSITIVE_INF) -> float:
    """
    The recursive minimax algorithm with alpha-beta pruning and dynamic depth.
    :param board: the current board
    :param root_player: the player who makes the move on the root board
    :param current_player: the player making the move on the current board
    :param depth: the current depth
    :param alpha: alpha factor in alpha-beta pruning
    :param beta: beta factor in alpha-beta pruning
    :return:
    """
    if depth == 0 or check_end_state(board, current_player) != GameState.STILL_PLAYING:
        # score = compute_score(board, root_player)
        score = compute_score_2(board, root_player)
        return score

    children = generate_child_boards(board, current_player)

    if current_player == root_player:
        max_score = NEGATIVE_INF
        for i in range(len(children)):
            score = minimax_algorithm(children[i], root_player, find_opponent(current_player), depth - 1, alpha, beta)
            max_score = np.maximum(max_score, score)
            alpha = np.maximum(alpha, score)
            if beta <= alpha:
                break
        return max_score
    else:
        min_score = POSITIVE_INF
        for i in range(len(children)):
            score = minimax_algorithm(children[i], root_player, find_opponent(current_player), depth - 1, alpha, beta)
            min_score = np.minimum(min_score, score)
            beta = np.minimum(beta, score)
            if beta <= alpha:
                break
        return min_score
