import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, GameState

# Sample random board variables
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
'''|==============|
|              |
|              |
|    X X       |
|    O X X     |
|  O X O O     |
|  O O X X     |
|==============|
|0 1 2 3 4 5 6 |'''

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

b3 = np.empty((6, 7), dtype=BoardPiece)
b3.fill(NO_PLAYER)
b3[0, 1] = PLAYER2
b3[1, 1] = PLAYER2
b3[0, 2] = PLAYER2
b3[2, 2] = PLAYER2
b3[1, 3] = PLAYER1
b3[1, 4] = PLAYER2
b3[1, 2] = PLAYER2
b3[3, 2] = PLAYER1
b3[0, 3] = PLAYER1
b3[2, 3] = PLAYER1
b3[3, 3] = PLAYER1
b3[0, 4] = PLAYER1
b3[2, 4] = PLAYER1
'''|==============|
|              |
|              |
|    X X       |
|    O X X     |
|  O O X O     |
|  O O X X     |
|==============|
|0 1 2 3 4 5 6 |'''

b4 = np.empty((6, 7), dtype=BoardPiece)
b4.fill(NO_PLAYER)
b4[0, 1] = PLAYER2
b4[1, 1] = PLAYER2
b4[0, 2] = PLAYER2
b4[2, 2] = PLAYER2
b4[1, 3] = PLAYER1
b4[1, 4] = PLAYER1
b4[1, 2] = PLAYER2
b4[3, 2] = PLAYER1
b4[0, 3] = PLAYER2
b4[2, 3] = PLAYER1
b4[3, 3] = PLAYER1
b4[0, 4] = PLAYER1
b4[2, 4] = PLAYER1
b4[0, 5] = PLAYER1
'''|==============|
|              |
|              |
|    X X       |
|    O X X     |
|  O O X X     |
|  O O O X X   |
|==============|
|0 1 2 3 4 5 6 |'''

b5 = np.empty((6, 7), dtype=BoardPiece)
b5.fill(PLAYER1)
b5[0:3, [1, 3, 5]] = PLAYER2
b5[3:6, [0, 2, 4, 6]] = PLAYER2


def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_apply_player_action():
    from agents.common import initialize_game_state, apply_player_action

    action = PlayerAction(3)
    player = PLAYER1
    board = initialize_game_state()  # TODO replace it with hardcoded board
    board_after_action = apply_player_action(board, action, player)
    assert board_after_action.shape == board.shape
    assert board_after_action.any() == PLAYER1
    assert board_after_action[:, action].any() == PLAYER1


def test_connect_four():
    from agents.common import connected_four

    assert not connected_four(b1, PLAYER1)
    assert not connected_four(b1, PLAYER2)
    assert not connected_four(b2, PLAYER1)
    assert connected_four(b2, PLAYER2)
    assert connected_four(b3, PLAYER1)
    assert not connected_four(b3, PLAYER2)
    assert connected_four(b4, PLAYER1)
    assert not connected_four(b4, PLAYER2)


def test_check_end_state():
    from agents.common import check_end_state

    assert check_end_state(b1, PLAYER1) == GameState.STILL_PLAYING
    assert check_end_state(b1, PLAYER2) == GameState.STILL_PLAYING
    assert check_end_state(b2, PLAYER2) == GameState.IS_WIN
    assert check_end_state(b3, PLAYER1) == GameState.IS_WIN
    assert check_end_state(b5, PLAYER1) == GameState.IS_DRAW
    assert check_end_state(b5, PLAYER2) == GameState.IS_DRAW


def test_pretty_print_board():
    from agents.common import pretty_print_board
    assert pretty_print_board(b4) == "|==============|\n" \
                                     "|              |\n" \
                                     "|              |\n" \
                                     "|    X X       |\n" \
                                     "|    O X X     |\n" \
                                     "|  O O X X     |\n" \
                                     "|  O O O X X   |\n" \
                                     "|==============|\n" \
                                     "|0 1 2 3 4 5 6 |"


def test_string_to_board():
    from agents.common import string_to_board
    b4_string = "|==============|\n" \
                "|              |\n" \
                "|              |\n" \
                "|    X X       |\n" \
                "|    O X X     |\n" \
                "|  O O X X     |\n" \
                "|  O O O X X   |\n" \
                "|==============|\n" \
                "|0 1 2 3 4 5 6 |\n"
    assert np.all(string_to_board(b4_string) == b4)
