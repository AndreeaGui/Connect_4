import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, GameState

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

b1_child1 = b1.copy()
b1_child1[0, 0] = PLAYER1
b1_child2 = b1.copy()
b1_child2[2, 1] = PLAYER1
b1_child3 = b1.copy()
b1_child3[4, 2] = PLAYER1
b1_child4 = b1.copy()
b1_child4[4, 3] = PLAYER1
b1_child5 = b1.copy()
b1_child5[3, 4] = PLAYER1
b1_child6 = b1.copy()
b1_child6[0, 5] = PLAYER1
b1_child7 = b1.copy()
b1_child7[0, 6] = PLAYER1


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


def test_generate_child_boards():
    from agents.agent_minimax.minimax import generate_child_boards

    child_list = generate_child_boards(b1, PLAYER1)
    assert np.all(b1_child1 == child_list[0])
    assert np.all(b1_child2 == child_list[1])
    assert np.all(b1_child3 == child_list[2])
    assert np.all(b1_child4 == child_list[3])
    assert np.all(b1_child5 == child_list[4])
    assert np.all(b1_child6 == child_list[5])
    assert np.all(b1_child7 == child_list[6])


def test_compute_score():
    from agents.agent_minimax.minimax import compute_score

    assert compute_score(b2, PLAYER1) == -100
    assert compute_score(b2, PLAYER2) == 100
    assert compute_score(b1, PLAYER1) == 0
    assert compute_score(b1, PLAYER2) == 0

