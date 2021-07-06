import numpy as np
from agents.common import BoardPiece, NO_PLAYER, PLAYER1, PLAYER2, PlayerAction, GameState
from agents.agent_mcts.mcts import MCTSNode

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

parent_node = MCTSNode(b1, PLAYER2)
parent_node.plays = 20
parent_node.wins = 10

b2 = b1.copy()
b2[0, 0] = PLAYER2
child_1 = MCTSNode(b2, PLAYER1, parent_node)
child_1.plays = 10
child_1.wins = 1

b3 = b1.copy()
b3[0, 5] = PLAYER2
child_2 = MCTSNode(b3, PLAYER1, parent_node)
child_2.plays = 10
child_2.wins = 9

b4 = b2.copy()
b4[1, 0] = PLAYER1
grandchild_1 = MCTSNode(b4, PLAYER2, child_1)
child_1.children.append(grandchild_1)


def test_ucb1():
    from agents.agent_mcts.mcts import upper_confidence_bound_1

    s1 = upper_confidence_bound_1(child_1.wins, child_1.plays, child_1.parent.plays)
    s2 = upper_confidence_bound_1(child_2.wins, child_2.plays, child_2.parent.plays)
    assert s1 < s2


def test_run_simulation():
    from agents.agent_mcts.mcts import run_simulation
    from agents.common import check_end_state

    for _ in range(10):
        final_board, game_status = run_simulation(parent_node, parent_node.player)
        assert check_end_state(final_board, parent_node.player) == game_status
        assert game_status == GameState.IS_WIN or game_status == GameState.IS_LOST
        assert game_status != GameState.STILL_PLAYING


def test_do_expansion():
    from agents.agent_mcts.mcts import do_expansion
    expanded_node = do_expansion(parent_node)
    assert np.all(expanded_node.board == child_1.board)
    assert expanded_node.player == child_1.player
    assert expanded_node.parent == child_1.parent

    # 1, 2, 3, 4
    for _ in range(4):
        do_expansion(parent_node)

    expanded_node_5 = do_expansion(parent_node)
    assert np.all(expanded_node_5.board == child_2.board)
    assert expanded_node_5.player == child_2.player
    assert expanded_node_5.parent == child_2.parent


def test_do_selection():
    from agents.agent_mcts.mcts import do_selection, do_expansion

    parent_node_1 = MCTSNode(b1, PLAYER2)
    for _ in parent_node_1.children_index:
        do_expansion(parent_node_1)
    assert do_selection(parent_node_1) == parent_node_1.children[0]

    parent_node_2 = MCTSNode(b1, PLAYER2)
    for i in parent_node_2.children_index:
        n = do_expansion(parent_node_2)
        if i == 4:
            n.wins = 2
            n.wins = 2
        print(i)
    assert do_selection(parent_node_2) == parent_node_2.children[4]


def test_back_propagate_statistics():
    from agents.agent_mcts.mcts import back_propagate_statistics

    back_propagate_statistics(grandchild_1, PLAYER1)
    assert grandchild_1.wins == 0
    assert grandchild_1.plays == 2
    assert child_1.wins == 2
    assert child_1.plays == 11
    assert parent_node.plays == 21
    assert parent_node.wins == 10
    assert child_2.wins == 9
    assert child_2.plays == 10
