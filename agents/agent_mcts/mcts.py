from agents.common import PlayerAction, BoardPiece, SavedState, GenMove, PLAYER1, PLAYER2, NO_PLAYER, GameState
from agents.common import connected_four, apply_player_action, check_end_state, initialize_game_state
from agents.common import pretty_print_board, find_opponent, possible_moves

import time
from typing import Optional, Tuple
import math
import numpy as np

C = math.sqrt(2)  # global exploration parameter


# Remark: this is a really nice implementation! You really did improve it a lot since our last discussion, I'm very
#  happy to see that. I only have very few suggestions for improvement, see below. Unfortunately, your agent is rather
#  slow, which hampers performance quite a lot.
#  I've added numba and the njit property decorator, which speeds things up considerably. You might not notice it
#  immediately, but try increasing the number of iterations :)
#  With ~2000, your agent plays pretty well already. The remaining hiccups are probably due to the fact that you don't
#  handle wins in your selection/expansion phases, which can lead to collecting wrong winners if both players have a
#  win.
# Remark: I don't know why, but at one point your agent returned an invalid move and played in a full column

class MCTSNode(object):
    # Remark: class docstring missing
    def __init__(self, board, player, parent=None):
        self.board = board
        self.player = player
        self.parent = parent  # is a MCTSNode
        self.children = []  # is a list of MCTSNode
        self.plays = 1
        self.wins = 0
        self.children_index = possible_moves(board)   # Remark: there are better names for this attribute (e.g. valid_moves)

    def compute_ucb1(self):
        return self.wins / self.plays + C * np.sqrt(np.log(self.parent.plays) / self.plays)


def upper_confidence_bound_1(wins, plays, parent_plays) -> float:
    """
    The function that computes UCB1 score.
    It is used to select between children the child that follows next in the MC tree traversal
    :param wins: child node wins from all its simulations
    :param plays:  child node all simulations number
    :param parent_plays: parent node total number of simulations
    :return: one node UCB1 score
    """
    # Remark: why do you need this? You have the same functionality as method (which is way more convenient, since you
    #  only need to pass in a node)
    # Remark: Ah, you're using it in the mcts algorithm function. This isn't necessary though, you could just call the
    #  method on the children as well.
    return wins / plays + C * np.sqrt(np.log(parent_plays) / plays)


def do_selection(root_node: MCTSNode):
    """
    The 1st part of the algorithm: starting from the root, a leaf is found. If the current node has all children
    already expanded, the algorithm selects between these children the one with highest UCB1 score and the search for
    a leaf continues.
    :param root_node: the first node of the MCTS tree
    :return: the leaf from which a new node will be created in the current simulation
    """
    current_node = root_node
    # Remark: you should also stop selecting at leaf nodes of the game (in this case only wins, a draw is covered in the
    #  condition for entering the while loop). This is most easily done by storing whether a node has a win as an
    #  attribute (in order to avoid having to check that every time you select).
    while len(current_node.children) == len(current_node.children_index):
        # Remark: this implementation is fine, but just as a sidenote: you could take refactoring to the extreme here
        #  and implement a function which returns the child with max ucb1 value. This might seem super minor, but I
        #  think splitting functions into its logical parts is an easy way to improve code. I've added an alternative
        #  selection function below (with a new method to determine whether the node is a leaf node for selection as
        #  well). Note how you could just write down this selection function without a single thought
        #  about implementation details. The refactoring keyphrase here is separation of concerns, and to some extent
        #  loosely coupled code.
        ucb_scores = np.array(
            [upper_confidence_bound_1(c.wins, c.plays, c.parent.plays) for c in current_node.children])
        # ucb_scores = np.array([c.compute_ucb1() for c in current_node.children])
        selected_node_index = np.argmax(ucb_scores)
        current_node = current_node.children[selected_node_index]  # we go to the next node
    return current_node


def alternative_selection_function(root_node: MCTSNode):
    current_node = root_node
    while current_node.is_fully_expanded():
        current_node = current_node.select_max_ucb1_child()
    return current_node
    # Remark: of course stopping at wins would have to be added here as well.


# Remark: is_fully_expanded() would be implemented inside the node class like this:
#       def is_leaf_node_for_selection(self):
#           is_fully_expanded = len(self.children) == len(self.children_index)  # True if all children are there
#           return is_fully_expanded



def do_expansion(current_node: MCTSNode):
    """
    The 2nd part of the algorithm: once a leaf was found, a new node is created (expanded).
    :param current_node: the found leaf node.
    :return: a newly created node, added to the MCTS tree. From this node the simulation will start.
    """
    next_child_index = current_node.children_index[len(current_node.children)]
    # Remark: you should expand randomly
    # Remark: nodes with a win shouldn't be expanded
    expanded_board = apply_player_action(current_node.board, next_child_index, current_node.player, copy=True)

    expanded_node = MCTSNode(expanded_board, find_opponent(current_node.player), current_node)
    current_node.children.append(expanded_node)
    return expanded_node


def run_simulation(start_node: MCTSNode, root_player: BoardPiece, print_final=False) -> (np.ndarray, GameState):
    """
    The 3rd part of the algorithm.
    This function runs a complete game with random moves from the start node board until one player wins.
    This is one simulation in the MCTS algorithm.
    :param start_node: the expended node from which we start the simulation
    :param root_player:
    :param print_final: flag variable for printing the final board of the game
    :return: the final board state (np.narray), the game end state (GameState)
    """
    current_board = start_node.board.copy()
    current_player = start_node.player
    while check_end_state(current_board, current_player) == GameState.STILL_PLAYING:
        # action = np.random.randint(0, 7)
        possible_actions = possible_moves(current_board)
        action = possible_actions[np.random.randint(len(possible_actions))]
        # Remark: why not action = np.random.choice(possible_actions)? Looks a bit simpler to me
        current_board = apply_player_action(current_board, np.int8(action), current_player)
        current_player = find_opponent(current_player)

    game_result = check_end_state(current_board, root_player)
    if print_final:
        print(pretty_print_board(current_board))
    return current_board, game_result
    # Remark: I'm not sure if introducing GameState.IS_LOST is such a good idea. You don't technically need it, because
    #  a loss is equivalent to a win for the opponent. But your check_end_state function contains 2 calls of
    #  connected_four (to check for wins of both players), even though in a running game, you only need to check the
    #  player who moved last. Since checking the end state is the most time-consuming part of the simulation (which, in
    #  turn, is probably the most time-consuming part of the whole mcts algorithm), you slow down your implementation
    #  quite a bit.


def back_propagate_statistics(expanded_node: MCTSNode, gain_wins_player: BoardPiece):
    """
    The 4th part of the algorithm: the node statistics wins and plays are updated for the current path.
    The current path contains all the nodes from the expanded and simulated node, back to the root.
    :param expanded_node: the node that was expanded for the current trial & starting node for the simulation
    :param gain_wins_player: the player that will have the wins statistics increased
    :return: nothing; the MCTS tree itself is updated
    """
    n = expanded_node  # Remark: n is not a good variable name, even though it's just a dummy for the loop
    while n.parent is not None:  # this happens to be true only for the root node
        n.plays += 1
        # update the wins for the losing nodes
        # because they are actually useful for their children - that have the opponent player of the loser
        if n.player == gain_wins_player:
            n.wins += 1
        n = n.parent
    # update for the root node
    n.plays += 1
    if n.player == gain_wins_player:
        n.wins += 1
        # Remark: you could change the condition to enter the while loop to while n is not None, that way you stop after
        #  the root only.
        # Remark: Now that I think about it: actually, the statistics of the root are irrelevant for the mcts algorithm :)


def mcts_algorithm(board: np.ndarray, root_player: BoardPiece, trials=100, profiling=False) -> list:
    """
        The Monte Carlo Tree Search algorithm.
        Starting from a given board, when the root_player has to do a move, it runs "trials" simulations in order to find
        which next move is the best. While doing so, it constructs a tree (data structure composed by MCTSNode objects,
        connected by .parent and .children references).
        MCTS has 4 phases: selection, expansion, simulation and back propagation.

        :param board: the game state for which the next action has to be decided
        :param root_player: the player that should do the next action
        :param trials: number of simulations the algorithm performs for constructing the MC tree before selecting a move
        :return: the MC tree as a list
    """
    root_node = MCTSNode(board, root_player)
    mcts_tree = [root_node]

    t = np.zeros((5, trials))
    for i in range(trials):
        t[0, i] = time.time()
        selected_node = do_selection(root_node)
        t[1, i] = time.time()

        expanded_node = do_expansion(selected_node)
        mcts_tree.append(expanded_node)
        t[2, i] = time.time()

        final_board, simulation_result = run_simulation(expanded_node, root_player, print_final=False)
        t[3, i] = time.time()

        if simulation_result == GameState.IS_LOST:
            gain_wins_player = root_player
        else:
            gain_wins_player = find_opponent(root_player)
        back_propagate_statistics(expanded_node, gain_wins_player)
        t[4, i] = time.time()

    if profiling:
        print("Selection: %.3f" % (t[1, :] - t[0, :]).sum())
        print("Expansion: %.3f" % (t[2, :] - t[1, :]).sum())
        print("Simulation: %.3f" % (t[3, :] - t[2, :]).sum())
        print("Back propagation: %.3f" % (t[4, :] - t[3, :]).sum())

    return mcts_tree

# Remark: unused code shouldn't be here anymore. I understand why you might wanna keep it around, but there's really no
#  point in being too sentimental. You improved your implementation, so get rid of the old one! If you really did need
#  it again at some point, you can get it from git (assuming you used it properly :))
# first version of the algorithm
def not_used_mcts_algorithm(board: np.ndarray, root_player: BoardPiece, trials=20) -> list:
    root_node = MCTSNode(board, root_player)
    mcts_tree = [root_node]

    for i in range(trials):
        current_node = root_node
        current_player = root_player

        # selection and expansion
        extended = False
        while not extended:
            if len(current_node.children) != 7:  # this corresponds to expansion
                new_child_found = False
                child_board = None
                while not new_child_found:
                    action_made = False
                    while not action_made:
                        action = np.random.randint(0, 7)
                        child_board = apply_player_action(current_node.board, np.int8(action), current_player,
                                                          copy=True)
                        if not np.all(child_board == current_node):
                            action_made = True

                    if not current_node.children:  # we cannot iterate over an empty list; this is the true leaf case
                        new_child_found = True
                    else:
                        repeated_child = False
                        for c in current_node.children:
                            if np.all(c.board == child_board):
                                repeated_child = True
                        if not repeated_child:
                            new_child_found = True

                child_node = MCTSNode(child_board, find_opponent(current_player))
                child_node.parent = current_node
                current_node.children.append(child_node)
                mcts_tree.append(child_node)
                extended = True

            else:  # UCB1 # this corresponds to selection
                ucb_scores = np.array(
                    [upper_confidence_bound_1(c.wins, c.plays, c.parent.plays) for c in current_node.children])
                selected_node_index = np.argmax(ucb_scores)
                # mcts_tree.append(selected_node)
                current_node = current_node.children[selected_node_index]
                current_player = find_opponent(current_player)

        # simulation
        # print(i)
        # print(pretty_print_board(child_node.board))
        final_board, simulation_result = run_simulation(child_node, print_final=False)
        # print(simulation_result)

        # back propagation
        # go upwards from the child node to the root via parent
        bp_node = child_node
        while bp_node.parent is not None:
            # print(pretty_print_board(bp_node.board))
            bp_node.plays += 1
            # update the wins for the losing nodes
            # because they are actually useful for their children - that have the opponent player of the loser
            if check_end_state(final_board, bp_node.player) == GameState.IS_LOST:
                bp_node.wins += 1
            bp_node = bp_node.parent

    return mcts_tree


def generate_move_mcts(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
                       ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generate the next move for the MCTS agent.
    :param board: the current board state
    :param player: the current player who should make the next move
    :param saved_state: the last saved state
    :return: the next action, the new saved state
    """
    profiling = True
    mcts_tree = mcts_algorithm(board, player, 1000, profiling)
    root_node = mcts_tree[0]
    ucb_scores = np.array(
        [upper_confidence_bound_1(c.wins, c.plays, c.parent.plays) for c in root_node.children])
    next_move = np.argmax(ucb_scores)
    # Remark: refactor determining the child with the highest ucb score into its own function (you could reuse the one
    #  from the selection actually).
    # Remark: It's more common to select the child with the highest win rate, but usually that's the same one (at least
    #  for high enough number of iterations)

    return np.int8(next_move), saved_state
