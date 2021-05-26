from agents.common import PlayerAction, BoardPiece, SavedState, GenMove
from typing import Optional, Callable, Tuple
import random
import numpy as np


def generate_move_random(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generate the next move for the random agent.
    :param board: current board
    :param player: current player making the next move
    :param saved_state: the last saved state of the board
    :return: the column of the next move, the nwe saved state
    """
    # Remark: what about full columns?
    action = PlayerAction(random.randint(0, 6))
    return action, saved_state
