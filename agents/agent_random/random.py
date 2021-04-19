from agents.common import PlayerAction, BoardPiece, SavedState, GenMove
from typing import Optional, Callable, Tuple
import random
import numpy as np


def generate_move_random(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    action = PlayerAction(random.randint(0, 6))
    return action, saved_state
