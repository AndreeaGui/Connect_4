import numpy as np
from typing import Optional, Callable
from agents.common import PlayerAction, BoardPiece, SavedState, GenMove
from agents.agent_random import generate_move
from agents.agent_minimax import generate_move
from agents.agent_mcts import generate_move


def user_move(board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState]):
    action = PlayerAction(-1)
    while not 0 <= action < board.shape[1]:
        try:
            action = PlayerAction(input("Column? "))
        except ValueError:
            print("Input could not be converted to the dtype PlayerAction, try entering an integer.")
    return action, saved_state


def human_vs_agent(
        generate_move_1: GenMove,
        generate_move_2: GenMove = user_move,
        player_1: str = "Player 1",
        player_2: str = "Player 2",
        args_1: tuple = (),
        args_2: tuple = (),
        init_1: Callable = lambda board, player: None,
        init_2: Callable = lambda board, player: None,
):
    import time
    from agents.common import PLAYER1, PLAYER2, PLAYER1_PRINT, PLAYER2_PRINT, GameState
    from agents.common import initialize_game_state, pretty_print_board, apply_player_action, check_end_state

    players = (PLAYER1, PLAYER2)
    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        move_average = []
        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                    players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                )
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args
                )
                move_time = time.time() - t0
                move_average.append(move_time)
                print(f"Move time: {move_time:.3f}s")
                # print("added")

                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)
                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw")
                    else:
                        print(
                            f'{player_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                        )
                    playing = False
                    break
        avg_time = np.array(move_average[1::2]).mean()
        print(f"Average time of the first player: {avg_time:.3f}s")


if __name__ == "__main__":
    # human_vs_agent(user_move)
    human_vs_agent(generate_move)
