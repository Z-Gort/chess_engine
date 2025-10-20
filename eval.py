import chess
from mcts import MCTS, Node


def play_match(
    sim_count_white,
    sim_count_black,
    verbose=False,
):
    """
    Play a single game between two MCTS bots.

    Args:
        sim_count_white: Number of simulations for white bot
        sim_count_black: Number of simulations for black bot
        weights_file_white: Path to weights file for white bot
        weights_file_black: Path to weights file for black bot
        verbose: If True, print board state after each move

    Returns:
        Result string: "1-0" (white wins), "0-1" (black wins), or "1/2-1/2" (draw)
    """
    board = chess.Board()

    # Initialize both bots
    print(f"Initializing White bot ({sim_count_white} sims)...")
    white_mcts = MCTS(Node(board.copy()))

    print(f"Initializing Black bot ({sim_count_black} sims)...")
    black_mcts = MCTS(Node(board.copy()))

    move_count = 0

    while not board.is_game_over():
        move_count += 1

        if board.turn == chess.WHITE:
            if verbose:
                print(f"\nMove {move_count} - White to move")
                print(board)
                print("--------------------------------")
            move = white_mcts.think_and_move(sim_count_white)
            black_mcts.make_move(move)
            board = white_mcts.root.board
        else:
            if verbose:
                print(f"\nMove {move_count} - Black to move")
                print(board)
                print("--------------------------------")
            move = black_mcts.think_and_move(sim_count_black)
            white_mcts.make_move(move)
            board = black_mcts.root.board

    # Game over - determine result
    outcome = board.outcome()
    if outcome:
        if outcome.winner == chess.WHITE:
            result = "1-0"
            result_text = "White wins"
        elif outcome.winner == chess.BLACK:
            result = "0-1"
            result_text = "Black wins"
        else:
            result = "1/2-1/2"
            result_text = "Draw"

        print(f"\nGame over after {move_count} moves!")
        print(f"Result: {result} ({result_text})")
        print(f"Termination: {outcome.termination}")
    else:
        result = "Unknown"
        print("\nGame ended without outcome")

    if verbose:
        print("\nFinal position:")
        print(board)

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("Chess Engine Evaluation: 50 sims vs 100 sims")
    print("=" * 60)

    result = play_match(sim_count_white=1, sim_count_black=200, verbose=True)
