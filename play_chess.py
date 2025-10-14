import chess
import random
from mcts import MCTS, Node

root = Node(chess.Board())
mcts = MCTS(root)
human_turn = True
while not mcts.root.board.is_game_over():
    if human_turn:
        print(mcts.root.board)
        print("--------------------------------")
        while True:
            moveStr = input("Enter your move: ")
            try:
                move = chess.Move.from_uci(moveStr)
            except:
                print("Invalid move, try again")
                continue
            if move not in mcts.root.board.legal_moves:
                print("Invalid move, try again")
                continue
            mcts.make_move(move)
            break
    else:
        mcts.think_and_move(simulations=1)
    human_turn = not human_turn
