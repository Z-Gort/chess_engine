import chess
import random

board = chess.Board()
human_turn = True
while not board.is_game_over():
    if human_turn:
        print(board)
        print("--------------------------------")
        while True:
            moveStr = input("Enter your move: ")
            try:
                move = chess.Move.from_uci(moveStr)
            except:
                print("Invalid move, try again")
                continue
            if move not in board.legal_moves:
                print("Invalid move, try again")
                continue
            board.push(move)
            break
    else:
        move = random.choice(list(board.legal_moves))
        board.push(move)
    human_turn = not human_turn
