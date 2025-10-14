import sys

while True:
    command = sys.stdin.readline().strip()
    if "uci" in command:
        print("id name Don")
        print("id author Zach Gorton")
        print("uciok")
    elif "isready" in command:
        print("readyok")
    elif "go" in command:
        print("bestmove e2e4")
    elif "quit" in command:
        break
