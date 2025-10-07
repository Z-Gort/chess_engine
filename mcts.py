from __future__ import annotations
from dataclasses import dataclass
from chess import Board


# dataclass for edge, save vals, has node that it ends at
# node--board I suppose (though only needed for leaves...), edges
# rootNode, inherits from node, has rollout method, init to have edges with noise
@dataclass
class Edge:
    P: float  # prior probability
    N: int  # number of times visited
    Q: float  # action value
    dest: Node

class Node:
    def __init__(self, board: Board):
        self.board = board
        self.edges = []
    
    def expand(self):
        for move in self.board.legal_moves:
            new_board = self.board.copy()
            new_board.push(move)
            self.edges.append(Edge(P=0.05, N=0, Q=0.0, dest=Node(new_board)))

class RootNode(Node):
    def __init__(self, board: Board):
        super().__init__(board)
        self.expand() # for training. will want to initialize edges from root to have dirichlet noise in prior probabilities
        