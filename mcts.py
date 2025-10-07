from __future__ import annotations
from chess import Board, Move
import math


# dataclass for edge, save vals, has node that it ends at
# node--board I suppose (though only needed for leaves...), edges
# rootNode, inherits from node, has rollout method, init to have edges with noise
class Edge:
    def __init__(self, P: float, N: int, Q: float, move: Move, dest: Node):
        self.P = P  # prior probability
        self.N = N  # number of times visited
        self.Q = Q  # action value
        self.move = move
        self.dest = dest

    def get_UCT(self, parent_N: int):
        # verify this is correct
        C = 1.5  # Controls exploration. What should this be for training vs play?
        UCT = self.Q + self.P * C * math.sqrt(parent_N) / (1 + self.N)
        return UCT


class Node:
    def __init__(self, board: Board):
        self.board = board
        self.edges: list[Edge] = []
        self.expanded = False

    def expand(self):
        self.expanded = True
        for move in self.board.legal_moves:
            new_board = self.board.copy()
            new_board.push(move)
            self.edges.append(Edge(P=0.05, N=0, Q=0.0, move=move, dest=Node(new_board)))


class RootNode(Node):
    def __init__(self, board: Board):
        super().__init__(board)
        self.expand()  # for training. will want to initialize edges from root to have dirichlet noise in prior probabilities

    def make_move(self):
        best_edge = self._get_best_edge(self, 1)
        return best_edge.move, best_edge.dest

    def run_simulations(self, simulations: int):
        expanded = 0
        cur_node = self
        parent_N = 1  # Just pass in 1 if we don't have a parent_N??
        edge_path = []
        while expanded < simulations:
            if not cur_node.edges:  # draw, loss, win
                break  # How to handle this??
            best_edge = self._get_best_edge(cur_node, parent_N)
            edge_path.append(best_edge)
            parent_N = best_edge.N
            cur_node = best_edge.dest
            if not cur_node.expanded:
                cur_node.expand()
                simulations += 1
                self._backpropogate(edge_path)
                cur_node = self  # start a new simulation

    def _backpropogate(self, edge_path: list[Edge]):
        # No idea what's going on here--verify this is correct
        new_Q = 0.05  # This is not how this works
        for i in range(len(edge_path), -1, -1):
            edge = edge_path[i]
            edge.N += 1
            if (len(edge_path) - i) % 2 == 0:
                edge.Q += new_Q
            else:
                edge.Q += 1 - new_Q

    def _get_best_edge(self, node: Node, parent_N: int):
        edge_UCTs = [edge.get_UCT(parent_N) for edge in node.edges]
        return node.edges[edge_UCTs.index(max(edge_UCTs))]
