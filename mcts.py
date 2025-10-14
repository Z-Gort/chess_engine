from chess import Board, Move, WHITE
import math


def pv_uniform_zero(board):
    moves = list(board.legal_moves)
    p = {m: 1.0 / len(moves) for m in moves} if moves else {}
    v = 0.0  # [-1,1]
    return p, v


class Node:
    def __init__(self, board: Board):
        self.board = board
        self.edges: list[Edge] = []
        self.expanded = False

    def expand(self, prior_dict: dict[Move, float]):
        self.expanded = True
        for move in self.board.legal_moves:
            self.edges.append(Edge(P=prior_dict[move], move=move))

    def get_best_edge(self):
        parent_sum_N = 1 + sum(edge.N for edge in self.edges)
        C = 1.5  # Controls exploration
        return max(
            self.edges,
            key=lambda e: e.Q
            + C * e.P * math.sqrt(parent_sum_N) / (1 + e.N),  # PUCT AlphaGo variant
        )

    def is_terminal(self):
        return self.board.is_game_over()

    def get_terminal_value(self) -> float:
        outcome = self.board.outcome()
        if outcome is None:  # not terminal
            raise RuntimeError("Called get_terminal_value on non-terminal node")
        if outcome.winner is None:
            return 0.0
        # value from POV of side-to-move at this node
        return 1.0 if outcome.winner == self.board.turn else -1.0


class Edge:
    def __init__(self, P, move):
        self.move = move
        self.P = P
        self.N: int = 0
        self.W = 0.0
        self.dest: Node | None = None

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N else 0.0


class MCTS:
    def __init__(self, node: Node):
        # self.expand(dirichlet_noise_dict)--for training. will want to initialize edges from root to have dirichlet noise in prior probabilities
        self.fn = pv_uniform_zero  # TODO
        self.root = node

    def make_move(self, move: Move):
        for edge in self.root.edges:
            if edge.move == move:
                if not edge.dest:
                    b2 = self.root.board.copy()
                    b2.push(edge.move)
                    edge.dest = Node(b2)
                self.root = edge.dest
                return
        # if edge not explored in root (first move or no simulations)
        b2 = self.root.board.copy()
        b2.push(move)
        self.root =Node(b2)

    def think_and_move(self, simulations: int): # simulations MUST be at least 1
        self._run_simulations(simulations)
        best_edge = max(self.root.edges, key=lambda e: e.N)
        # Make sure a child exists
        if not best_edge.dest:
            b2 = self.root.board.copy()
            b2.push(best_edge.move)
            best_edge.dest = Node(b2)
        self.make_move(best_edge.move)

    def _run_simulations(self, simulations: int):
        for _ in range(simulations):
            node = self.root
            edge_path = []
            while node.expanded and not node.is_terminal():
                best_edge = node.get_best_edge()
                edge_path.append(best_edge)
                if not best_edge.dest:
                    b2 = node.board.copy()
                    b2.push(best_edge.move)
                    best_edge.dest = Node(b2)
                node = best_edge.dest

            if node.is_terminal():
                leaf_value = node.get_terminal_value()
            else:
                prior_dict, leaf_value = self.fn(node.board)
                node.expand(prior_dict)

            self._backpropogate(edge_path, leaf_value)

    def _backpropogate(self, edge_path: list[Edge], leaf_value: float):
        v = leaf_value
        for edge in reversed(edge_path):
            edge.N += 1
            edge.W += v
            v = -v  # flip persepective
