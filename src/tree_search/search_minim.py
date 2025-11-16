import chess
from .model import Evaluator
import torch
from .chess_utils import encode

class EvaluatorWrapper:
    def __init__(self, model_path, n_blocks=8):
        self.model = Evaluator(in_channels=19, channels=32, n_blocks=n_blocks)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.eval_cache = {}

    def evaluate(self, board):
        fen = board.fen()
        if fen in self.eval_cache:
            return self.eval_cache[fen]

        with torch.no_grad():
            x = encode(board)
            output = self.model(x)
            score = output.item() * 100
        
        if len(self.eval_cache) > 50000:
            self.eval_cache.clear()
        
        self.eval_cache[fen] = score
        return score if board.turn == chess.WHITE else -score

def negamax(board: chess.Board, depth: int, evaluator: EvaluatorWrapper) -> float:
    # Terminal positions
    if board.is_game_over():
        if board.is_checkmate():
            return -1e8
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0
        return 0
    
    # Leaf node - evaluate
    if depth == 0:
        return evaluator.evaluate(board)

    max_value = -float("inf")
    moves = list(board.legal_moves)

    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, evaluator)
        board.pop()

        if score > max_value:
            max_value = score

    return max_value

def negamax_root(board: chess.Board, depth: int, evaluator: EvaluatorWrapper):
    best_score = -float("inf")
    best_move = None
    moves = list(board.legal_moves)

    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, evaluator)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move, best_score
