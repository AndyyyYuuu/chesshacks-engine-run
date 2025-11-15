import chess
from .model import Evaluator
import torch
from .chess_utils import encode

class TranspositionTable:
    def __init__(self, max_size=100000):
        self.table = {}
        self.max_size = max_size
    
    def get_key(self, board):
        return board.fen()
    
    def lookup(self, board: chess.Board, depth: int, alpha: float, beta: float) -> tuple[float, str, chess.Move] | None:

        key = self.get_key(board)
        if key not in self.table:
            return None
        
        entry = self.table[key]
        
        # Only use if stored depth is sufficient
        if entry['depth'] < depth:
            return None
        
        score = entry['score']
        flag = entry['flag']
        
        # Alpha-Beta caching
        if flag == 'EXACT':
            return (score, flag, entry.get('best_move'))
        elif flag == 'LOWER' and score >= beta:
            return (beta, flag, entry.get('best_move'))
        elif flag == 'UPPER' and score <= alpha:
            return (alpha, flag, entry.get('best_move'))
        
        return None 
    
    def store(self, board, depth, score, flag, best_move=None):

        if len(self.table) >= self.max_size:
            # clear half when full
            items = list(self.table.items())
            self.table = dict(items[self.max_size//2:])
        
        key = self.get_key(board)
        self.table[key] = {
            'depth': depth,
            'score': score,
            'flag': flag,
            'best_move': best_move
        }
    
    def clear(self):
        self.table.clear()

class EvaluatorWrapper:
    def __init__(self, model_path):
        self.model = Evaluator(in_channels=19, channels=32, n_blocks=8)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def evaluate(self, board):
        with torch.no_grad():
            x = encode(board)  # (1, 19, 8, 8)
            output = self.model(x)  # (1, 1)
            return output.item() * 100  # scale to centipawns

def _capture_value(board, move):
    # Most valuable victim - least valuable attacker
    
    if not board.is_capture(move):
        return 0
    
    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    victim_value = piece_values.get(victim.piece_type, 0) if victim else 0
    attacker_value = piece_values.get(attacker.piece_type, 0) if attacker else 0
    
    return victim_value * 10 - attacker_value

def order_moves(board, moves, tt_move:chess.Move=None):
    ordered = []
    captures = []
    others = []
    
    for move in moves:
        if tt_move and move == tt_move:
            ordered.insert(0, move)
        elif board.is_capture(move):
            captures.append(move)
        else:
            others.append(move)
    
    captures.sort(key=lambda m: _capture_value(board, m), reverse=True)
    
    return ordered + captures + others

def quiescence_search(board, depth, alpha, beta, evaluator):
    if depth <= 0:
        return evaluator.evaluate(board)

    stand_pat = evaluator.evaluate(board)

    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    captures = [move for move in board.legal_moves if board.is_capture(move)]
    captures = order_moves(board, captures)

    for move in captures:
        board.push(move)
        score = -quiescence_search(board, depth - 1, -beta, -alpha, evaluator)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def negamax(board: chess.Board, depth: int, alpha: float, beta: float, evaluator: EvaluatorWrapper) -> float:
    if depth == 0 or board.is_game_over():
        return quiescence_search(board, 8, alpha, beta, evaluator)

    max_value = -float("inf")
    moves = order_moves(board, list(board.legal_moves))

    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, evaluator)
        board.pop()

        if score > max_value:
            max_value = score

        if score > alpha:
            alpha = score

        if alpha >= beta:
            break

    return max_value

def negamax_root(board: chess.Board, depth: int, evaluator: EvaluatorWrapper):
    best_score = -float("inf")
    best_move = None
    alpha = -1e9
    beta  = 1e9


    moves = order_moves(board, list(board.legal_moves))

    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, evaluator)
        board.pop()

        print("Root check:", move, score)

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

    return best_move, best_score


if __name__ == "__main__":
    evaluator = EvaluatorWrapper("best_model.pt")

    board = chess.Board()
    print(board)

    depth = 4
    best_move, best_score = negamax_root(board, depth, evaluator)

    print(f"Best Move: {best_move}, Score: {best_score}")
