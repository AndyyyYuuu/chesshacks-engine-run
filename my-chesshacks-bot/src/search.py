import chess
from .model import Evaluator
import torch
from .chess_utils import encode

class TranspositionTable:
    def __init__(self, max_size=100000):
        self.table = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get_key(self, board):
        return board.fen()
    
    def lookup(self, board: chess.Board, depth: int, alpha: float, beta: float) -> tuple[float, str, chess.Move] | None:
        key = self.get_key(board)
        if key not in self.table:
            self.misses += 1
            return None
        
        entry = self.table[key]
        stored_depth = entry['depth']
        score = entry['score']
        flag = entry['flag']
        
        # Use entry if depth is sufficient OR if it's an exact score
        if stored_depth < depth and flag != 'EXACT':
            self.misses += 1
            return None
        
        # Alpha-Beta caching
        if flag == 'EXACT':
            self.hits += 1
            return (score, flag, entry.get('best_move'))
        elif flag == 'LOWER' and score >= beta:
            self.hits += 1
            return (beta, flag, entry.get('best_move'))
        elif flag == 'UPPER' and score <= alpha:
            self.hits += 1
            return (alpha, flag, entry.get('best_move'))
        
        self.misses += 1
        return None
    
    def get_hit_rate(self):
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total * 100 
    
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
        self.hits = 0
        self.misses = 0

class EvaluatorWrapper:
    def __init__(self, model_path):
        self.model = Evaluator(in_channels=19, channels=32, n_blocks=8)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.eval_cache = {}  # Cache evaluations
        self.cache_hits = 0
        self.cache_misses = 0

    def evaluate(self, board):
        fen = board.fen()
        if fen in self.eval_cache:
            self.cache_hits += 1
            return self.eval_cache[fen]
        
        self.cache_misses += 1
        with torch.no_grad():
            x = encode(board)  # (1, 19, 8, 8)
            output = self.model(x)  # (1, 1)
            score = output.item() * 100  # scale to centipawns
        
        # Limit cache size to prevent memory issues
        if len(self.eval_cache) > 50000:
            self.eval_cache.clear()
        
        self.eval_cache[fen] = score
        return score
    
    def get_cache_stats(self):
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total * 100

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

def quiescence_search(board, depth, alpha, beta, evaluator, tt=None):
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
        score = -quiescence_search(board, depth - 1, -beta, -alpha, evaluator, tt)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def negamax(board: chess.Board, depth: int, alpha: float, beta: float, evaluator: EvaluatorWrapper, tt: TranspositionTable = None) -> float:
    # Check transposition table first
    if tt:
        tt_result = tt.lookup(board, depth, alpha, beta)
        if tt_result is not None:
            score, flag, stored_move = tt_result
            return score  # Return cached score
    
    # Handle terminal positions
    if board.is_game_over():
        if board.is_checkmate():
            return -1e8 if board.turn == chess.WHITE else 1e8
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0
        # Other draw conditions
        return 0
    
    if depth == 0:
        return quiescence_search(board, 8, alpha, beta, evaluator, tt)

    # Get TT move for ordering (even if lookup didn't return usable score)
    tt_move = None
    if tt:
        key = tt.get_key(board)
        if key in tt.table:
            tt_move = tt.table[key].get('best_move')

    max_value = -float("inf")
    best_move = None
    moves = order_moves(board, list(board.legal_moves), tt_move)

    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, evaluator, tt)
        board.pop()

        if score > max_value:
            max_value = score
            best_move = move

        if score > alpha:
            alpha = score

        if alpha >= beta:
            # Beta cutoff - store as LOWER bound
            if tt:
                tt.store(board, depth, beta, 'LOWER', best_move)
            return beta

    # Determine flag for storing
    if max_value <= alpha:  # All moves were <= alpha (shouldn't happen with proper initialization)
        flag = 'UPPER'
    else:  # We found a move > alpha
        flag = 'EXACT'
    
    # Store in transposition table
    if tt:
        tt.store(board, depth, max_value, flag, best_move)
    
    return max_value

def negamax_root(board: chess.Board, depth: int, evaluator: EvaluatorWrapper, tt: TranspositionTable = None):
    best_score = -float("inf")
    best_move = None
    alpha = -1e9
    beta  = 1e9

    tt_move = None
    if tt:
        key = tt.get_key(board)
        if key in tt.table:
            tt_move = tt.table[key].get('best_move')

    moves = order_moves(board, list(board.legal_moves), tt_move)

    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, evaluator, tt)
        board.pop()

        print("Root check:", move, score)

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

    # Print statistics
    if tt:
        print(f"TT hit rate: {tt.get_hit_rate():.1f}% ({tt.hits}/{tt.hits + tt.misses})")
    if hasattr(evaluator, 'get_cache_stats'):
        print(f"Eval cache hit rate: {evaluator.get_cache_stats():.1f}%")

    return best_move, best_score


if __name__ == "__main__":
    evaluator = EvaluatorWrapper("best_model.pt")
    tt = TranspositionTable()

    board = chess.Board()
    print(board)

    depth = 4
    best_move, best_score = negamax_root(board, depth, evaluator, tt)

    print(f"Best Move: {best_move}, Score: {best_score}")
