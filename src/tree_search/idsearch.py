import chess
import time
import torch
from .zobrist import zobrist_hash
from .transposition import TranspositionTable, NodeType
from .chess_utils import encode
from .model import Evaluator, TinyHistoryTransformer

global node_count
node_count = 0
model = TinyHistoryTransformer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("src/tree_search/attn_model.pt", map_location=device))
model.eval()

HISTORY = [[0] * 64 for _ in range(64)]
KILLERS = [[None, None] for _ in range(128)] 

MVV_LVA = [
    # attacker:   P    N    B    R    Q    K
    [105, 205, 305, 405, 505, 605],  # victim pawn
    [104, 204, 304, 404, 504, 604],  # victim knight
    [103, 203, 303, 403, 503, 603],  # victim bishop
    [102, 202, 302, 402, 502, 602],  # victim rook
    [101, 201, 301, 401, 501, 601],  # victim queen
    [100, 200, 300, 400, 500, 600],  # victim king (should not happen)
]

SEE_VALUE = {
    chess.PAWN:   100,
    chess.KNIGHT: 300,
    chess.BISHOP: 325,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,  # Used only internally
}


def see(board: chess.Board, move: chess.Move) -> int:
    """
    Static Exchange Evaluation (SEE) using SWAP algorithm.
    Returns: material gain (>0 = good capture, <0 = losing)
    """
    # If it's not a capture, no SEE needed
    if not board.is_capture(move):
        return 0
    
    attackers = board.attackers(board.turn, move.to_square)
    defenders = board.attackers(not board.turn, move.to_square)

    # Attacking piece value
    moving_piece = board.piece_type_at(move.from_square)
    if board.is_en_passant(move):
        victim_piece = chess.PAWN
    else:
        victim_piece = board.piece_type_at(move.to_square)

    gain = []
    side = board.turn
    occupied = board.occupied

    gain.append(SEE_VALUE[victim_piece])

    # Remove attacker from occupied mask for later passes
    from_sq = move.from_square

    occ = occupied ^ chess.BB_SQUARES[from_sq]

    # Attacker list (buckets by piece value)
    # This is SWAP: simulate next-least-valuable recapture
    def least_valuable_attacker(side, occ):
        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            pieces = board.pieces_mask(piece_type, side) & occ
            if pieces:
                # Among attackers, pick any (python-chess handles actual legality)
                for sq in chess.SquareSet(pieces):
                    if sq in board.attackers(side, move.to_square):
                        return piece_type, sq
        return None, None

    # side flips each round
    depth = 1
    while True:
        side = not side
        piece_type, sq = least_valuable_attacker(side, occ)
        if piece_type is None:
            break

        gain.append(SEE_VALUE[piece_type] - gain[depth - 1])
        occ ^= chess.BB_SQUARES[sq]
        depth += 1

        # stop early: losing recapture
        if gain[depth - 1] < 0:
            break

    # propagate best choices backward
    for i in range(len(gain) - 2, -1, -1):
        gain[i] = max(-gain[i + 1], gain[i])

    return gain[0]


def score_capture(board, move):
    v = see(board, move)
    
    if v >= 0:
        # Winning captures: high score + MVV-LVA tie-break
        victim = board.piece_type_at(move.to_square)
        attacker = board.piece_type_at(move.from_square)
        if victim is None:
            victim = chess.PAWN  # en passant
        return 5_000_000 + (victim * 10 - attacker) * 10 + v
    else:
        # Losing captures: still searched but late
        return 1_000_000 + v  # v is negative



def score_move(board, move, tt_move, depth):
    
    # 1. TT move
    if move == tt_move:
        return 10_000_000
    
    # Captures: SEE-based
    if board.is_capture(move):
        return score_capture(board, move)

    # 3. Killers
    if move == KILLERS[depth][0]:
        return 4_000_000
    if move == KILLERS[depth][1]:
        return 3_000_000

    # 4. History heuristic (for quiet moves)
    return HISTORY[move.from_square][move.to_square]



TT = TranspositionTable(size_mb=64)
MATE_VALUE = 100000
DRAW_VALUE = 0

def store_score(score, ply):
    """Convert score before storing into TT (mate distance normalization)."""
    if score > MATE_VALUE - 2000:
        return score + ply
    if score < -MATE_VALUE + 2000:
        return score - ply
    return score


def retrieve_score(score, ply):
    """Convert score back after retrieving from TT."""
    if score > MATE_VALUE - 2000:
        return score - ply
    if score < -MATE_VALUE + 2000:
        return score + ply
    return score

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# model = Evaluator(in_channels=19)
# state = torch.load("model.pt", map_location="cpu")
# model.load_state_dict(state)
# model.eval()



def evaluate(board):
    # material = 0
    # for piece_type in PIECE_VALUES:
    #     material += len(board.pieces(piece_type, chess.WHITE)) * PIECE_VALUES[piece_type]
    #     material -= len(board.pieces(piece_type, chess.BLACK)) * PIECE_VALUES[piece_type]
    # return material
    val = model.forward(encode(board)).item()
    return val if board.turn == chess.WHITE else -val


def negamax(board, depth, alpha, beta, ply, eval_fn):
    global node_count
    node_count += 1

    key = zobrist_hash(board)

    # Transposition Table Lookup
    entry = TT.probe(key)
    tt_move = None
    if entry and entry.depth >= depth:
        score = retrieve_score(entry.score, ply)
        if entry.flag == NodeType.EXACT:
            return score
        if entry.flag == NodeType.LOWER:
            alpha = max(alpha, score)
        elif entry.flag == NodeType.UPPER:
            beta = min(beta, score)
        if alpha >= beta:
            return score
        tt_move = entry.best_move
    
    # Terminal Positions
    if board.is_checkmate():
        return -MATE_VALUE + ply
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return DRAW_VALUE
    
    # If Leaf -> Quiescence Search
    if depth == 0:
        return quiescence_search(board, alpha, beta, eval_fn)
        # return evaluate(board)
    
    best_val = -float("inf")
    best_move = None
    original_alpha = alpha

    moves = list(board.legal_moves)
    moves.sort(
        key=lambda m: score_move(board, m, tt_move, ply),
        reverse=True
    )

    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, eval_fn)
        board.pop()

        if score > best_val:
            best_val = score
            best_move = move

        if best_val > alpha:
            alpha = best_val

        # --- Beta cutoff ---
        if alpha >= beta:
            if not board.is_capture(move):
                if KILLERS[ply][0] != move:
                    KILLERS[ply][1] = KILLERS[ply][0]
                    KILLERS[ply][0] = move
                HISTORY[move.from_square][move.to_square] += depth * depth
            break
    
    # --- Store in TT ---
    if best_val <= original_alpha:
        flag = NodeType.UPPER
    elif best_val >= beta:
        flag = NodeType.LOWER
    else:
        flag = NodeType.EXACT

    TT.store(
        key,
        depth,
        store_score(best_val, ply),
        flag,
        best_move
    )

    return best_val


def search_with_iterative_deepening(board, max_depth, eval_fn):
    best_move = None
    best_score = None

    for depth in range(1, max_depth + 1):
        key = zobrist_hash(board)
        entry = TT.probe(key)
        tt_move = entry.best_move if entry else None

        moves = list(board.legal_moves)
        moves.sort(key=lambda m: score_move(board, m, tt_move, 0), reverse=True)

        alpha = -float("inf")
        beta = float("inf")
        local_best = None

        for move in moves:
            board.push(move)
            score = -negamax(board, depth - 1, -beta, -alpha, 1, eval_fn)
            board.pop()

            if score > alpha:
                alpha = score
                local_best = move

        best_move = local_best
        best_score = alpha

        print(f"Depth {depth}: best {best_move} score {best_score}")

    return best_move, best_score



def quiescence_search(board, alpha, beta, eval_fn):
    global node_count
    node_count += 1
    static_eval = eval_fn(board)
    #print("Eval at leaf:", board.fen(), static_eval)

    best_value = static_eval
    if not board.is_check():
        if best_value >= beta:
            return beta
        if best_value > alpha:
            alpha = best_value
    else:
        alpha = max(alpha, static_eval)

    for move in board.legal_moves:
        if not board.is_capture(move):
            continue

        board.push(move)
        score = -quiescence_search(board, -beta, -alpha, eval_fn)
        board.pop()

        if score >= beta:
            return beta
        if score > best_value:
            best_value = score
        if score > alpha:
            alpha = score

    return best_value
