##############################################################
# CHESS ENGINE WITH 8-BOARD HISTORY FOR TRANSFORMER EVALUATION
##############################################################

import chess
import torch
from collections import deque

from .zobrist import zobrist_hash
from .transposition import TranspositionTable, NodeType
from .model import TinyHistoryTransformer, encode_transformer_117

DEBUG = False
def dbg(*args):
    if DEBUG:
        print(*args)

##############################################################
# LOAD MODEL
##############################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TinyHistoryTransformer()
state = torch.load(
    "src/tree_search/attn_model.pt",
    map_location=device
)
model.load_state_dict(state)
model.eval()

##############################################################
# SEARCH CONSTANTS
##############################################################

node_count = 0
TT = TranspositionTable(size_mb=64)

HISTORY = [[0]*64 for _ in range(64)]
KILLERS = [[None, None] for _ in range(128)]

MATE_VALUE = 100_000
DRAW_VALUE = 0

##############################################################
# BOARD HISTORY — ALWAYS SAFE (NEVER UNDERFLOWS)
##############################################################

HISTORY_STACK = deque(maxlen=8)

def reset_history_stack(board: chess.Board):
    """Initialize 8 copies of the board so encoder always has 8 boards."""
    HISTORY_STACK.clear()
    for _ in range(8):
        HISTORY_STACK.appendleft(board.copy())

def push(board, move):
    """Balanced safe push: push board AND history."""
    board.push(move)
    HISTORY_STACK.appendleft(board.copy())

def pop(board):
    """Balanced safe pop: pop board AND pop history."""
    board.pop()
    # NEVER underflows because HISTORY_STACK always has >=1 at root
    if len(HISTORY_STACK) > 1:
        HISTORY_STACK.popleft()

##############################################################
# SEE EVALUATION + MOVE ORDERING
##############################################################

SEE_VALUE = {
    chess.PAWN:   100,
    chess.KNIGHT: 300,
    chess.BISHOP: 325,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

def see(board, move):
    if not board.is_capture(move):
        return 0

    victim = chess.PAWN if board.is_en_passant(move) \
             else board.piece_type_at(move.to_square)

    gain = [SEE_VALUE[victim]]

    side = board.turn
    occ = board.occupied ^ chess.BB_SQUARES[move.from_square]

    def least_val_attacker(side, occ):
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            mask = board.pieces_mask(pt, side) & occ
            if mask:
                for sq in chess.SquareSet(mask):
                    if sq in board.attackers(side, move.to_square):
                        return pt, sq
        return None, None

    depth = 1
    while True:
        side = not side
        pt, sq = least_val_attacker(side, occ)
        if pt is None:
            break
        gain.append(SEE_VALUE[pt] - gain[depth-1])
        occ ^= chess.BB_SQUARES[sq]
        depth += 1
        if gain[-1] < 0:
            break

    for i in range(len(gain)-2, -1, -1):
        gain[i] = max(-gain[i+1], gain[i])
    return gain[0]


def score_capture(board, move):
    v = see(board, move)
    victim = board.piece_type_at(move.to_square)
    if victim is None:
        victim = chess.PAWN
    attacker = board.piece_type_at(move.from_square)

    if v >= 0:
        return 5_000_000 + (victim*10 - attacker)*10 + v
    return 1_000_000 + v


def score_move(board, move, tt_move, depth):
    if move == tt_move:
        return 10_000_000
    if board.is_capture(move):
        return score_capture(board, move)
    if move == KILLERS[depth][0]:
        return 4_000_000
    if move == KILLERS[depth][1]:
        return 3_000_000
    return HISTORY[move.from_square][move.to_square]

##############################################################
# TT SCORE PACKING
##############################################################

def store_score(score, ply):
    if score > MATE_VALUE - 2000:
        return score + ply
    if score < -MATE_VALUE + 2000:
        return score - ply
    return score

def retrieve_score(score, ply):
    if score > MATE_VALUE - 2000:
        return score - ply
    if score < -MATE_VALUE + 2000:
        return score + ply
    return score

##############################################################
# TRANSFORMER EVALUATION (8 BOARD HISTORY)
##############################################################

@torch.no_grad()
def evaluate(board):
    boards = list(HISTORY_STACK)
    tensor = encode_transformer_117(boards, history_size=8).unsqueeze(0)
    val = model(tensor).item()
    return val if board.turn == chess.WHITE else -val

##############################################################
# QUIESCENCE SEARCH
##############################################################

def quiescence_search(board, alpha, beta):
    global node_count
    node_count += 1

    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    for move in board.legal_moves:
        if not board.is_capture(move):
            continue

        push(board, move)
        score = -quiescence_search(board, -beta, -alpha)
        pop(board)

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha

##############################################################
# NEGAMAX
##############################################################

def negamax(board, depth, alpha, beta, ply):
    global node_count
    node_count += 1

    key = zobrist_hash(board)
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

    if board.is_checkmate():
        return -MATE_VALUE + ply
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return DRAW_VALUE

    if depth == 0:
        #return quiescence_search(board, alpha, beta)
        return evaluate(board)

    best_val = -float("inf")
    original_alpha = alpha
    best_move = None

    moves = list(board.legal_moves)
    moves.sort(key=lambda m: score_move(board, m, tt_move, ply), reverse=True)

    for move in moves:
        push(board, move)
        score = -negamax(board, depth-1, -beta, -alpha, ply+1)
        pop(board)

        if score > best_val:
            best_val = score
            best_move = move
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if not board.is_capture(move):
                if KILLERS[ply][0] != move:
                    KILLERS[ply][1] = KILLERS[ply][0]
                    KILLERS[ply][0] = move
                HISTORY[move.from_square][move.to_square] += depth * depth
            break

    if best_val <= original_alpha:
        flag = NodeType.UPPER
    elif best_val >= beta:
        flag = NodeType.LOWER
    else:
        flag = NodeType.EXACT

    TT.store(key, depth, store_score(best_val, ply), flag, best_move)
    return best_val

##############################################################
# ITERATIVE DEEPENING
##############################################################

def search_with_iterative_deepening(board, max_depth):
    reset_history_stack(board)

    best_move = None
    best_score = None

    for depth in range(1, max_depth+1):
        entry = TT.probe(zobrist_hash(board))
        tt_move = entry.best_move if entry else None

        moves = list(board.legal_moves)
        moves.sort(key=lambda m: score_move(board, m, tt_move, 0), reverse=True)

        alpha = -float("inf")
        beta = float("inf")
        local_best = None

        for move in moves:
            push(board, move)
            score = -negamax(board, depth-1, -beta, -alpha, 1)
            pop(board)

            if score > alpha:
                alpha = score
                local_best = move

        best_move = local_best
        best_score = alpha

        print(f"Depth {depth}: best={best_move}, score={best_score:.4f}")

    return best_move, best_score

##############################################################
# END OF FILE
##############################################################
