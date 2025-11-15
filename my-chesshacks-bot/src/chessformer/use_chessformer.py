import torch
import chess

from .model import ChessFormer, ChessFormerInputTokenizer, flip_board

def move_to_index(move: chess.Move):
    return move.from_square * 64 + move.to_square

def index_to_move(idx: int):
    return idx // 64, idx % 64

PROMO_MAP = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

def sample_from_model(board, model, tokenizer, temperature=1.0, top_k=None):
    """
    Given a python-chess Board, return a chess.Move chosen by your model.
    Always returns a **legal move**.
    """

    # Build 8-history window
    history = [board.copy()]
    node = board.copy()
    while node.move_stack and len(history) < 8:
        node.pop()
        history.append(node.copy())

    history = list(reversed(history))
    while len(history) < 8:
        history.append(history[-1])

    # Flip so side-to-move = white
    flipped_history = [flip_board(b) for b in history]

    # Tokenize → (1,64,117)
    tokens = tokenizer.encode(flipped_history).unsqueeze(0)

    # Inference
    with torch.no_grad():
        move_logits, promo_logits = model(tokens)
        move_logits = move_logits[0]  # shape (4096,)
        promo_logits = promo_logits[0]  # shape (64,4)

    # Mask illegal moves
    legal = list(board.legal_moves)
    legal_ids = []
    for m in legal:
        legal_ids.append(move_to_index(m))

    logits = move_logits[legal_ids]

    # Top-k
    if top_k is not None and len(logits) > top_k:
        values, idxs = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float('-inf'))
        for i, v in zip(idxs, values):
            mask[i] = v
        logits = mask

    # Softmax
    probs = torch.softmax(logits / temperature, dim=0)

    # Sample move
    choice = torch.multinomial(probs, 1).item()
    chosen_id = legal_ids[choice]

    # Rebuild move
    fs, ts = index_to_move(chosen_id)
    move = chess.Move(fs, ts)

    # Promotion handling
    piece = board.piece_at(fs)
    if piece and piece.piece_type == chess.PAWN:
        rank = chess.square_rank(ts)
        if rank == 7:  # promotion rank for white after flip
            promo_probs = torch.softmax(promo_logits[fs] / temperature, dim=0)
            promo_idx = torch.multinomial(promo_probs, 1).item()
            move.promotion = PROMO_MAP[promo_idx]

    return move

def get_dist(board, model, tokenizer, temperature=1.0, top_k=None):
    # Build 8-history window
    history = [board.copy()]
    node = board.copy()
    while node.move_stack and len(history) < 8:
        node.pop()
        history.append(node.copy())

    history = list(reversed(history))
    while len(history) < 8:
        history.append(history[-1])

    # Flip so side-to-move = white
    flipped_history = [flip_board(b) for b in history]

    # Tokenize -> (1,64,117)
    tokens = tokenizer.encode(flipped_history).unsqueeze(0)

    # Inference
    with torch.no_grad():
        move_logits, promo_logits = model(tokens)
        move_logits = move_logits[0]  # shape (4096,)
        promo_logits = promo_logits[0]  # shape (64,4)

    # Mask illegal moves
    legal = list(board.legal_moves)
    
    # Group moves by (from_square, to_square) since promotions share the same index
    move_groups = {}
    for m in legal:
        idx = move_to_index(m)
        if idx not in move_groups:
            move_groups[idx] = []
        move_groups[idx].append(m)
    
    # Get unique indices and their logits
    unique_ids = list(move_groups.keys())
    logits = move_logits[unique_ids]

    # Top-k
    if top_k is not None and len(logits) > top_k:
        values, idxs = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float('-inf'))
        for i, v in zip(idxs, values):
            mask[i] = v
        logits = mask

    # Softmax
    probs = torch.softmax(logits / temperature, dim=0)
    
    # Build dictionary of UCI: prob
    uci_probs = {}
    
    for i, idx in enumerate(unique_ids):
        moves = move_groups[idx]
        move_prob = probs[i].item()
        
        # Get from_square and to_square from first move (all moves in group share these)
        fs = moves[0].from_square
        ts = moves[0].to_square
        
        # Check if this is a promotion move
        piece = board.piece_at(fs)
        if piece and piece.piece_type == chess.PAWN:
            rank = chess.square_rank(ts)
            if rank == 7:  # promotion rank for white after flip
                # Get promotion probabilities
                promo_probs = torch.softmax(promo_logits[fs] / temperature, dim=0)
                
                # Split the move probability across the 4 promotion types
                for promo_idx, promo_piece in enumerate(PROMO_MAP):
                    promo_move = chess.Move(fs, ts, promotion=promo_piece)
                    # Only include if it's actually a legal move
                    if promo_move in legal:
                        promo_prob = promo_probs[promo_idx].item()
                        uci_probs[promo_move.uci()] = move_prob * promo_prob
            else:
                # Regular pawn move, no promotion
                # Should only be one move in the group
                uci_probs[moves[0].uci()] = move_prob
        else:
            # Non-pawn move
            # Should only be one move in the group
            uci_probs[moves[0].uci()] = move_prob
    
    return uci_probs

model = ChessFormer()
model.load_state_dict(torch.load("src/chessformer/chessformer_policy.pt", map_location="cpu"))
model.eval()

tokenizer = ChessFormerInputTokenizer()
