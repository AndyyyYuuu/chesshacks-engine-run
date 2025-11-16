import torch
import chess

def encode(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros((19, 8, 8), dtype=torch.float32)

    mapping = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for sq, piece in board.piece_map().items():
        idx = mapping[piece.piece_type] + (0 if piece.color == chess.WHITE else 6)
        planes[idx, sq // 8, sq % 8] = 1.0

    # Turn plane
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights
    planes[13][:] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14][:] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15][:] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16][:] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # --- 1 phase plane ---
    phase = min(board.fullmove_number / 100, 1.0)
    planes[17][:] = phase

    # --- 1 en passant plane ---
    if board.ep_square is not None:
        r = board.ep_square // 8
        c = board.ep_square % 8
        planes[18][r][c] = 1.0

    return planes.unsqueeze(0) # shape: (1, 19, 8, 8)


