import chess
import random

# 12 piece types: WP,W N,WB,WR,WQ,WK, BP,BN,BB,BR,BQ,BK
ZOBRIST_PIECES = [[random.getrandbits(64) for _ in range(12)] for _ in range(64)]

# 4 castling rights: white K, white Q, black K, black Q
ZOBRIST_CASTLING = [random.getrandbits(64) for _ in range(4)]

# en-passant file
ZOBRIST_EP = [random.getrandbits(64) for _ in range(8)]

# side to move
ZOBRIST_TURN = random.getrandbits(64)

def zobrist_hash(board: chess.Board):
    h = 0

    # pieces
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = (piece.color * 6) + (piece.piece_type - 1)
            h ^= ZOBRIST_PIECES[square][idx]

    # castling
    if board.has_kingside_castling_rights(chess.WHITE):
        h ^= ZOBRIST_CASTLING[0]
    if board.has_queenside_castling_rights(chess.WHITE):
        h ^= ZOBRIST_CASTLING[1]
    if board.has_kingside_castling_rights(chess.BLACK):
        h ^= ZOBRIST_CASTLING[2]
    if board.has_queenside_castling_rights(chess.BLACK):
        h ^= ZOBRIST_CASTLING[3]

    # ----- CORRECT EP HASH -----
    ep = board.ep_square
    if ep is not None:
        file = chess.square_file(ep)

        # Side to move can capture EP
        us = board.turn

        if us == chess.WHITE:
            # white pawns must be on rank 4 & adjacent file
            for df in (-1, +1):
                sq = ep - 8 + df  # pawn must come from rank 5
                if 0 <= sq < 64 and chess.square_rank(sq) == 4:
                    if board.piece_type_at(sq) == chess.PAWN and board.color_at(sq) == chess.WHITE:
                        h ^= ZOBRIST_EP[file]
                        break

        else:
            # black pawns on rank 3 & adjacent file
            for df in (-1, +1):
                sq = ep + 8 + df  # pawn comes from rank 4
                if 0 <= sq < 64 and chess.square_rank(sq) == 3:
                    if board.piece_type_at(sq) == chess.PAWN and board.color_at(sq) == chess.BLACK:
                        h ^= ZOBRIST_EP[file]
                        break
    # -----------------------------------

    if board.turn == chess.BLACK:
        h ^= ZOBRIST_TURN

    return h & 0xFFFFFFFFFFFFFFFF




