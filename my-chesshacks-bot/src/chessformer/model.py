import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import math

# =======================================================================
# Legal Move Mask Generator
# =======================================================================

def make_move_mask(board: chess.Board):
    mask = torch.zeros(64, 64, dtype=torch.float32)
    for move in board.legal_moves:
        mask[move.from_square, move.to_square] = 1.0
    return mask

# =======================================================================
# Board Flip (side-to-move always white)
# =======================================================================

def flip_board(board):
    if board.turn == chess.WHITE:
        return board.copy()
    else:
        return board.mirror()


# =======================================================================
# Mish Activation
# =======================================================================

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))



# =======================================================================
# Bias-Free Multihead Attention (with dropout)
# =======================================================================

class AttentionNoBias(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0

        self.head_dim = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.W_Q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (Q @ K.transpose(-2, -1)) * self.scale
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ V).transpose(1,2).reshape(B,T,C)
        return self.out_proj(out)



# =======================================================================
# DeepNorm Transformer Encoder Layer (Post-LN)
# =======================================================================

class DeepNormEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffd_dim, alpha, dropout=0.1):
        super().__init__()
        self.alpha = alpha

        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.attn = AttentionNoBias(d_model, n_heads, dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ffd_dim, bias=True),
            Mish(),
            nn.Dropout(dropout),
            nn.Linear(ffd_dim, d_model, bias=True)
        )

    def forward(self, x):
        h = self.ln1(x)
        x = x + self.alpha * self.attn(h)

        h = self.ln2(x)
        x = x + self.alpha * self.ff(h)

        return x



# =======================================================================
# ChessFormer Tokenizer 117-dim
# =======================================================================

class ChessFormerInputTokenizer:
    def __init__(self, history_size=8):
        self.history_size = history_size
        self.piece_to_idx = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }

    def piece_one_hot(self, piece):
        v = torch.zeros(12)
        if piece is None:
            return v
        base = self.piece_to_idx[piece.piece_type]
        if piece.color == chess.BLACK:
            base += 6
        v[base] = 1.0
        return v
    
    def encode_single(self, board):
        return torch.stack([self.piece_one_hot(board.piece_at(sq)) for sq in chess.SQUARES])

    def encode(self, boards):
        # Extend to 8 history
        while len(boards) < 8:
            boards.append(boards[-1])

        boards = [flip_board(b) for b in boards[:8]]

        # Piece history
        hist = torch.cat([self.encode_single(b) for b in boards], dim=1)  # (64,96)

        # EP
        ep_vec = torch.zeros(8)
        if boards[0].ep_square is not None:
            ep_vec[chess.square_file(boards[0].ep_square)] = 1.0
        ep = ep_vec.unsqueeze(0).repeat(64,1)

        # Castling
        cas = torch.tensor([
            boards[0].has_kingside_castling_rights(chess.WHITE),
            boards[0].has_queenside_castling_rights(chess.WHITE),
            boards[0].has_kingside_castling_rights(chess.BLACK),
            boards[0].has_queenside_castling_rights(chess.BLACK),
        ], dtype=torch.float).unsqueeze(0).repeat(64,1)

        # Fifty move
        fifty = torch.full((64,1), boards[0].halfmove_clock / 100.0)

        # Repetition flags
        fens = [b.fen() for b in boards]
        rep_vec = torch.tensor([1.0 if fens.count(f) > 1 else 0.0 for f in fens])
        rep = rep_vec.unsqueeze(0).repeat(64,1)

        # Final tensor = 117 dims
        return torch.cat([hist, ep, cas, fifty, rep], dim=1)



# =======================================================================
# Policy Head (Square-to-Square + Promotions)
# =======================================================================

class PolicyHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.move_head = nn.Linear(64 * d_model, 64 * 64)
        self.promo_head = nn.Linear(d_model, 4)

    def forward(self, enc):
        B, _, D = enc.shape

        # Flatten for move head (B, 64*d)
        enc_flat = enc.reshape(B, 64 * D)

        # Move logits (B,4096)
        move_logits = self.move_head(enc_flat)

        # Promotion logits (B,64,4)
        promo_logits = self.promo_head(enc)

        return move_logits, promo_logits





# =======================================================================
# ChessFormer Model
# =======================================================================

class ChessFormer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_layers=8, ffd_dim=256):
        super().__init__()

        self.token_proj = nn.Linear(117, d_model)

        self.add_offsets = nn.Parameter(torch.zeros(n_layers, 64, d_model))
        self.mul_offsets = nn.Parameter(torch.ones(n_layers, 64, d_model))

        alpha = (2 * n_layers) ** 0.25  # DeepNorm

        self.layers = nn.ModuleList([
            DeepNormEncoderLayer(d_model, n_heads, ffd_dim, alpha)
            for _ in range(n_layers)
        ])

        self.policy_head = PolicyHead(d_model)

    def forward(self, tokens):
        x = self.token_proj(tokens)

        for i, layer in enumerate(self.layers):
            x = (x + self.add_offsets[i]) * self.mul_offsets[i]
            x = layer(x)

        return self.policy_head(x)




class PolicyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, move_logits, promo_logits, move_target, promo_target, is_promo):
        # move logits CE
        move_logprobs = F.log_softmax(move_logits, dim=-1)
        L_move = -(move_target * move_logprobs).sum(dim=-1).mean()

        # promo logits CE
        # promo_logits: (B,64,4)
        # promo_target: (B,64,4)
        # is_promo: (B,) boolean
        promo_logprobs = F.log_softmax(promo_logits, dim=-1)
        L_promo = -(promo_target * promo_logprobs).sum(dim=-1).sum(dim=-1)

        # Only count losses where target is a real promotion
        L_promo = (L_promo * is_promo.float()).mean()

        return L_move + L_promo
