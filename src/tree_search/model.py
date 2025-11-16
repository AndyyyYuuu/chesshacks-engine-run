import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out
    

class Evaluator(nn.Module):
    def __init__(self, in_channels=18, channels=128, n_blocks=8):
        super(Evaluator, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(n_blocks)]
        )
        self.conv_out = nn.Conv2d(channels, 1, kernel_size=1)
        self.fc = nn.Linear(8 * 8, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.res_blocks(out)
        out = self.conv_out(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    




    
def clipped_relu(x):
    return torch.clamp(x, 0.0, 1.0)


class NNUE(nn.Module):
    def __init__(self, embed_dim=128, device="cpu"):
        super().__init__()

        self.device = torch.device(device)

        self.PIECE_TYPES = 10     # no kings
        self.SQUARES = 64
        self.KING_BUCKETS = 64

        self.embedding = nn.Embedding(
            num_embeddings=self.PIECE_TYPES * self.SQUARES * self.KING_BUCKETS,
            embedding_dim=embed_dim
        ).to(self.device)     # <<< FULL MPS HERE

        self.l1  = nn.Linear(embed_dim, 32).to(self.device)
        self.l2  = nn.Linear(32, 32).to(self.device)
        self.out = nn.Linear(32, 1).to(self.device)

    # ---------------------------------------------------
    def index(self, piece_type, piece_square, king_square):
        return (
            piece_type * (self.SQUARES * self.KING_BUCKETS)
            + piece_square * self.KING_BUCKETS
            + king_square
        )

    # ---------------------------------------------------
    def encode(self, board):
        indices = []

        king_square = board.king(board.turn)

        for square, piece in board.piece_map().items():

            if piece.piece_type == chess.KING:
                continue

            if piece.color == chess.WHITE:
                ptype = piece.piece_type - 1   # 0–4
            else:
                ptype = 5 + (piece.piece_type - 1)  # 5–9

            idx = self.index(ptype, square, king_square)
            indices.append(idx)

        # <<< CRITICAL: move indices to SAME DEVICE as embedding
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    # ---------------------------------------------------
    def forward(self, board):
        idxs = self.encode(board)         # (N) on MPS

        vecs = self.embedding(idxs)       # (N, embed_dim), STILL on MPS
        acc  = vecs.sum(dim=0, keepdim=True)  # (1, embed_dim)

        x = clipped_relu(acc)
        x = clipped_relu(self.l1(x))
        x = clipped_relu(self.l2(x))
        out = self.out(x)
        return out


# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # model = NNUE()
# # board = chess.Board()
# # print(f"Total parameters: {count_parameters(model):,}")








import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.clamp(out, 0.0, 1.0)  # clipped ReLU

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = torch.clamp(out, 0.0, 1.0)
        return out


# --------------------------------------------------------
# Full Evaluator Network
# --------------------------------------------------------
class Evaluator(nn.Module):
    def __init__(self, in_channels=19, channels=64, n_blocks=4):
        super().__init__()

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(n_blocks)]
        )

        # Output conv to 1 channel
        self.conv_out = nn.Conv2d(channels, 1, kernel_size=1)

        # Global average pooling into a scalar
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = torch.clamp(out, 0.0, 1.0)

        out = self.res_blocks(out)
        out = self.conv_out(out)          # shape: (B, 1, 8, 8)

        # Global average pooling
        out = out.mean(dim=[2, 3])        # (B, 1)

        out = self.fc(out)                # (B, 1)
        return out




# model = Evaluator(in_channels=19)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# board = chess.Board()
# print(f"Total parameters: {count_parameters(model):,}")






#_--------------------------------------------------------




import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import math

# ---------------------------------------------------------------------
# Const maps
# ---------------------------------------------------------------------
PIECE_MAP = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q":10, "k":11,
}

# ---------------------------------------------------------------------
# Flip POV (side-to-move becomes White)
# ---------------------------------------------------------------------
def flip_pov(board: chess.Board):
    if board.turn == chess.WHITE:
        return board.copy()
    return board.mirror()


# ---------------------------------------------------------------------
# Final 117-D tokenizer (vectorized)
# ---------------------------------------------------------------------
@torch.no_grad()
def encode_transformer_117(boards, history_size=8):
    """
    boards: list of chess.Board with most recent first (boards[0] is current)
    returns: tensor (64, 117)
    """

    # ============================================================
    # 1) Pad history (before flipping!)
    # ============================================================
    original = boards[:]        # keep raw boards for repetition detection
    while len(boards) < history_size:
        boards.append(boards[-1])

    original = original[:history_size]
    while len(original) < history_size:
        original.append(original[-1])

    # Now flip all boards for POV
    boards = [flip_pov(b) for b in boards[:history_size]]


    # ============================================================
    # 2) Encode piece history (96 dims)
    # ============================================================
    hist_planes = torch.zeros((history_size, 64, 12), dtype=torch.float32)

    for h, b in enumerate(boards):
        fen = b.board_fen()
        sq = 0
        for ch in fen:
            if ch == "/":
                continue
            if ch.isdigit():
                sq += int(ch)
            else:
                hist_planes[h, sq, PIECE_MAP[ch]] = 1.0
                sq += 1

    # Merge: (history, 64,12) -> (64, 12*history=96)
    hist_planes = hist_planes.transpose(0,1).reshape(64, 12 * history_size)


    # ============================================================
    # 3) En-passant file (8 dims)
    # ============================================================
    b0 = boards[0]
    ep = torch.zeros((64, 8), dtype=torch.float32)
    if b0.ep_square is not None:
        ep_file = chess.square_file(b0.ep_square)
        ep[:, ep_file] = 1.0


    # ============================================================
    # 4) Castling rights (4 dims)
    # ============================================================
    cas = torch.tensor([
        b0.has_kingside_castling_rights(chess.WHITE),
        b0.has_queenside_castling_rights(chess.WHITE),
        b0.has_kingside_castling_rights(chess.BLACK),
        b0.has_queenside_castling_rights(chess.BLACK),
    ], dtype=torch.float32).repeat(64, 1)


    # ============================================================
    # 5) Fifty-move clock (1 dim)
    # ============================================================
    fifty = torch.full((64,1), float(b0.halfmove_clock) / 100.0)


    # ============================================================
    # 6) Repetition vector (8 dims)
    #    MUST ALWAYS BE LENGTH 8 — this was your dimension bug.
    # ============================================================
    fen_list = [b.fen() for b in original[:history_size]]

    # Pad if necessary (very rare but must do!)
    while len(fen_list) < history_size:
        fen_list.append(fen_list[-1])

    rep_vec = torch.tensor(
        [1.0 if fen_list.count(f) > 1 else 0.0 for f in fen_list],
        dtype=torch.float32
    ).view(1, history_size)

    rep = rep_vec.repeat(64, 1)   # (64,8)


    # ============================================================
    # 7) Concatenate into final (64,117)
    # ============================================================
    final = torch.cat([hist_planes, ep, cas, fifty, rep], dim=1)

    return final  # (64,117)



class TinyHistoryTransformer(nn.Module):
    def __init__(self, d_model=128, n_heads=8, mlp_dim=256, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(117, d_model)

        # stack of transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "mlp": nn.Sequential(
                    nn.Linear(d_model, mlp_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(mlp_dim, d_model)
                )
            })
            for _ in range(num_layers)
        ])

        self.value_head = nn.Linear(d_model, 1)

    def forward(self, tokens):
        # tokens: (B,64,117)
        x = self.input_proj(tokens)

        for layer in self.layers:
            # ATTENTION BLOCK
            h = layer["ln1"](x)
            a, _ = layer["attn"](h, h, h)
            x = x + a

            # MLP BLOCK
            h = layer["ln2"](x)
            x = x + layer["mlp"](h)

        # global average pool
        x = x.mean(dim=1)

        return F.tanh(self.value_head(x))



# model = TinyHistoryTransformer()
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Total parameters: {count_parameters(model):,}")
