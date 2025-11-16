from .utils import chess_manager, GameContext
from chess import Move, Board
import random
import time
import torch
from .chessformer.use_chessformer import get_dist
from .chessformer.model import ChessFormer, ChessFormerInputTokenizer

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis

model = ChessFormer(384, 8, 8, 512)
model.load_state_dict(torch.load("src/chessformer/chessformer_large.pt", map_location="cpu"))
model.eval()

tokenizer = ChessFormerInputTokenizer()

@chess_manager.entrypoint
def get_move(ctx: GameContext) -> Move:
    #return Move.from_uci("e7e5")
    move_probs = get_dist(ctx.board, model, tokenizer)
    ctx.logProbabilities({Move.from_uci(m):v for m,v in move_probs.items()})
    return Move.from_uci(max(move_probs, key=move_probs.get))

@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    pass
