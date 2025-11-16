from .utils import chess_manager, GameContext
from chess import Move, Board
import random
import time
#from .tree_search.search import negamax_root, EvaluatorWrapper, TranspositionTable
from .tree_search.search_main import search_with_iterative_deepening
from .tree_search.model import TinyHistoryTransformer, encode_transformer_117
import torch

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis
#tt = TranspositionTable()
# evaluator = EvaluatorWrapper("src/tree_search/best_model_12blks_v1.pt", n_blocks=12)

model = TinyHistoryTransformer()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load("src/tree_search/attn_model.pt", map_location=device))
model.eval()

@chess_manager.entrypoint
def get_move(ctx: GameContext) -> Move:
    #return Move.from_uci("e7e5")
    # move, score = negamax_root(ctx.board, 3, evaluator)
    move, score = search_with_iterative_deepening(ctx.board, 2)
    print(type(move))
    print(f"Move: {move}, Score: {score}")
    return move


@chess_manager.reset
def reset_func(ctx: GameContext):
    #tt.clear()
    pass
