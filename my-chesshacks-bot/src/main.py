from .utils import chess_manager, GameContext
from chess import Move, Board
import random
import time
from .search import negamax_root, EvaluatorWrapper, TranspositionTable

# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis
tt = TranspositionTable()
evaluator = EvaluatorWrapper("best_model.pt")


@chess_manager.entrypoint
def get_move(ctx: GameContext) -> Move:
    move, score = negamax_root(ctx.board, 3, evaluator, tt)
    print(move, score)
    return move

def test_func(ctx: GameContext) -> Move:
    print("Cooking move...")
    print(ctx.board.move_stack)
    time.sleep(0.1)

    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        raise ValueError("No legal moves available (i probably lost didn't i)")

    move_weights = [random.random() for _ in legal_moves]
    total_weight = sum(move_weights)
    # Normalize so probabilities sum to 1
    move_probs = {
        move: weight / total_weight
        for move, weight in zip(legal_moves, move_weights)
    }
    ctx.logProbabilities(move_probs)

    return random.choices(legal_moves, weights=move_weights, k=1)[0]


@chess_manager.reset
def reset_func(ctx: GameContext):
    # This gets called when a new game begins
    # Should do things like clear caches, reset model state, etc.
    tt.clear()
