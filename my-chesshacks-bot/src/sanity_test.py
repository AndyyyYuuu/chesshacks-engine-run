from chess import Move, Board
import random
import time
from .search import negamax_root, EvaluatorWrapper

evaluator = EvaluatorWrapper("best_model.pt")
move, score = negamax_root(ctx.board, 3, evaluator)
print(move)