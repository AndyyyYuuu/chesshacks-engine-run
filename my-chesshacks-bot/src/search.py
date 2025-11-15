import chess
from .model import Evaluator
import torch
from .chess_utils import encode

class EvaluatorWrapper:
    def __init__(self, model_path):
        self.model = Evaluator(in_channels=19, channels=32, n_blocks=8)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def evaluate(self, board):
        with torch.no_grad():
            x = encode(board)  # shape: (1, 19, 8, 8)
            output = self.model(x)  # shape: (1, 1)
            return output.item() * 100  # Scale back to centipawns

def quiescence_search(board, depth, alpha, beta, evaluator):
    if depth <= 0:
        return evaluator.evaluate(board)

    stand_pat = evaluator.evaluate(board)

    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    # IMPORTANT: list() prevents generator corruption
    for move in list(board.legal_moves):
        if not board.is_capture(move):
            continue

        board.push(move)
        score = -quiescence_search(board, depth - 1, -beta, -alpha, evaluator)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def negamax(board, depth, alpha, beta, evaluator):
    if depth == 0 or board.is_game_over():
        return quiescence_search(board, 8, alpha, beta, evaluator)

    max_value = -float("inf")

    for move in list(board.legal_moves):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, evaluator)
        board.pop()

        if score > max_value:
            max_value = score

        if score > alpha:
            alpha = score

        if alpha >= beta:
            break

    return max_value

def negamax_root(board, depth, evaluator):
    best_score = -float("inf")
    best_move = None
    alpha = -1e9
    beta  = 1e9

    for move in list(board.legal_moves):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, evaluator)
        board.pop()

        print("Root check:", move, score)

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

    return best_move, best_score


if __name__ == "__main__":
    evaluator = EvaluatorWrapper("best_model.pt")

    board = chess.Board()
    print(board)

    depth = 4
    best_move, best_score = negamax_root(board, depth, evaluator)

    print(f"Best Move: {best_move}, Score: {best_score}")
