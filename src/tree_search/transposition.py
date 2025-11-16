import enum
from dataclasses import dataclass
import chess

class NodeType(enum.Enum):
    EXACT = 0      # exact evaluation
    LOWER = 1      # alpha bound (fail-high)
    UPPER = 2      # beta bound (fail-low)


@dataclass
class TTEntry:
    key: int          # zobrist hash
    depth: int        # search depth
    score: int        # stored score (may be mate-corrected)
    flag: NodeType    # EXACT / LOWER / UPPER
    best_move: chess.Move # python-chess Move


class TranspositionTable:
    def __init__(self, size_mb=64):
        entries = (size_mb * 1024 * 1024) // 32  # 32 bytes approx per entry
        self.size = 1
        while self.size < entries:
            self.size <<= 1
        self.mask = self.size - 1
        self.table = [None] * self.size

    def clear(self):
        for i in range(self.size):
            self.table[i] = None

    def store(self, key, depth, score, flag, best_move):
        idx = key & self.mask
        entry = self.table[idx]

        if entry is not None and entry.depth > depth:
            return  # Don't overwrite deeper entries

        self.table[idx] = TTEntry(key, depth, score, flag, best_move)

    def probe(self, key):
        idx = key & self.mask
        entry = self.table[idx]
        if entry is None:
            return None

        if entry.key != key:
            return None  # Real collision, reject

        return entry




