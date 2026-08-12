from __future__ import annotations

import json
import os
from typing import List


class ProvenanceLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._file = open(path, "a")

    def log(
        self,
        global_step: int,
        epoch: int,
        batch_idx: int,
        idx: List[int],
        smiles: List[str],
    ) -> None:
        row = {
            "global_step": global_step,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "idx": list(idx),
            "smiles": list(smiles),
        }
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
