# MODELER/executor/splitter.py

import numpy as np
from sklearn.model_selection import KFold


class FixedKFoldSplitter:
    """
    Fixed internal splitter for Modeler.

    Policy:
    - K-Fold with n_splits = 5
    - shuffle = True
    - base_random_seed is provided externally (CAE input)
    - run_id is used to shift model random seed
    """

    def __init__(self, base_random_seed: int, n_splits: int = 5):
        self.base_random_seed = base_random_seed
        self.n_splits = n_splits

        self.kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.base_random_seed,
        )


    def split(self, X):
        """
        Yield (run_id, train_idx, valid_idx)
        """
        for run_id, (train_idx, valid_idx) in enumerate(self.kf.split(X)):
            yield run_id, train_idx, valid_idx

    def get_model_seed(self, run_id: int) -> int:
        """
        Derive model-level random seed from base seed.
        """
        return self.base_random_seed + (run_id + 1)



