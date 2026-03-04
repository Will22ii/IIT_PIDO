# MODELER/executor/splitter.py

from sklearn.model_selection import KFold


class FixedKFoldSplitter:
    """
    Fixed internal splitter for Modeler.

    Policy:
    - K-Fold with n_splits = 5 (default)
    - repeated K-Fold with n_repeats = 1 (default)
    - shuffle = True for each repeat
    - base_random_seed is provided externally (CAE input)
    - run_id is used to shift model random seed
    """

    def __init__(
        self,
        base_random_seed: int,
        n_splits: int = 5,
        n_repeats: int = 1,
    ):
        self.base_random_seed = base_random_seed
        self.n_splits = max(int(n_splits), 2)
        self.n_repeats = max(int(n_repeats), 1)


    def split(self, X):
        """
        Yield (run_id, train_idx, valid_idx)
        """
        run_id = 0
        for repeat_id in range(self.n_repeats):
            # Use a deterministic repeat-specific seed.
            split_seed = int(self.base_random_seed + repeat_id)
            kf = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=split_seed,
            )
            for train_idx, valid_idx in kf.split(X):
                yield run_id, train_idx, valid_idx
                run_id += 1

    def get_model_seed(self, run_id: int) -> int:
        """
        Derive model-level random seed from base seed.
        """
        return self.base_random_seed + (run_id + 1)

    def get_repeat_id(self, run_id: int) -> int:
        return int(run_id) // int(self.n_splits)

    def get_fold_id(self, run_id: int) -> int:
        return int(run_id) % int(self.n_splits)



