import numpy as np
import itertools

def create_design(levels, Replicates=1, nested=None):
    """
    Create crossed design matrix F.

    Optional nested factor: child nested in parent.

    nested example:
        nested = {"parent": 0, "child": 2}

    Child levels are treated as LOCAL levels inside each parent,
    and automatically recoded to unique global ids.
    """
    factor_levels = [np.array(lvls) for lvls in levels]

    combos = list(itertools.product(*factor_levels))

    F = np.repeat(np.array(combos, dtype=int), Replicates, axis=0)

    if nested is not None:

        parent = nested["parent"]
        child = nested["child"]

        parent_levels = list(levels[parent])
        child_levels = list(levels[child])

        parent_index = {v: i for i, v in enumerate(parent_levels)}
        child_index = {v: i for i, v in enumerate(child_levels)}

        n_child = len(child_levels)

        for i in range(F.shape[0]):

            p = F[i, parent]
            c = F[i, child]

            pi = parent_index[p]
            ci = child_index[c]

            F[i, child] = pi * n_child + ci

    return F