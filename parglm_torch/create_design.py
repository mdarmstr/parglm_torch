import numpy as np

def create_design(levels, Replicates=1):
    """
    Creates a design matrix F given levels and number of replicates.
    """
    import itertools
    factor_levels = [np.array(lvls) for lvls in levels]
    all_combinations = list(itertools.product(*factor_levels))
    F = []
    for comb in all_combinations:
        F.extend([comb] * Replicates)
    F = np.array(F)
    return F