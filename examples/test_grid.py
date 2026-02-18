import os
import numpy as np
import torch
from scipy.io import savemat
import pandas as pd

from parglm_torch.parglm import parglm
from parglm_torch.test_space import parglm_test_grid, materialize_case

outdir = "parglm_test_cases"
os.makedirs(outdir, exist_ok=True)
grid = parglm_test_grid()

for i, case in enumerate(grid):

    print(f"Generating case {i}")
    X, F, kwargs = materialize_case(case, seed=123)

    # convert to numpy for MATLAB
    X_np = X.cpu().numpy()
    F_np = F.astype(np.int32)

    # kwargs as string
    kwargs_str = str(kwargs)

    filename = os.path.join(outdir, f"case_{i}.mat")

    T, _ = parglm(X_np,F_np,**kwargs)
    F_ratios = T["F"].dropna().to_numpy(dtype=float)

    print(F_ratios)

    savemat(
        filename,
        {
            "X": X_np,
            "F": F_np,
            "F_ratios": F_ratios,
            "kwargs_str": kwargs_str,
            "case_index": i,
        },
    )

    print("Saved:", filename)

