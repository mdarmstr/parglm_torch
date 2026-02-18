import numpy as np
import torch
from parglm_torch.create_design import create_design

def parglm_test_grid():

    reps = 5
    vars = 400

    A = [0, 1, 2, 3]
    B = [0, 1, 2]

    # C levels defined locally inside each A level
    C_local = [0, 1]

    grid = [
        # X = A + B (fixed)
        dict(
            levels=[A, B],
            nested=None,
            Replicates=reps,
            vars=vars,
            parglm=dict(
                Model="linear",
                Preprocessing=2,
                Ts=1,
                Random=[0, 0],
                Ordinal=[0, 0],
                Coding=[0, 0],
                Nested=[],
            ),
        ),

        # X = A + B (A fixed, B random)
        dict(
            levels=[A, B],
            nested=None,
            Replicates=reps,
            vars=vars,
            parglm=dict(
                Model="linear",
                Preprocessing=2,
                Ts=2,
                Random=[0, 1],
                Ordinal=[0, 0],
                Coding=[0, 0],
                Nested=[],
            ),
        ),

        # X = A + B + AB (A fixed, B fixed)
        dict(
            levels=[A, B],
            nested=None,
            Replicates=reps,
            vars=vars,
            parglm=dict(
                Model=[[0, 1]],
                Preprocessing=2,
                Ts=1,
                Random=[0, 0],
                Ordinal=[0, 0],
                Coding=[0, 0],
                Nested=[],
            ),
        ),

        # X = A + B + AB (A fixed, B random)
        dict(
            levels=[A, B],
            nested=None,
            Replicates=reps,
            vars=vars,
            parglm=dict(
                Model=[[0, 1]],
                Preprocessing=2,
                Ts=2,
                Random=[0, 1],
                Ordinal=[0, 0],
                Coding=[0, 0],
                Nested=[],
            ),
        ),

        # X = A + B + AB + C(A)
        # (A fixed, B fixed, C(A) random)
        dict(
            levels=[A, B, C_local],
            nested={"parent": 0, "child": 2},
            Replicates=reps,
            vars=vars,
            parglm=dict(
                Model=[[0, 1]],
                Preprocessing=2,
                Ts=2,
                Random=[0, 0, 1],
                Ordinal=[0, 0, 0],
                Coding=[0, 0, 0],
                Nested=[[0, 2]],
            ),
        ),

        # X = A + B + AB + C(A)
        # (A fixed, B random, C(A) random)
        dict(
            levels=[A, B, C_local],
            nested={"parent": 0, "child": 2},
            Replicates=reps,
            vars=vars,
            parglm=dict(
                Model=[[0, 1]],
                Preprocessing=2,
                Ts=2,
                Random=[0, 1, 1],
                Ordinal=[0, 0, 0],
                Coding=[0, 0, 0],
                Nested=[[0, 2]],
            ),
        ),

    ]

    return grid

def materialize_case(case, seed=123, device="cpu"):

    F = create_design(
        case["levels"],
        Replicates=case["Replicates"],
        nested=case["nested"],
    )

    torch.manual_seed(seed)

    X = torch.randn(
        (F.shape[0], case["vars"]),
        dtype=torch.float64,
        device=device,
    )

    kwargs = case["parglm"].copy()

    kwargs["Permutations"] = 1

    kwargs["device"] = device

    return X, F, kwargs