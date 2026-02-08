import numpy as np
import torch
from parglm_torch.simuleMV import simuleMV
from parglm_torch.parglm import parglm

def create_nested_design_two_nested_factors(
    A_levels=(0, 1, 2),
    B_levels_per_A=None,
    C_levels_per_B=None,
    Replicates=4,
    seed=123,
):
    """
    Build F for a nested design with two nested factors:
      B nested in A, and C nested in B

    Factors in F columns:
      col0 = A
      col1 = B
      col2 = C

    Returns:
      F : (N, 3) numpy array
      Nested : [[0,1],[1,2]]  # 0-indexed
      maps : dict with the nesting structure used
    """
    rng = np.random.default_rng(seed)

    A_levels = list(A_levels)

    # Default: different B sets per A
    if B_levels_per_A is None:
        # e.g. A=0 -> B in {0,1}, A=1 -> B in {2,3,4}, A=2 -> B in {5,6}
        B_levels_per_A = {
            0: [0, 1],
            1: [2, 3, 4],
            2: [5, 6],
        }

    # Default: different C sets per B
    if C_levels_per_B is None:
        # e.g. each B has either 2 or 3 C-levels
        C_levels_per_B = {}
        next_c = 0
        for a in A_levels:
            for b in B_levels_per_A[a]:
                n_c = 2 if (b % 2 == 0) else 3
                C_levels_per_B[b] = list(range(next_c, next_c + n_c))
                next_c += n_c

    # Build rows: for each A, for each B in that A, for each C in that B, replicate
    rows = []
    for a in A_levels:
        for b in B_levels_per_A[a]:
            for c in C_levels_per_B[b]:
                rows.extend([[a, b, c]] * Replicates)

    F = np.array(rows, dtype=int)
    Nested = [[0, 1], [1, 2]]  # B in A, C in B (0-indexed)

    maps = {
        "A_levels": A_levels,
        "B_levels_per_A": B_levels_per_A,
        "C_levels_per_B": C_levels_per_B,
    }
    return F, Nested, maps


def simulate_nested_dataset(
    vars=200,
    Replicates=4,
    LevelCorr=5,
    effect_scales=(2.0, 1.5, 1.0),  # magnitude of A, B(A), C(B) effects
    seed=123,
    device="cpu",
):
    """
    Simulate X for the nested design F with two nested factors.
    X = noise + A_effect + B(A)_effect + C(B)_effect

    Uses your simuleMV(obs, vars, LevelCorr=...) for correlated noise.
    """
    # --- design ---
    F, Nested, maps = create_nested_design_two_nested_factors(
        Replicates=Replicates,
        seed=seed,
    )
    N = F.shape[0]

    # --- deterministic RNG for effects ---
    rng = np.random.default_rng(seed + 1)

    # Create one effect vector per level of each factor
    # A effect: per A level
    A_levels = maps["A_levels"]
    A_vec = {a: rng.standard_normal(vars) for a in A_levels}

    # B effect: per B level (nested in A by construction)
    all_B = sorted({b for blist in maps["B_levels_per_A"].values() for b in blist})
    B_vec = {b: rng.standard_normal(vars) for b in all_B}

    # C effect: per C level (nested in B)
    all_C = sorted({c for clist in maps["C_levels_per_B"].values() for c in clist})
    C_vec = {c: rng.standard_normal(vars) for c in all_C}

    sA, sB, sC = effect_scales

    # Build the mean structure row-wise from F
    X_mean = np.zeros((N, vars), dtype=np.float32)
    for i in range(N):
        a, b, c = F[i]
        X_mean[i, :] = (
            sA * A_vec[a] +
            sB * B_vec[b] +
            sC * C_vec[c]
        )

    # Noise (correlated) using your simuleMV
    X_noise = simuleMV(int(N), int(vars), LevelCorr=LevelCorr)  # returns torch tensor

    # Combine
    X = torch.tensor(X_mean, dtype=torch.float32) + X_noise
    X = X.to(device)

    return X, F, Nested


#-------------------------
#Example usage
#-------------------------
X, F, Nested = simulate_nested_dataset(vars=200, Replicates=4, LevelCorr=5, seed=7)
print("X:", X.shape, "F:", F.shape, "Nested:", Nested)
print("Unique A:", np.unique(F[:,0]))
print("Unique B:", len(np.unique(F[:,1])))
print("Unique C:", len(np.unique(F[:,2])))
print(F)

T, parglmo = parglm(
    X,
    F,
    Model="linear",
    Preprocessing=2,
    Permutations=100,
    Ts=2,
    Ordinal=[0,0,0],
    Coding=[0,0,0],          
    Nested=[[0,1],[1,2]], 
    Random = [0,1,1],
    device="cpu"
)

print(T)