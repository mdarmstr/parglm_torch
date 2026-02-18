import torch
import numpy as np
import pandas as pd
import time

from itertools import combinations, combinations_with_replacement, product
from parglm_torch.simuleMV import *
from scipy.io import savemat

def parglm(X, F, Model='linear', Preprocessing=2, Permutations=1000, Ts=1,
           Ordinal=None, Fmtc=0, Coding=None, Nested=None, Random=None, device='cpu'):
    """

    Parallel General Linear Model to obtain multivariate factor and interaction
    matrices in a crossed experimental design and permutation testing for multivariate
    statistical significance.

    Parameters:
    X : torch.Tensor or numpy.ndarray
        Data matrix of shape (N, M), where each row is a measurement, each column a variable.
    F : torch.Tensor or numpy.ndarray
        Design matrix of shape (N, F), where columns correspond to factors and rows to levels.

    Optional Parameters:
    Model : str or int or list of lists, default='linear'
        Defines the model to be used.
    Preprocessing : int, default=2
        Preprocessing option: 0 (none), 1 (mean-centering), 2 (auto-scaling).
    Permutations : int, default=1000
        Number of permutations for the permutation test.
    Ts : int, default=1
        Test statistic to use: 0 (SSQ), 1 (F-value), 2 (hierarchical F-value).
    Ordinal : list or numpy.ndarray, default=None
        Indicates whether factors are nominal (0) or ordinal (1).
    Fmtc : int, default=0
        Correction for multiple testing: 0 (none), 1 (Bonferroni), 2 (Holm/Hochberg), 3 (FDR), 4 (Q-value).
    Coding : list or numpy.ndarray, default=None
        Type of coding of factors: 0 (sum/deviation), 1 (reference).
    Nested : list of lists, default=None
        Pairs of nested factors, e.g., [[1, 2], [2, 3]] if factor 2 is nested in 1, and 3 in 2.
    Random: list or numpy.ndarray, default=None
        Indicates whether factors are fixed (0) or random (1).

    Returns:
    T : pandas.DataFrame
        ANOVA-like output table.
    parglmo : dict
        Dictionary containing detailed results and matrices.
    """
    # Convert inputs to torch tensors if they aren't already
    if not isinstance(X, torch.Tensor):
        if np.iscomplexobj(X):
           X = torch.tensor(X, dtype=torch.complex64)
        else:
           X = torch.tensor(X, dtype=torch.float)
    if not isinstance(F, torch.Tensor):
        F = torch.tensor(F, dtype=torch.float)

    # Check: if any column in F has only one unique value then return an error.
    for col in range(F.shape[1]):
        unique_vals = torch.unique(F[:, col])
        if unique_vals.numel() == 1:
            raise ValueError(f"Design matrix F has column {col} with a single unique value: {unique_vals.item()}. "
                             "Each factor must have at least two unique values.")

    X = X.to(device)
    F = F.to(device)

    N, M = X.shape
    n_factors = F.shape[1]

    # Handle optional parameters
    if Ordinal is None:
        Ordinal = torch.zeros(n_factors, dtype=torch.int)
    else:
        Ordinal = torch.tensor(Ordinal, dtype=torch.int)
    if Coding is None:
        Coding = torch.zeros(n_factors, dtype=torch.int)
    else:
        Coding = torch.tensor(Coding, dtype=torch.int)
    if Nested is None:
        Nested = []
    else:
        # force list of [parent, child]
        Nested = [list(map(int, pair)) for pair in Nested]

        for pair in Nested:
            if len(pair) != 2:
                raise ValueError("Nested must be a list of [parent, child] pairs")

            parent, child = pair

            if not (0 <= parent < n_factors and 0 <= child < n_factors):
                raise ValueError(
                    f"Nested pair {pair} out of range for {n_factors} factors "
                    "(expected 0-indexed)"
                )

            if parent == child:
                raise ValueError(f"Nested pair {pair} is invalid (parent == child)")
            
    nested_child_to_parent = {child: parent for parent, child in Nested}
    nested_children = set(nested_child_to_parent.keys())

    if Random is None:
        Random = torch.zeros(n_factors, dtype=torch.int64)
    else:
        Random = torch.tensor(Random, dtype=torch.int64)
        if Random.numel() != n_factors:
            raise ValueError("Random must be length n_factors")
    Random = Random.to(device)

    # Determine interactions based on Model
    def allinter(factors, order):
        if order > 2:
            interactions = allinter(factors, order - 1)
            new_interactions = []
            for inter in interactions:
                max_inter = max(inter)
                for j in factors:
                    if j > max_inter:
                        new_interactions.append(inter + [j])
            interactions.extend(new_interactions)
            return interactions
        else:
            interactions = []
            for i in factors:
                for j in factors:
                    if j > i:
                        interactions.append([i, j])
            return interactions

    factors = list(range(n_factors))
    if Nested:
        nested_factors = [pair[1] for pair in Nested]
        factors = [f for f in factors if f not in nested_factors]

    if Model == 'linear':
        interactions = []
    elif Model == 'interaction':
        interactions = allinter(factors, 2)
    elif Model == 'full':
        interactions = allinter(factors, n_factors)
    elif isinstance(Model, int):
        if Model >= 2 and Model <= n_factors:
            interactions = allinter(factors, Model)
        else:
            raise ValueError('Invalid Model parameter')
    elif isinstance(Model, list):
        interactions = Model
    else:
        raise ValueError('Invalid Model parameter')

    n_interactions = len(interactions)

    # Multiple test correction factor
    mtcc = n_factors + n_interactions if Fmtc else 1

    # Initialize result variables
    SSQ_factors = torch.zeros(Permutations * mtcc + 1, n_factors)
    SSQ_interactions = torch.zeros(Permutations * mtcc + 1, n_interactions)
    F_factors = torch.zeros(Permutations * mtcc + 1, n_factors)
    F_interactions = torch.zeros(Permutations * mtcc + 1, n_interactions)
    p_factor = torch.zeros(n_factors)
    p_interaction = torch.zeros(n_interactions)

    # Initialize parglmo dictionary
    parglmo = {}
    parglmo['factors'] = [{} for _ in range(n_factors)]
    parglmo['interactions'] = [{} for _ in range(n_interactions)]
    parglmo['data'] = X.clone()
    parglmo['prep'] = Preprocessing
    parglmo['design'] = F
    parglmo['n_factors'] = n_factors
    parglmo['n_interactions'] = n_interactions
    parglmo['n_perm'] = Permutations
    parglmo['ts'] = Ts
    parglmo['ordinal'] = Ordinal
    parglmo['fmtc'] = Fmtc
    parglmo['coding'] = Coding
    parglmo['nested'] = Nested
    parglmo["random"] = Random

    # Preprocess the data
    def preprocess2D(X, Preprocessing=2):
        if Preprocessing == 0:
            Xs = X.clone()
            m = torch.zeros(X.shape[1])
            dt = torch.ones(X.shape[1])
        elif Preprocessing == 1:
            m = torch.mean(X, dim=0)
            Xs = X - m
            dt = torch.ones(X.shape[1])
        elif Preprocessing == 2:
            m = torch.mean(X, dim=0)
            s = torch.std(X, dim=0, unbiased=False)
            dt = s
            Xs = (X - m) / s
        else:
            raise ValueError('Invalid Preprocessing option')
        return Xs, m, dt

    X, m, dt = preprocess2D(X, Preprocessing=Preprocessing)
    X = X / dt.to(device)  # Scale the data
    parglmo['scale'] = dt

    # Create the Design Matrix D
    D_list = []
    n = 0
    if Preprocessing:
        D_list.append(torch.ones(N, 1, device=device))  
        n += 1

    parglmo['n_levels'] = {}
    for f in range(n_factors):
        parglmo['factors'][f] = {}
        if Ordinal[f] == 1:
            D_col, _, _ = preprocess2D(F[:, f].unsqueeze(1), Preprocessing=1)
            D_list.append(D_col.to(device))
            parglmo['factors'][f]['Dvars'] = [n]
            n += 1
            parglmo['factors'][f]['order'] = 1
            parglmo['factors'][f]['factors'] = []
        else:
            if (not Nested) or (f not in nested_children):

                parglmo['factors'][f]['factors'] = []
                uF = torch.unique(F[:, f], sorted=True)   # global levels
                parglmo['n_levels'][f] = int(uF.numel())

                Dvars = []

                # create L-1 columns, each indicates membership in level i (i>=1)
                for i in range(1, int(uF.numel())):
                    D_col = (F[:, f] == uF[i]).float().unsqueeze(1)   # Nx1
                    D_list.append(D_col.to(device))
                    Dvars.append(n)
                    n += 1

                parglmo['factors'][f]['Dvars'] = Dvars

                # apply baseline coding to rows at uF[0]
                if len(Dvars) > 0:
                    base_mask = (F[:, f] == uF[0])
                    recent_cols = D_list[-len(Dvars):]
                    for ccol in recent_cols:
                        ccol[base_mask] = 0 if Coding[f] == 1 else -1

                parglmo['factors'][f]['order'] = 1
            # nested factor 
            else:
                parent = nested_child_to_parent[f]  # parent factor index

                # store nesting chain
                # child.factors = [parent, parent.factors...]
                parent_chain = parglmo['factors'][parent].get('factors', [])
                parglmo['factors'][f]['factors'] = [parent] + parent_chain

                # order increments
                parent_order = parglmo['factors'][parent].get('order', 1)
                parglmo['factors'][f]['order'] = parent_order + 1

                u_parent = torch.unique(F[:, parent], sorted=True)

                Dvars = []
                n_levels_total = 0  # total number of child levels across blocks

                # For each parent level, create its own local dummy variables for child
                for lvl in u_parent:
                    rind = (F[:, parent] == lvl)              # boolean mask (N,)
                    child_vals = torch.unique(F[rind, f], sorted=True)
                    L = int(child_vals.numel())
                    n_levels_total += L

                    # Build L-1 columns for this parent block
                    cols_this_block = []
                    for i in range(1, L):
                        col = torch.zeros(N, 1, device=device)
                        col[rind] = (F[rind, f] == child_vals[i]).float().unsqueeze(1)
                        D_list.append(col)
                        cols_this_block.append(col)
                        Dvars.append(n)
                        n += 1

                    # Baseline coding within this parent block:
                    # rows with child_vals[0] get -1 (deviation) or 0 (reference)
                    if L > 1:
                        base_mask = rind & (F[:, f] == child_vals[0])
                        for ccol in cols_this_block:
                            ccol[base_mask] = 0 if Coding[f] == 1 else -1

                parglmo['factors'][f]['Dvars'] = Dvars
                parglmo['n_levels'][f] = n_levels_total

    D_list = [d.to(device) for d in D_list]
    D = torch.cat(D_list, dim=1)

    # Function to compute interaction termі
    def computaDint(interactions, factors, D):
        if len(interactions) > 1:
            deepD = computaDint(interactions[1:], factors, D)
            Dout_list = []
            for k in factors[interactions[0]]['Dvars']:
                D_k = D[:, k].unsqueeze(1)
                for l in range(deepD.shape[1]):
                    Dout_col = D_k * deepD[:, l].unsqueeze(1)
                    Dout_list.append(Dout_col)
            Dout = torch.cat(Dout_list, dim=1)
        else:
            Dvars = factors[interactions[0]]['Dvars']
            Dout = D[:, Dvars]
        return Dout

    # Add interactions to the Design Matrix
    for i, interaction in enumerate(interactions):
        Dout = computaDint(interaction, parglmo['factors'], D)
        D = torch.cat((D, Dout), dim=1)
        parglmo['interactions'][i]['Dvars'] = list(range(n, n + Dout.shape[1]))
        parglmo['interactions'][i]['factors'] = interaction
        parglmo['interactions'][i]['type'] = int(torch.sum(Random[torch.tensor(interaction, device=device)] == 1) > 0)
        n = D.shape[1]
        parglmo['interactions'][i]['order'] = max([parglmo['factors'][f]['order'] for f in interaction]) + 1

    # Degrees of freedom
    Tdf = N
    mdf = 1 if Preprocessing else 0
    Rdf = Tdf - mdf
    df = torch.zeros(n_factors, dtype=torch.int64, device=device)
    for f in range(n_factors):
        if Ordinal[f]:
            df[f] = 1
        else:
            df[f] = len(parglmo['factors'][f]['Dvars'])
        Rdf -= df[f]

    df_int = torch.zeros(n_interactions, dtype=torch.int64, device=device)
    for i in range(n_interactions):
        facs = parglmo['interactions'][i]['factors']  # defined in the interaction loop
        df_int[i] = torch.prod(df[torch.tensor(facs, device=device)])
        Rdf -= df_int[i]

    if Rdf < 0:
        print('Warning: degrees of freedom exhausted')
        return

    # DEBUG: nested sanity check (correct)
    # Each nested-design column must only be active (nonzero) within ONE parent level.
    for child, parent in nested_child_to_parent.items():
        child_cols = parglmo['factors'][child]['Dvars']
        if len(child_cols) == 0:
            continue

        for k in child_cols:
            nz = (D[:, k] != 0)
            parent_levels_used = torch.unique(F[nz, parent], sorted=True)
            assert parent_levels_used.numel() == 1, (
                f"Nested column {k} for child factor {child} is active in multiple "
                f"parent levels of factor {parent}: {parent_levels_used.tolist()}"
            )
    print("Nested sanity check passed: each nested column belongs to exactly one parent level.")

    print("\n--- D summary ---")
    print("D shape:", tuple(D.shape))
    for f in range(n_factors):
        dv = parglmo["factors"][f]["Dvars"]
        print(
            f"Factor {f}: #Dvars={len(dv)}, order={parglmo['factors'][f].get('order')}, "
            f"nested_chain={parglmo['factors'][f].get('factors')}"
        )

    A, B = 0, 1
    B_cols = parglmo["factors"][B]["Dvars"]

    print("\n--- Column ownership: B(A) ---")
    for k in B_cols[:min(10, len(B_cols))]:  # print a few
        nz = (D[:, k] != 0)
        owners = torch.unique(F[nz, A]).detach().cpu().numpy().tolist()
        print(f"B col {k}: active A-levels = {owners}")
    
    C = 2
    C_cols = parglmo["factors"][C]["Dvars"]

    print("\n--- Column ownership: C(B) ---")
    for k in C_cols[:min(10, len(C_cols))]:
        nz = (D[:, k] != 0)
        owners = torch.unique(F[nz, B]).detach().cpu().numpy().tolist()
        print(f"C col {k}: active B-levels = {owners}")

    print("\n--- D (nonzero pattern) first 40 rows, first 40 cols ---")
    Dn = (D[:40, :40] != 0).int().detach().cpu().numpy()
    print(Dn)

    # Handle missing data
    Xnan = X.clone()
    nan_mask = torch.isnan(X)
    nan_indices = torch.nonzero(nan_mask)
    rows_with_nan = torch.unique(nan_indices[:, 0])

    for ru in rows_with_nan:
        nan_cols = nan_indices[nan_indices[:, 0] == ru, 1]
        D_row = D[ru, :].unsqueeze(0)
        D_diff = D - D_row.repeat(N, 1)
        sum_D_diff_sq = torch.sum(torch.abs(D_diff) ** 2, dim=1)
        ind2 = torch.nonzero(sum_D_diff_sq == 0).squeeze()
        for c in nan_cols:
            X_ind2_c = X[ind2, c]
            not_nan = ~torch.isnan(X_ind2_c)
            if torch.sum(not_nan) > 0:
                X[ru, c] = torch.mean(X_ind2_c[not_nan])
            else:
                X_c = X[:, c]
                not_nan_c = ~torch.isnan(X_c)
                X[ru, c] = torch.mean(X_c[not_nan_c])

    parglmo['data'] = X
    parglmo['Xnan'] = Xnan

    if X.is_complex():
        SSQ_X = torch.sum(torch.abs(torch.diag(X @ X.T.conj()))).item()
        D = D.to(torch.complex64)
    else:
        SSQ_X = torch.sum(X ** 2).item()

    # GLM model calibration with LS
    pD = torch.pinverse(D.T @ D) @ D.T
    B = pD @ X
    X_residuals = X - D @ B
    parglmo['D'] = D
    parglmo['B'] = B

    # Compute effects and sum of squares
    if Preprocessing:
        parglmo['inter'] = D[:, 0:1] @ B[0:1, :]
        SSQ_inter = torch.sum(torch.abs(parglmo['inter']) ** 2).item()
    else:
        parglmo['inter'] = 0
        SSQ_inter = 0

    if X_residuals.is_complex():
        SSQ_residuals = torch.sum(
            torch.abs(torch.diag(X_residuals @ X_residuals.T.conj()))
        ).item()
    else:
        SSQ_residuals = torch.sum(X_residuals ** 2).item()

    # Factors
    for f in range(n_factors):
        Dvars = parglmo['factors'][f]['Dvars']
        mat = D[:, Dvars] @ B[Dvars, :]
        parglmo['factors'][f]['matrix'] = mat
        SSQ_factors[0, f] = torch.sum(torch.abs(mat) ** 2).item()

    # Interactions
    for i, interaction in enumerate(parglmo['interactions']):
        Dvars = interaction['Dvars']
        mat = D[:, Dvars] @ B[Dvars, :]
        interaction['matrix'] = mat
        SSQ_interactions[0, i] = torch.sum(torch.abs(mat) ** 2).item()


    # Normalize at the final step
    parglmo['effects'] = 100 * np.array(
        [SSQ_inter] + SSQ_factors[0, :].tolist() + SSQ_interactions[0, :].tolist() + [SSQ_residuals]
    ) / (SSQ_X)

    parglmo['residuals'] = X_residuals

    MS_e = SSQ_residuals / Rdf.item()

    # init references
    for f in range(n_factors):
        parglmo['factors'][f]['refF'] = []
        parglmo['factors'][f]['refI'] = []

    for i in range(n_interactions):
        parglmo['interactions'][i]['refI'] = []

    if Ts == 2:
        # Factors: collect random nested factors and random interactions containing factor
        for f in range(n_factors):
            # random nested factors f2 where f is in f2's nesting chain
            for f2 in range(n_factors):
                if int(Random[f2].item()) == 1:
                    chain = parglmo['factors'][f2].get('factors', [])
                    if (f in chain) and (df[f2].item() > 0):
                        MS_f2 = SSQ_factors[0, f2].item() / df[f2].item()
                        if MS_e < MS_f2:
                            parglmo['factors'][f]['refF'].append(f2)

            # random interactions containing f where at least one "rest" factor is random
            for i in range(n_interactions):
                facs = parglmo['interactions'][i]['factors']
                if f in facs and df_int[i].item() > 0:
                    rest = [g for g in facs if g != f]
                    if any(int(Random[g].item()) == 1 for g in rest):
                        MS_i = SSQ_interactions[0, i].item() / df_int[i].item()
                        if MS_e < MS_i:
                            parglmo['factors'][f]['refI'].append(i)

        # Interactions: collect higher-order interactions that contain current interaction
        for i in range(n_interactions):
            facs_i = set(parglmo['interactions'][i]['factors'])
            for i2 in range(n_interactions):
                facs_i2 = set(parglmo['interactions'][i2]['factors'])
                if facs_i.issubset(facs_i2) and (len(facs_i2) > len(facs_i)) and df_int[i2].item() > 0:
                    rest = list(facs_i2 - facs_i)
                    if any(int(Random[g].item()) == 1 for g in rest):
                        MS_i2 = SSQ_interactions[0, i2].item() / df_int[i2].item()
                        if MS_e < MS_i2:
                            parglmo['interactions'][i]['refI'].append(i2)

    # Compute nominal F-values for factors before the loop
    for f in range(n_factors):
        if Ts == 2:
            refF = parglmo['factors'][f].get('refF', [])
            refI = parglmo['factors'][f].get('refI', [])

            SSref = sum(SSQ_factors[0, f2].item() for f2 in refF) + \
                    sum(SSQ_interactions[0, i].item() for i in refI)
            Dfref = sum(df[f2].item() for f2 in refF) + \
                    sum(df_int[i].item() for i in refI)

            if SSref == 0:
                MSref = SSQ_residuals / Rdf.item()
            else:
                MSref = SSref / Dfref

            F_value = (SSQ_factors[0, f].item() / df[f].item()) / MSref
        else:
            F_value = (SSQ_factors[0, f].item() / df[f].item()) / (SSQ_residuals / Rdf.item())

        F_factors[0, f] = F_value

    # Compute nominal F-values for interactions (observed)
    for i in range(n_interactions):
        if Ts == 2:
            refI = parglmo['interactions'][i].get('refI', [])

            SSref = sum(SSQ_interactions[0, i2].item() for i2 in refI)
            Dfref = sum(df_int[i2].item() for i2 in refI)

            if SSref == 0:
                MSref = SSQ_residuals / Rdf.item()
            else:
                MSref = SSref / Dfref

            F_value = (SSQ_interactions[0, i].item() / df_int[i].item()) / MSref
        else:
            F_value = (SSQ_interactions[0, i].item() / df_int[i].item()) / (SSQ_residuals / Rdf.item())

        F_interactions[0, i] = F_value

    # Permutations
    for j in range(1, Permutations * mtcc + 1):
        perms = torch.randperm(N, device=device)
        X_perm = Xnan[perms, :].clone()

        # Handle missing data in permuted X
        nan_mask_perm = torch.isnan(X_perm)
        nan_indices_perm = torch.nonzero(nan_mask_perm)
        rows_with_nan_perm = torch.unique(nan_indices_perm[:, 0])

        for ru in rows_with_nan_perm:
            nan_cols = nan_indices_perm[nan_indices_perm[:, 0] == ru, 1]
            D_row = D[ru, :].unsqueeze(0)
            D_diff = D - D_row.repeat(N, 1)
            sum_D_diff_sq = torch.sum(torch.abs(D_diff) ** 2, dim=1)
            ind2 = torch.nonzero(sum_D_diff_sq == 0).squeeze()
            for c in nan_cols:
                X_ind2_c = X_perm[ind2, c]
                not_nan = ~torch.isnan(X_ind2_c)
                if torch.sum(not_nan) > 0:
                    X_perm[ru, c] = torch.mean(X_ind2_c[not_nan])
                else:
                    X_c = X_perm[:, c]
                    not_nan_c = ~torch.isnan(X_c)
                    X_perm[ru, c] = torch.mean(X_c[not_nan_c])

        B_perm = pD @ X_perm
        X_residuals_perm = X_perm - D @ B_perm
        SSQ_residuals_perm = torch.sum(torch.abs(X_residuals_perm) ** 2).item()

        # Factors
        SSQf = []
        for f in range(n_factors):
            Dvars = parglmo['factors'][f]['Dvars']
            factor_matrix = D[:, Dvars] @ B_perm[Dvars, :]
            SSQf.append(torch.sum(torch.abs(factor_matrix) ** 2).item())
        SSQ_factors[j, :] = torch.tensor(SSQf, device=device, dtype=SSQ_factors.dtype)

        # Interactions
        SSQi = []
        for i_int, interaction in enumerate(parglmo['interactions']):
            Dvars = interaction['Dvars']
            interaction_matrix = D[:, Dvars] @ B_perm[Dvars, :]
            SSQi.append(torch.sum(torch.abs(interaction_matrix) ** 2).item())
        SSQ_interactions[j, :] = torch.tensor(SSQi, device=device, dtype=SSQ_interactions.dtype)

        # F Factors
        Ff = []
        MS_e_perm = SSQ_residuals_perm / Rdf.item()

        for f in range(n_factors):
            if Ts == 2:
                refF = parglmo['factors'][f].get('refF', [])
                refI = parglmo['factors'][f].get('refI', [])

                SSref = 0.0
                Dfref = 0.0

                for f2 in refF:
                    SSref += float(SSQf[f2])
                    Dfref += float(df[f2].item())

                for i2 in refI:
                    SSref += float(SSQi[i2])
                    Dfref += float(df_int[i2].item())

                MSref = (SSref / Dfref) if SSref > 0 else MS_e_perm
                F_value = (float(SSQf[f]) / float(df[f].item())) / MSref
            else:
                F_value = (float(SSQf[f]) / float(df[f].item())) / MS_e_perm

            Ff.append(F_value)

        F_factors[j, :] = torch.tensor(Ff, device=device, dtype=F_factors.dtype)

        # Interactions
        Fi = []
        for i_int in range(n_interactions):
            if Ts == 2:
                refI = parglmo['interactions'][i_int].get('refI', [])

                SSref = 0.0
                Dfref = 0.0

                for i2 in refI:
                    SSref += float(SSQi[i2])
                    Dfref += float(df_int[i2].item())

                MSref = (SSref / Dfref) if SSref > 0 else MS_e_perm
                F_value = (float(SSQi[i_int]) / float(df_int[i_int].item())) / MSref
            else:
                F_value = (float(SSQi[i_int]) / float(df_int[i_int].item())) / MS_e_perm

            Fi.append(F_value)

        F_interactions[j, :] = torch.tensor(Fi, device=device, dtype=F_interactions.dtype)
    # Select test statistic
    ts_factors = F_factors if Ts else SSQ_factors
    ts_interactions = F_interactions if Ts else SSQ_interactions

    # Calculate p-values
    for f in range(n_factors):
        p_factor[f] = (torch.sum(ts_factors[1:, f] >= ts_factors[0, f]) + 1) / (Permutations * mtcc + 1)
    for i in range(n_interactions):
        p_interaction[i] = (torch.sum(ts_interactions[1:, i] >= ts_interactions[0, i]) + 1) / (Permutations * mtcc + 1)

    parglmo['p'] = torch.cat((p_factor, p_interaction)).numpy()

    # Multiple test correction
    if mtcc > 1:
        p_values = parglmo['p']
        if Fmtc == 1:
            parglmo['p'] = np.minimum(1, p_values * mtcc)
        elif Fmtc == 2:
            sorted_indices = np.argsort(p_values)
            for ind in range(mtcc):
                p_values[sorted_indices[ind]] = min(1, p_values[sorted_indices[ind]] * (mtcc - ind))
            parglmo['p'] = p_values
        elif Fmtc == 3:
            sorted_indices = np.argsort(p_values)
            for ind in reversed(range(mtcc)):
                p_values[sorted_indices[ind]] = min(1, p_values[sorted_indices[ind]] * mtcc / (ind + 1))
            parglmo['p'] = p_values
        elif Fmtc == 4:
            sorted_indices = np.argsort(p_values)
            q_values = p_values.copy()
            q_values[sorted_indices[-1]] = p_values[sorted_indices[-1]]
            for ind in reversed(range(mtcc - 1)):
                q_values[sorted_indices[ind]] = min(1, p_values[sorted_indices[ind]] * mtcc / (ind + 1), q_values[sorted_indices[ind + 1]])
            parglmo['p'] = q_values

    # Create ANOVA-like output table
    names = ['Mean'] if Preprocessing else []
    names += [f'Factor {f+1}' for f in range(n_factors)]
    names += [f"Interaction {'-'.join(map(str, [i+1 for i in interaction]))}" for interaction in interactions]
    names += ['Residuals', 'Total']

    SSQ_list = [SSQ_inter] if Preprocessing else []
    SSQ_list += SSQ_factors[0, :].tolist()
    SSQ_list += SSQ_interactions[0, :].tolist()
    SSQ_list += [SSQ_residuals, SSQ_X]

    par_list = parglmo['effects'].tolist() + [100]
    DoF = [mdf] if Preprocessing else []
    DoF += df.tolist()
    
    if len(Fi) > 0:
        DoF.extend([t.item() for t in df_int])
    
    DoF.append(Rdf.item())
    DoF.append(Tdf)
    
    MSQ = [np.abs(s) / d if d != 0 else np.nan for s, d in zip(SSQ_list, DoF)]
    
    F_list = [np.nan]
    F_list += F_factors[0, :].tolist()
    F_list += F_interactions[0, :].tolist()
    F_list += [np.nan, np.nan]
    p_values = [np.nan]
    p_values += parglmo['p'].tolist()
    p_values += [np.nan, np.nan]

    den_ref_col = []

    # Mean row
    if Preprocessing:
        den_ref_col.append(np.nan)

    # Factor rows
    for f in range(n_factors):
        rf = parglmo['factors'][f].get('refF', [])
        ri = parglmo['factors'][f].get('refI', [])

        if (rf or ri):
            den_ref_col.append(f"F{rf};I{ri}")
        else:
            den_ref_col.append("Residuals")  

    # Interaction rows
    for i in range(n_interactions):
        ri = parglmo['interactions'][i].get('refI', [])
        if ri:
            den_ref_col.append(f"I{ri}")
        else:
            den_ref_col.append("Residuals")

    # Residuals + Total
    den_ref_col += [np.nan, np.nan]
    
    data = {
        'Source': names,
        'SumSq': SSQ_list,
        'PercSumSq': par_list,
        'df': DoF,
        'MeanSq': MSQ,
        'F': F_list,
        'Pvalue': p_values,
        'Den_ref': den_ref_col 
    }

    T = pd.DataFrame(data)

    return T, parglmo
