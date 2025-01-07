import torch
import numpy as np
from scipy.io import savemat
from parglm_torch.create_design import create_design
from parglm_torch.parglm import parglm
from parglm_torch.simuleMV import simuleMV
import time

def benchmark_parglm(X, F, device='cpu'):
    torch.cuda.empty_cache()  # Clear cache for accurate results
    X = X.to(device)
    F = F.to(device)

    start = time.perf_counter()

    table, parglmo = parglm(X, F, Model=[[0, 1]], Preprocessing=1, device=device,Permutations=1000)

    torch.cuda.synchronize()  # Ensure GPU computations finish
    end = time.perf_counter()

    return table, parglmo, end - start

# Parameters
reps = 4
vars = 400
levels = [[1, 2, 3, 4], [1, 2, 3]]

# Create the design matrix F
F = create_design(levels, Replicates=reps)

# Initialize complex X (real and imaginary parts)
X_real = np.zeros((F.shape[0], vars))
X_imag = np.zeros((F.shape[0], vars))

# Generate data with significant interaction
for i in range(len(levels[0])):
    for j in range(len(levels[1])):
        indices = np.where((F[:, 0] == levels[0][i]) & (F[:, 1] == levels[1][j]))[0]
        sim_data_real = simuleMV(reps, vars, LevelCorr=8)  # Real part
        sim_data_imag = simuleMV(reps, vars, LevelCorr=8)  # Imaginary part
        rand_vec_real = np.random.randn(1, vars)  # Random real noise
        rand_vec_imag = np.random.randn(1, vars)  # Random imaginary noise

        # Combine real and imaginary parts
        X_real[indices, :] = sim_data_real + rand_vec_real
        X_imag[indices, :] = sim_data_imag + rand_vec_imag

# Create complex matrix
X_complex = X_real + 1j * X_imag

# Convert X and F to torch tensors (complex)
X_tensor = torch.tensor(X_complex, dtype=torch.complex64)
F_tensor = torch.tensor(F, dtype=torch.float32)
X_np = X_tensor.numpy()
F_np = F_tensor.numpy()

savemat('data.mat', {'X': X_np, 'F': F_np})
# Run the parglm function with interaction model
cpu_table, cpu_parglmo, cpu_time = benchmark_parglm(X_tensor, F_tensor, device='cpu')
print(f"CPU Time: {cpu_time:.4f} seconds")

gpu_table, gpu_parglmo, gpu_time = benchmark_parglm(X_tensor, F_tensor, device='cuda')
print(f"GPU Time: {gpu_time:.4f} seconds")
    

