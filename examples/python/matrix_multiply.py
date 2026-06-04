# APG Example: matrix_multiply (Python/Numba)
# Computes C = A * B for NxN matrices stored as flat arrays.
# Expected parallelism: polyhedral (outer loop rows are independent)
import numpy as np

def matrix_multiply(A, B, C, N):
    for i in range(N):
        for j in range(N):
            tmp = 0.0
            for k in range(N):
                tmp = tmp + A[i * N + k] * B[k * N + j]
            C[i * N + j] = tmp
