# APG Example: dot_product (Python/Numba)
# Computes the dot product of two arrays A and B.
# Expected parallelism: reduction (sum reduction on result)
import numpy as np

def dot_product(A, B, N):
    s = 0.0
    for i in range(N):
        s = s + A[i] * B[i]
    return s
