# APG Example: vector_scale (Python/Numba)
# A simple affine loop scaling every element of array A by scalar s.
# Expected parallelism: polyhedral (no dependencies between iterations)
import numpy as np

def vector_scale(A, s, N):
    for i in range(N):
        A[i] = A[i] * s
