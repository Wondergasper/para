/* matrix_multiply.c — Dense matrix multiplication (polyhedral nest)
 *
 * Parallelism type: POLYHEDRAL
 * Loop nest depth: 3 (i, j, k)
 * Parallelisable loops: i and j (outer two)
 * k is a reduction — must use reduction or local accumulator
 * Expected transform: #pragma omp parallel for collapse(2)
 *
 * Note: This is the canonical test for 2D loop tiling (CTT Phase 2).
 */
void matrix_multiply(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++) {
                s += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = s;
        }
    }
}
