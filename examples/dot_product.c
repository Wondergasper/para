/* dot_product.c — Dot product (reduction loop)
 *
 * Parallelism type: POLYHEDRAL
 * Reduction variable: sum (safe with OpenMP reduction clause)
 * Expected transform: #pragma omp parallel for reduction(+:sum)
 */
float dot_product(float* A, float* B, int N) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}
