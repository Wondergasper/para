/* stencil_1d.c — 1D averaging stencil (loop-carried dependency via reads)
 *
 * Parallelism type: POLYHEDRAL
 * Reads: A[i-1], A[i], A[i+1]
 * Writes: B[i]
 * Dependency analysis: A and B are SEPARATE arrays → no true WAW/RAW
 * So the loop IS parallelisable (reads from A, writes to B only).
 *
 * Expected transform: #pragma omp parallel for
 * (Works because A is read-only and B is written at disjoint indices)
 *
 * Compare to prefix_sum.c where the same array is both read and written.
 */
void stencil_1d(float* A, float* B, int N) {
    for (int i = 1; i < N - 1; i++) {
        B[i] = (A[i - 1] + A[i] + A[i + 1]) / 3.0f;
    }
}
