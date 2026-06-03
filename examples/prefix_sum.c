/* prefix_sum.c — Exclusive prefix sum (inherently sequential)
 *
 * Parallelism type: POLYHEDRAL (but loop-carried RAW — sequential)
 * Reads: A[i-1]
 * Writes: A[i]
 * Loop-carried dependency: A[i] = A[i] + A[i-1] → RAW on A
 *
 * This loop CANNOT be trivially parallelised with OpenMP parallel for.
 * Parallel prefix sum algorithms exist (Hillis-Steele, Blelloch),
 * but they require a fundamentally different algorithm — not just a pragma.
 *
 * Expected APG behaviour: classify as polyhedral with RAW dependency,
 * T2 should either:
 *   a) Return the sequential version unchanged (conservative correct), OR
 *   b) Attempt a parallel prefix algorithm (advanced)
 */
void prefix_sum(float* A, int N) {
    for (int i = 1; i < N; i++) {
        A[i] += A[i - 1];
    }
}
