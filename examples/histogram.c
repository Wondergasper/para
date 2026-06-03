/* histogram.c — Histogram update with indirect indexing (irregular)
 *
 * Parallelism type: IRREGULAR
 * Indirect index: hist[data[i]]  →  hist[B[i]] pattern
 * Race risk: HIGH — two threads may write to hist[data[i]] simultaneously
 * Expected transform: LLM must generate atomic or critical section
 *
 * Correct OpenMP version requires:
 *   #pragma omp parallel for
 *   ...
 *     #pragma omp atomic
 *     hist[data[i]]++;
 *
 * Or privatised local histograms with a reduction merge phase.
 */
void histogram(int* hist, int* data, int N, int num_bins) {
    for (int i = 0; i < N; i++) {
        hist[data[i]]++;
    }
}
