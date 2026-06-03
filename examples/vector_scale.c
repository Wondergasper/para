void vector_scale(float* A, float s, int N) {
    for (int i = 0; i < N; i++) {
        A[i] *= s;
    }
}
