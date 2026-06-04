// APG Example: matrix_multiply (Rust/Rayon)
// Computes C = A * B for NxN matrices stored as flat Vec<f32>.
// Expected parallelism: polyhedral (outer loop rows are independent)
pub fn matrix_multiply(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut tmp: f32 = 0.0;
            for k in 0..n {
                tmp += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = tmp;
        }
    }
}
