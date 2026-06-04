// APG Example: dot_product (Rust/Rayon)
// Computes the dot product of two slices A and B.
// Expected parallelism: reduction (parallel fold + sum)
pub fn dot_product(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut s: f32 = 0.0;
    for i in 0..n {
        s += a[i] * b[i];
    }
    s
}
