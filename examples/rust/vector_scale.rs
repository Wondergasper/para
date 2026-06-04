// APG Example: vector_scale (Rust/Rayon)
// A simple affine loop scaling every element of slice A by scalar s.
// Expected parallelism: polyhedral (independent iterations)
pub fn vector_scale(a: &mut [f32], s: f32, n: usize) {
    for i in 0..n {
        a[i] *= s;
    }
}
