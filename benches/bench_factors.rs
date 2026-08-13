// benches/dense_marginalize.rs
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use loopy::factor::{DenseFactor, UnaryFactor, Factor};
use ndarray::array;

fn bench_dense_marginalize(c: &mut Criterion) {
    // build a 3-variable dense factor with small cardinalities
    let scope = vec![0usize, 1usize, 2usize];
    let data = array![
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]]
    ].mapv(|x: f64| x.ln()).into_dyn();
    let f = DenseFactor::new(scope, data);

    c.bench_function("dense_marginalize_0", |b| {
        b.iter(|| {
            let g = black_box(&f).marginalize(&[0usize]);
            black_box(g);
        })
    });
}

fn bench_unary_marginalize(c: &mut Criterion) {
    let f = UnaryFactor::new(0, vec![1.0f64; 100]);
    c.bench_function("unary_marginalize", |b| {
        b.iter(|| {
            let g = black_box(&f).marginalize(&[0usize]);
            black_box(g);
        })
    });
}

criterion_group!(benches, bench_dense_marginalize, bench_unary_marginalize);
criterion_main!(benches);
