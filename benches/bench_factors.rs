// benches/dense_marginalize.rs
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use loopy::factor::{DenseFactor, UnaryFactor, Factor};
use ndarray::IxDyn;
use ndarray::ArrayD;

fn bench_dense_marginalize(c: &mut Criterion) {
    // 1D dense factor with 100 elements (log-space)
    let data_vec = vec![1.0f64; 100].into_iter().map(|x| x.ln()).collect::<Vec<f64>>();
    let data = ArrayD::from_shape_vec(IxDyn(&[100]), data_vec).unwrap();
    let f = DenseFactor::new(vec![0usize], data);

    c.bench_function("dense_marginalize_100", |b| {
        b.iter(|| {
            let g = black_box(&f).marginalize(&[0usize]);
            black_box(g);
        })
    });
}

fn bench_unary_marginalize(c: &mut Criterion) {
    // Unary factor with 100 elements (log-space)
    // If UnaryFactor::new expects log-space, pass ln(1.0) = 0.0
    let f = UnaryFactor::new(0, vec![0.0f64; 100]); // log(1.0) == 0.0
    c.bench_function("unary_marginalize_100", |b| {
        b.iter(|| {
            let g = black_box(&f).marginalize(&[0usize]);
            black_box(g);
        })
    });
}

criterion_group!(benches, bench_dense_marginalize, bench_unary_marginalize);
criterion_main!(benches);
