// benches/dense_marginalize.rs
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use loopy::factor::{DenseFactor, UnaryFactor, Factor, FactorKind};
use ndarray::IxDyn;
use ndarray::ArrayD;

fn bench_dense_marginalize(c: &mut Criterion) {
    // 1D dense factor with 100 elements (log-space)
    let data_vec = vec![1.0f64; 100]
        .into_iter()
        .map(|x| x.ln())
        .collect::<Vec<f64>>();

    let data = ArrayD::from_shape_vec(IxDyn(&[100]), data_vec).unwrap();
    let f = DenseFactor::new(vec![0usize], data);

    c.bench_function("dense_marginalize_100", |b| {
        b.iter(|| {
            let out = black_box(f.clone()).marginalize(&[0usize]);
            // touch the result so the optimizer can't elide the work
            match out {
                FactorKind::Dense(d) => black_box(d.data().len()),
                FactorKind::Unary(u) => black_box(u.data().len()),
                FactorKind::Scalar(s) => black_box(s.value() as usize),
            }
        })
    });
}

fn bench_unary_marginalize(c: &mut Criterion) {
    // Unary factor with 100 elements (log-space)
    let f = UnaryFactor::new(0, vec![0.0f64; 100]); // log(1.0) == 0.0

    c.bench_function("unary_marginalize_100", |b| {
        b.iter(|| {
            let out = black_box(f.clone()).marginalize(&[0usize]);
            match out {
                FactorKind::Dense(d) => black_box(d.data().len()),
                FactorKind::Unary(u) => black_box(u.data().len()),
                FactorKind::Scalar(s) => black_box(s.value() as usize),
            }
        })
    });
}

criterion_group!(
    benches,
    bench_dense_marginalize,
    bench_unary_marginalize
);
criterion_main!(benches);
