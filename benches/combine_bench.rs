use criterion::{Criterion, criterion_group, criterion_main};
use loopy::factor::{DenseFactor, FactorKind, FactorOps, UnaryFactor};
use ndarray::{ArrayD, IxDyn};
use std::hint::black_box;

fn consume_result(out: FactorKind) {
    match out {
        FactorKind::Dense(d) => black_box(d.data().len()),
        FactorKind::Unary(u) => black_box(u.data().len()),
        FactorKind::Scalar(s) => black_box(s.value() as usize),
    };
}

fn dense_factor(scope: Vec<usize>, shape: &[usize]) -> DenseFactor {
    let size = shape.iter().product();

    let data = (0..size).map(|x| x as f64 + 1.0).collect::<Vec<f64>>();

    let data = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();

    DenseFactor::new(scope, data)
}

fn unary_factor(var: usize, size: usize) -> UnaryFactor {
    UnaryFactor::new(var, (0..size).map(|x| x as f64 + 1.0).collect())
}

fn bench_dense_x_dense(c: &mut Criterion) {
    //
    // Disjoint scopes:
    // [0, 1] × [2, 3]
    //
    c.bench_function("combine_dense_x_dense_disjoint", |b| {
        let f = dense_factor(vec![0, 1], &[20, 20]);
        let g = dense_factor(vec![2, 3], &[20, 20]);

        b.iter(|| {
            let out = black_box(f.clone()).combine(FactorKind::Dense(g.clone()));
            consume_result(out);
        })
    });

    //
    // One shared variable:
    // [0, 1] × [1, 2]
    //
    c.bench_function("combine_dense_x_dense_intersect", |b| {
        let f = dense_factor(vec![0, 1], &[20, 20]);
        let g = dense_factor(vec![1, 2], &[20, 20]);

        b.iter(|| {
            let out = black_box(f.clone()).combine(FactorKind::Dense(g.clone()));
            consume_result(out);
        })
    });

    //
    // Unsorted scopes:
    // [1, 0] × [2, 1]
    //
    c.bench_function("combine_dense_x_dense_unsorted", |b| {
        let f = dense_factor(vec![1, 0], &[20, 20]);
        let g = dense_factor(vec![2, 1], &[20, 20]);

        b.iter(|| {
            let out = black_box(f.clone()).combine(FactorKind::Dense(g.clone()));
            consume_result(out);
        })
    });
}

fn bench_dense_x_unary(c: &mut Criterion) {
    //
    // Shared variable:
    // Dense [0, 1] × Unary [1]
    //
    c.bench_function("combine_dense_x_unary_intersect", |b| {
        let f = dense_factor(vec![0, 1], &[100, 100]);
        let u = unary_factor(1, 100);

        b.iter(|| {
            let out = black_box(f.clone()).combine(FactorKind::Unary(u.clone()));
            consume_result(out);
        })
    });

    //
    // Disjoint variable:
    // Dense [0, 1] × Unary [2]
    //
    c.bench_function("combine_dense_x_unary_disjoint", |b| {
        let f = dense_factor(vec![0, 1], &[100, 100]);
        let u = unary_factor(2, 100);

        b.iter(|| {
            let out = black_box(f.clone()).combine(FactorKind::Unary(u.clone()));
            consume_result(out);
        })
    });
}

fn bench_unary_x_unary(c: &mut Criterion) {
    //
    // Same variable → Unary result
    //
    c.bench_function("combine_unary_x_unary_intersect", |b| {
        let f = unary_factor(0, 1000);
        let g = unary_factor(0, 1000);

        b.iter(|| {
            let out = black_box(f.clone()).combine(FactorKind::Unary(g.clone()));
            consume_result(out);
        })
    });

    //
    // Different variables -> Dense result
    //
    c.bench_function("combine_unary_x_unary_disjoint", |b| {
        let f = unary_factor(0, 100);
        let g = unary_factor(1, 100);

        b.iter(|| {
            let out = black_box(f.clone()).combine(FactorKind::Unary(g.clone()));
            consume_result(out);
        })
    });
}

criterion_group!(
    benches,
    bench_dense_x_dense,
    bench_dense_x_unary,
    bench_unary_x_unary,
);

criterion_main!(benches);
