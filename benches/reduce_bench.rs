use criterion::{Criterion, criterion_group, criterion_main};
use loopy::factor::{
    DenseFactor, FactorKind, FactorOps, LogMaxProduct, LogSumProduct, UnaryFactor, VariableId,
};
use ndarray::{ArrayD, IxDyn};
use std::hint::black_box;

fn consume_result(out: FactorKind) {
    match out {
        FactorKind::Dense(d) => black_box(d.data().len()),
        FactorKind::Unary(u) => black_box(u.data().len()),
        FactorKind::Scalar(s) => black_box(s.value() as usize),
    };
}

fn make_dense(scope: Vec<VariableId>, shape: &[usize]) -> DenseFactor {
    let size = shape.iter().product();

    let data = vec![0.0_f64; size];

    DenseFactor::new(scope, ArrayD::from_shape_vec(IxDyn(shape), data).unwrap())
}

fn bench_dense_reduce_single_axis(c: &mut Criterion) {
    let f = make_dense(
        vec![VariableId::new(0), VariableId::new(1), VariableId::new(2)],
        &[20, 20, 20],
    );

    c.bench_function("dense_reduce_sum_single_axis", |b| {
        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogSumProduct>>::reduce(
                black_box(f.clone()),
                black_box(&[VariableId::new(0)]),
            );

            consume_result(out);
        })
    });

    c.bench_function("dense_reduce_max_single_axis", |b| {
        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogMaxProduct>>::reduce(
                black_box(f.clone()),
                black_box(&[VariableId::new(0)]),
            );

            consume_result(out);
        })
    });
}

fn bench_dense_reduce_multiple_axes(c: &mut Criterion) {
    let f = make_dense(
        vec![VariableId::new(0), VariableId::new(1), VariableId::new(2)],
        &[20, 20, 20],
    );

    c.bench_function("dense_reduce_sum_multiple_axes", |b| {
        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogSumProduct>>::reduce(
                black_box(f.clone()),
                black_box(&[VariableId::new(0), VariableId::new(2)]),
            );

            consume_result(out);
        })
    });

    c.bench_function("dense_reduce_max_multiple_axes", |b| {
        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogMaxProduct>>::reduce(
                black_box(f.clone()),
                black_box(&[VariableId::new(0), VariableId::new(2)]),
            );

            consume_result(out);
        })
    });
}

fn bench_dense_reduce_to_scalar(c: &mut Criterion) {
    let f = make_dense(
        vec![VariableId::new(0), VariableId::new(1), VariableId::new(2)],
        &[20, 20, 20],
    );

    c.bench_function("dense_reduce_sum_to_scalar", |b| {
        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogSumProduct>>::reduce(
                black_box(f.clone()),
                black_box(&[VariableId::new(0), VariableId::new(1), VariableId::new(2)]),
            );

            consume_result(out);
        })
    });

    c.bench_function("dense_reduce_max_to_scalar", |b| {
        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogMaxProduct>>::reduce(
                black_box(f.clone()),
                black_box(&[VariableId::new(0), VariableId::new(1), VariableId::new(2)]),
            );

            consume_result(out);
        })
    });
}

fn bench_dense_reduce_out_of_scope(c: &mut Criterion) {
    let f = make_dense(
        vec![VariableId::new(0), VariableId::new(1), VariableId::new(2)],
        &[20, 20, 20],
    );

    c.bench_function("dense_reduce_sum_out_of_scope", |b| {
        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogSumProduct>>::reduce(
                black_box(f.clone()),
                black_box(&[VariableId::new(99)]),
            );

            consume_result(out);
        })
    });

    c.bench_function("dense_reduce_max_out_of_scope", |b| {
        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogMaxProduct>>::reduce(
                black_box(f.clone()),
                black_box(&[VariableId::new(99)]),
            );

            consume_result(out);
        })
    });
}

fn bench_unary_reduce(c: &mut Criterion) {
    let f = UnaryFactor::new(VariableId::new(0), vec![0.0_f64; 100]);

    c.bench_function("unary_reduce_sum", |b| {
        b.iter(|| {
            let out = <UnaryFactor as FactorOps<LogSumProduct>>::reduce(
                black_box(f.clone()),
                black_box(&[VariableId::new(0)]),
            );

            consume_result(out);
        })
    });

    c.bench_function("unary_reduce_max", |b| {
        b.iter(|| {
            let out = <UnaryFactor as FactorOps<LogMaxProduct>>::reduce(
                black_box(f.clone()),
                black_box(&[VariableId::new(0)]),
            );

            consume_result(out);
        })
    });
}

fn bench_unary_reduce_out_of_scope(c: &mut Criterion) {
    let f = UnaryFactor::new(VariableId::new(0), vec![0.0_f64; 100]);

    c.bench_function("unary_reduce_sum_out_of_scope", |b| {
        b.iter(|| {
            let out = <UnaryFactor as FactorOps<LogSumProduct>>::reduce(
                black_box(f.clone()),
                black_box(&[VariableId::new(99)]),
            );

            consume_result(out);
        })
    });

    c.bench_function("unary_reduce_max_out_of_scope", |b| {
        b.iter(|| {
            let out = <UnaryFactor as FactorOps<LogMaxProduct>>::reduce(
                black_box(f.clone()),
                black_box(&[VariableId::new(99)]),
            );

            consume_result(out);
        })
    });
}

criterion_group!(
    benches,
    bench_dense_reduce_single_axis,
    bench_dense_reduce_multiple_axes,
    bench_dense_reduce_to_scalar,
    bench_dense_reduce_out_of_scope,
    bench_unary_reduce,
    bench_unary_reduce_out_of_scope,
);

criterion_main!(benches);
