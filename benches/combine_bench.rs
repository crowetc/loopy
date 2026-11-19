use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use loopy::factor::{DenseFactor, FactorKind, FactorOps, UnaryFactor};
use loopy::semiring::LogSumProduct;
use loopy::variable::VariableId;
use ndarray::{ArrayD, IxDyn};
use std::hint::black_box;

fn consume_result(out: FactorKind) {
    black_box(out);
}

fn dense_factor(scope: Vec<VariableId>, shape: &[usize]) -> DenseFactor {
    let size = shape.iter().product();

    let data = (0..size).map(|x| x as f64 + 1.0).collect::<Vec<f64>>();

    let data = ArrayD::from_shape_vec(IxDyn(shape), data).unwrap();

    DenseFactor::new(scope, data)
}

fn unary_factor(var: VariableId, size: usize) -> UnaryFactor {
    UnaryFactor::new(var, (0..size).map(|x| x as f64 + 1.0).collect())
}

fn bench_dense_x_dense(c: &mut Criterion) {
    //
    // Disjoint scopes:
    // [0, 1] × [2, 3]
    //
    c.bench_function("combine_dense_x_dense_disjoint", |b| {
        let f = dense_factor(vec![VariableId::new(0), VariableId::new(1)], &[20, 20]);
        let g = dense_factor(vec![VariableId::new(2), VariableId::new(3)], &[20, 20]);

        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogSumProduct>>::combine(
                black_box(f.clone()),
                &FactorKind::Dense(g.clone()),
            );

            consume_result(out);
        })
    });

    //
    // One shared variable:
    // [0, 1] × [1, 2]
    //
    c.bench_function("combine_dense_x_dense_intersect", |b| {
        let f = dense_factor(vec![VariableId::new(0), VariableId::new(1)], &[20, 20]);
        let g = dense_factor(vec![VariableId::new(1), VariableId::new(2)], &[20, 20]);

        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogSumProduct>>::combine(
                black_box(f.clone()),
                &FactorKind::Dense(g.clone()),
            );

            consume_result(out);
        })
    });

    //
    // Unsorted scopes:
    // [1, 0] × [2, 1]
    //
    c.bench_function("combine_dense_x_dense_unsorted", |b| {
        let f = dense_factor(vec![VariableId::new(1), VariableId::new(0)], &[20, 20]);
        let g = dense_factor(vec![VariableId::new(2), VariableId::new(1)], &[20, 20]);

        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogSumProduct>>::combine(
                black_box(f.clone()),
                &FactorKind::Dense(g.clone()),
            );

            consume_result(out);
        })
    });
}

fn bench_dense_x_dense_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("combine_dense_x_dense_matching");

    for nvars in [2, 4, 8] {
        let scope: Vec<_> = (0..nvars).map(VariableId::new).collect();
        let shape = vec![2; nvars];

        let lhs = dense_factor(scope.clone(), &shape);
        let rhs = FactorKind::Dense(dense_factor(scope.clone(), &shape));

        // Existing API: clone the input, then combine.
        // Both operations are included in the measurement.
        group.bench_with_input(BenchmarkId::new("combine", nvars), &nvars, |b, _| {
            b.iter(|| {
                let out = <DenseFactor as FactorOps<LogSumProduct>>::combine(
                    black_box(lhs.clone()),
                    black_box(&rhs),
                );

                consume_result(out);
            });
        });

        // New API: reuse a compatible destination.
        // The destination is allocated once, outside the timed loop.
        group.bench_with_input(BenchmarkId::new("combine_into", nvars), &nvars, |b, _| {
            let mut output = FactorKind::Dense(dense_factor(scope.clone(), &shape));

            b.iter(|| {
                <DenseFactor as FactorOps<LogSumProduct>>::combine_into(
                    black_box(&lhs),
                    black_box(&rhs),
                    black_box(&mut output),
                );

                black_box(&output);
            });
        });
    }

    group.finish();
}

fn bench_dense_x_unary(c: &mut Criterion) {
    //
    // Shared variable:
    // Dense [0, 1] × Unary [1]
    //
    c.bench_function("combine_dense_x_unary_intersect", |b| {
        let f = dense_factor(vec![VariableId::new(0), VariableId::new(1)], &[100, 100]);
        let u = unary_factor(VariableId::new(1), 100);

        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogSumProduct>>::combine(
                black_box(f.clone()),
                &FactorKind::Unary(u.clone()),
            );

            consume_result(out);
        })
    });

    //
    // Disjoint variable:
    // Dense [0, 1] × Unary [2]
    //
    c.bench_function("combine_dense_x_unary_disjoint", |b| {
        let f = dense_factor(vec![VariableId::new(0), VariableId::new(1)], &[100, 100]);
        let u = unary_factor(VariableId::new(2), 100);

        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogSumProduct>>::combine(
                black_box(f.clone()),
                &FactorKind::Unary(u.clone()),
            );

            consume_result(out);
        })
    });

    //
    // Shared variable, binary cardinality:
    // Dense [0, 1] × Unary [1]
    //
    c.bench_function("combine_dense_x_unary_intersect_binary", |b| {
        let f = dense_factor(vec![VariableId::new(0), VariableId::new(1)], &[2, 2]);
        let u = unary_factor(VariableId::new(1), 2);

        b.iter(|| {
            let out = <DenseFactor as FactorOps<LogSumProduct>>::combine(
                black_box(f.clone()),
                &FactorKind::Unary(u.clone()),
            );

            consume_result(out);
        })
    });
}

fn bench_dense_x_unary_into(c: &mut Criterion) {
    let mut group = c.benchmark_group("combine_dense_x_unary_matching");

    for card in [2, 10, 100] {
        let scope = vec![VariableId::new(0), VariableId::new(1)];
        let shape = [card, card];

        for layout in [
            "last_axis_contiguous",
            "first_axis_contiguous",
            "last_axis_noncontiguous",
        ] {
            let (lhs, unary_var) = match layout {
                "last_axis_contiguous" => (dense_factor(scope.clone(), &shape), VariableId::new(1)),

                "first_axis_contiguous" => {
                    (dense_factor(scope.clone(), &shape), VariableId::new(0))
                }

                "last_axis_noncontiguous" => {
                    // DenseFactor::new canonicalizes the scope,
                    // producing a transposed memory layout.
                    let lhs = dense_factor(vec![VariableId::new(1), VariableId::new(0)], &shape);

                    assert!(!lhs.data().is_standard_layout());

                    (lhs, VariableId::new(1))
                }

                _ => unreachable!(),
            };

            let rhs = FactorKind::Unary(unary_factor(unary_var, card));

            let case = format!("{layout}/{card}");

            // Existing API: clone the dense input, then combine.
            group.bench_with_input(BenchmarkId::new("combine", &case), &card, |b, _| {
                b.iter(|| {
                    let out = <DenseFactor as FactorOps<LogSumProduct>>::combine(
                        black_box(lhs.clone()),
                        black_box(&rhs),
                    );

                    consume_result(out);
                });
            });

            // New API: reuse a compatible dense destination.
            group.bench_with_input(BenchmarkId::new("combine_into", &case), &card, |b, _| {
                let mut output = FactorKind::Dense(dense_factor(scope.clone(), &shape));

                b.iter(|| {
                    <DenseFactor as FactorOps<LogSumProduct>>::combine_into(
                        black_box(&lhs),
                        black_box(&rhs),
                        black_box(&mut output),
                    );

                    black_box(&output);
                });
            });
        }
    }

    group.finish();
}

fn bench_unary_x_unary(c: &mut Criterion) {
    //
    // Same variable -> Unary result
    //
    c.bench_function("combine_unary_x_unary_intersect", |b| {
        let f = unary_factor(VariableId::new(0), 1000);
        let g = unary_factor(VariableId::new(0), 1000);

        b.iter(|| {
            let out = <UnaryFactor as FactorOps<LogSumProduct>>::combine(
                black_box(f.clone()),
                &FactorKind::Unary(g.clone()),
            );

            consume_result(out);
        })
    });

    //
    // Different variables -> Dense result
    //
    c.bench_function("combine_unary_x_unary_disjoint", |b| {
        let f = unary_factor(VariableId::new(0), 100);
        let g = unary_factor(VariableId::new(1), 100);

        b.iter(|| {
            let out = <UnaryFactor as FactorOps<LogSumProduct>>::combine(
                black_box(f.clone()),
                &FactorKind::Unary(g.clone()),
            );

            consume_result(out);
        })
    });
}

criterion_group!(
    benches,
    bench_dense_x_dense,
    bench_dense_x_dense_into,
    bench_dense_x_unary,
    bench_dense_x_unary_into,
    bench_unary_x_unary,
);

criterion_main!(benches);
