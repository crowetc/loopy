use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use loopy::belief_state::BeliefState;
use loopy::factor::{DenseFactor, UnaryFactor};
use loopy::factor_graph::FactorGraph;
use loopy::schedule::{Schedule, Synchronous};
use loopy::semiring::LogSumProduct;
use loopy::variable::{Variable, VariableId};
use ndarray::{Array2, Array3};
use std::hint::black_box;

fn binary_variable(name: &str) -> Variable {
    Variable::discrete(name, ["0", "1"])
}

fn pairwise_factor(lhs: VariableId, rhs: VariableId) -> DenseFactor {
    DenseFactor::from_linear(
        vec![lhs, rhs],
        Array2::from_shape_vec((2, 2), vec![0.8, 0.2, 0.2, 0.8])
            .unwrap()
            .into_dyn(),
    )
}

fn ternary_factor(a: VariableId, b: VariableId, c: VariableId) -> DenseFactor {
    DenseFactor::from_linear(
        vec![a, b, c],
        Array3::from_shape_vec((2, 2, 2), vec![0.8, 0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2])
            .unwrap()
            .into_dyn(),
    )
}

fn pairwise_chain(length: usize) -> BeliefState<LogSumProduct> {
    assert!(length >= 2);

    let mut graph = FactorGraph::new();

    let variables: Vec<_> = (0..length)
        .map(|index| {
            graph
                .add_variable(binary_variable(&format!("x{index}")))
                .unwrap()
        })
        .collect();

    graph
        .add_factor(UnaryFactor::from_linear(variables[0], vec![0.8, 0.2]))
        .unwrap();

    for pair in variables.windows(2) {
        graph.add_factor(pairwise_factor(pair[0], pair[1])).unwrap();
    }

    BeliefState::from_graph(graph)
}

fn ternary_chain(length: usize) -> BeliefState<LogSumProduct> {
    assert!(length >= 3 && length % 2 == 1);

    let mut graph = FactorGraph::new();

    let variables: Vec<_> = (0..length)
        .map(|index| {
            graph
                .add_variable(binary_variable(&format!("x{index}")))
                .unwrap()
        })
        .collect();

    graph
        .add_factor(UnaryFactor::from_linear(variables[0], vec![0.8, 0.2]))
        .unwrap();

    // Adjacent ternary factors share one variable.
    for triple in variables.windows(3).step_by(2) {
        graph
            .add_factor(ternary_factor(triple[0], triple[1], triple[2]))
            .unwrap();
    }

    BeliefState::from_graph(graph)
}

fn bench_initialization(c: &mut Criterion) {
    for length in [10, 100, 1000] {
        c.bench_function(&format!("initialize_pairwise_{length}"), |b| {
            b.iter(|| {
                black_box(pairwise_chain(black_box(length)));
            });
        });
    }

    for length in [11, 101, 1001] {
        c.bench_function(&format!("initialize_ternary_{length}"), |b| {
            b.iter(|| {
                black_box(ternary_chain(black_box(length)));
            });
        });
    }
}

fn bench_first_step(c: &mut Criterion) {
    for length in [10, 100, 1000] {
        c.bench_function(&format!("synchronous_first_step_pairwise_{length}"), |b| {
            b.iter_batched(
                || (pairwise_chain(length), Synchronous::new()),
                |(mut state, mut schedule)| {
                    black_box(schedule.step(black_box(&mut state)));
                },
                BatchSize::PerIteration,
            );
        });
    }

    for length in [11, 101, 1001] {
        c.bench_function(&format!("synchronous_first_step_ternary_{length}"), |b| {
            b.iter_batched(
                || (ternary_chain(length), Synchronous::new()),
                |(mut state, mut schedule)| {
                    black_box(schedule.step(black_box(&mut state)));
                },
                BatchSize::PerIteration,
            );
        });
    }
}

fn bench_synchronous_pairwise(c: &mut Criterion) {
    for length in [10, 100, 1000] {
        c.bench_function(&format!("synchronous_step_pairwise_{length}"), |b| {
            let mut state = pairwise_chain(length);
            let mut schedule = Synchronous::new();

            // Execute one iteration before measuring repeated updates.
            schedule.step(&mut state);

            b.iter(|| {
                let residual = schedule.step(black_box(&mut state));
                black_box(residual);
            });
        });
    }
}

fn bench_synchronous_ternary(c: &mut Criterion) {
    for length in [11, 101, 1001] {
        c.bench_function(&format!("synchronous_step_ternary_{length}"), |b| {
            let mut state = ternary_chain(length);
            let mut schedule = Synchronous::new();

            // Execute one iteration before measuring repeated updates.
            schedule.step(&mut state);

            b.iter(|| {
                let residual = schedule.step(black_box(&mut state));
                black_box(residual);
            });
        });
    }
}

criterion_group!(
    benches,
    bench_initialization,
    bench_first_step,
    bench_synchronous_pairwise,
    bench_synchronous_ternary,
);

criterion_main!(benches);
