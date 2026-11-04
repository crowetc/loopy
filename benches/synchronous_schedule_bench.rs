
//! Benchmarks for synchronous belief propagation.
//!
//! These benchmarks characterize the performance of Loopy's synchronous
//! message-passing schedule across graph sizes, factor arities, variable
//! cardinalities, and graph topologies.
//!
//! Three aspects of execution are measured:
//! - Initialization: construction of a factor graph and its belief state.
//! - First step: the initial synchronous update of a fresh belief state.
//! - Repeated step: the cost of subsequent synchronous updates.
//!
//! Repeated-step benchmarks execute one update before measurement, then
//! reuse the same belief state for all measured iterations. They do not
//! reset the state or check for convergence.
//!
//! Consequently, these benchmarks measure per-iteration computational
//! cost, not the number of iterations or total time required to converge.
//! This distinction is particularly important for cyclic graphs.

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use loopy::belief_state::BeliefState;
use loopy::factor::{DenseFactor, UnaryFactor};
use loopy::factor_graph::FactorGraph;
use loopy::schedule::{Schedule, Synchronous};
use loopy::semiring::LogSumProduct;
use loopy::variable::{Variable, VariableId};
use ndarray::{Array2, Array3};
use std::hint::black_box;

// -----------------------------------------------------------------------------
// Graph construction
// -----------------------------------------------------------------------------

fn discrete_variable(name: &str, cardinality: usize) -> Variable {
    Variable::discrete(name, (0..cardinality).map(|index| index.to_string()))
}

fn pairwise_factor(lhs: VariableId, rhs: VariableId, cardinality: usize) -> DenseFactor {
    // Favor equal states while assigning nonzero weight to unequal states.
    let data = Array2::from_shape_fn(
        (cardinality, cardinality),
        |(i, j)| {
            if i == j { 0.8 } else { 0.2 }
        },
    );

    DenseFactor::from_linear(vec![lhs, rhs], data.into_dyn())
}

fn ternary_factor(a: VariableId, b: VariableId, c: VariableId) -> DenseFactor {
    DenseFactor::from_linear(
        vec![a, b, c],
        Array3::from_shape_vec((2, 2, 2), vec![0.8, 0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2])
            .unwrap()
            .into_dyn(),
    )
}

fn pairwise_chain(length: usize, cardinality: usize) -> BeliefState<LogSumProduct> {
    assert!(length >= 2);
    assert!(cardinality >= 2);

    let mut graph = FactorGraph::new();

    let variables: Vec<_> = (0..length)
        .map(|index| {
            graph
                .add_variable(discrete_variable(&format!("x{index}"), cardinality))
                .unwrap()
        })
        .collect();

    let evidence: Vec<_> = (0..cardinality)
        .map(|index| if index == 0 { 0.8 } else { 0.2 })
        .collect();

    graph
        .add_factor(UnaryFactor::from_linear(variables[0], evidence))
        .unwrap();

    for pair in variables.windows(2) {
        graph
            .add_factor(pairwise_factor(pair[0], pair[1], cardinality))
            .unwrap();
    }

    BeliefState::from_graph(graph)
}

fn pairwise_tree(length: usize) -> BeliefState<LogSumProduct> {
    assert!(length >= 2);

    let mut graph = FactorGraph::new();

    let variables: Vec<_> = (0..length)
        .map(|index| {
            graph
                .add_variable(discrete_variable(&format!("x{index}"), 2))
                .unwrap()
        })
        .collect();

    graph
        .add_factor(UnaryFactor::from_linear(variables[0], vec![0.8, 0.2]))
        .unwrap();

    // A binary tree with parent(i) = (i - 1) / 2.
    for child in 1..length {
        let parent = (child - 1) / 2;

        graph
            .add_factor(pairwise_factor(variables[parent], variables[child], 2))
            .unwrap();
    }

    BeliefState::from_graph(graph)
}

fn pairwise_cycle(length: usize) -> BeliefState<LogSumProduct> {
    assert!(length >= 3);

    let mut graph = FactorGraph::new();

    let variables: Vec<_> = (0..length)
        .map(|index| {
            graph
                .add_variable(discrete_variable(&format!("x{index}"), 2))
                .unwrap()
        })
        .collect();

    graph
        .add_factor(UnaryFactor::from_linear(variables[0], vec![0.8, 0.2]))
        .unwrap();

    for pair in variables.windows(2) {
        graph
            .add_factor(pairwise_factor(pair[0], pair[1], 2))
            .unwrap();
    }

    // Close the chain to introduce a single cycle.
    graph
        .add_factor(pairwise_factor(variables[length - 1], variables[0], 2))
        .unwrap();

    BeliefState::from_graph(graph)
}

fn ternary_chain(length: usize) -> BeliefState<LogSumProduct> {
    assert!(length >= 3 && length % 2 == 1);

    let mut graph = FactorGraph::new();

    let variables: Vec<_> = (0..length)
        .map(|index| {
            graph
                .add_variable(discrete_variable(&format!("x{index}"), 2))
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

// -----------------------------------------------------------------------------
// Benchmark helpers
// -----------------------------------------------------------------------------

/// Measures repeated synchronous updates of an existing belief state.
///
/// The first update is excluded from timing. Subsequent iterations reuse
/// the same state, regardless of whether its messages have converged.
fn bench_repeated_step(
    c: &mut Criterion,
    name: &str,
    make_state: impl Fn() -> BeliefState<LogSumProduct>,
) {
    c.bench_function(name, |b| {
        let mut state = make_state();
        let mut schedule = Synchronous::new();

        schedule.step(&mut state);

        b.iter(|| {
            let residual = schedule.step(black_box(&mut state));
            black_box(residual);
        });
    });
}

// -----------------------------------------------------------------------------
// Initialization and first-step benchmarks
// -----------------------------------------------------------------------------

fn bench_initialization(c: &mut Criterion) {
    for length in [10, 100, 1000] {
        c.bench_function(&format!("initialize_pairwise_{length}"), |b| {
            b.iter(|| {
                black_box(pairwise_chain(black_box(length), 2));
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
                || (pairwise_chain(length, 2), Synchronous::new()),
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

// -----------------------------------------------------------------------------
// Per-iteration scaling benchmarks
// -----------------------------------------------------------------------------

fn bench_synchronous_pairwise(c: &mut Criterion) {
    for length in [10, 100, 1000] {
        bench_repeated_step(c, &format!("synchronous_step_pairwise_{length}"), || {
            pairwise_chain(length, 2)
        });
    }
}

fn bench_synchronous_ternary(c: &mut Criterion) {
    for length in [11, 101, 1001] {
        bench_repeated_step(c, &format!("synchronous_step_ternary_{length}"), || {
            ternary_chain(length)
        });
    }
}

fn bench_synchronous_cardinality(c: &mut Criterion) {
    const LENGTH: usize = 100;

    for cardinality in [2, 4, 8, 16] {
        bench_repeated_step(
            c,
            &format!("synchronous_step_pairwise_cardinality_{cardinality}"),
            || pairwise_chain(LENGTH, cardinality),
        );
    }
}

fn bench_synchronous_tree(c: &mut Criterion) {
    for length in [10, 100, 1000] {
        bench_repeated_step(c, &format!("synchronous_step_tree_{length}"), || {
            pairwise_tree(length)
        });
    }
}

fn bench_synchronous_cycle(c: &mut Criterion) {
    for length in [10, 100, 1000] {
        bench_repeated_step(c, &format!("synchronous_step_cycle_{length}"), || {
            pairwise_cycle(length)
        });
    }
}

criterion_group!(
    benches,
    bench_initialization,
    bench_first_step,
    bench_synchronous_pairwise,
    bench_synchronous_ternary,
    bench_synchronous_cardinality,
    bench_synchronous_tree,
    bench_synchronous_cycle,
);

criterion_main!(benches);
