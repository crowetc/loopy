use criterion::{Criterion, criterion_group, criterion_main};
use loopy::belief_state::BeliefState;
use loopy::factor::{DenseFactor, UnaryFactor};
use loopy::factor_graph::FactorGraph;
use loopy::schedule::{Schedule, Synchronous};
use loopy::semiring::LogSumProduct;
use loopy::variable::Variable;
use ndarray::Array2;
use std::hint::black_box;

fn binary_variable(name: &str) -> Variable {
    Variable::discrete(name, ["0", "1"])
}

fn pairwise_factor(
    lhs: loopy::variable::VariableId,
    rhs: loopy::variable::VariableId,
) -> DenseFactor {
    DenseFactor::from_linear(
        vec![lhs, rhs],
        Array2::from_shape_vec((2, 2), vec![0.8, 0.2, 0.2, 0.8])
            .unwrap()
            .into_dyn(),
    )
}

fn chain(length: usize) -> BeliefState<LogSumProduct> {
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

fn bench_synchronous_step(c: &mut Criterion) {
    for length in [10, 100, 1000] {
        c.bench_function(&format!("synchronous_step_chain_{length}"), |b| {
            let mut state = chain(length);
            let mut schedule = Synchronous::new();

            // Prime message state so the benchmark measures the
            // steady-state update path rather than first-message creation.
            schedule.step(&mut state);

            b.iter(|| {
                let residual = schedule.step(black_box(&mut state));
                black_box(residual);
            });
        });
    }
}

criterion_group!(benches, bench_synchronous_step);
criterion_main!(benches);
