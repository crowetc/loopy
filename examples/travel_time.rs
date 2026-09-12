//! Travel time estimation with discrete factor-graph inference.
//!
//! This example models how weather and traffic conditions influence travel
//! time and demonstrates:
//!
//! - named discrete variable domains,
//! - unary and multidimensional factors,
//! - sum-product inference for marginal beliefs,
//! - max-product inference for best-supported states,
//! - synchronous message passing over the same factor graph,
//! - illustrative timing of each inference run.

use std::time::Instant;

use loopy::{
    BeliefState, DenseFactor, FactorGraph, FactorKind, FactorNormalize, FactorOps, LogMaxProduct,
    LogSumProduct, RunOptions, Schedule, Semiring, Synchronous, UnaryFactor, Variable, VariableId,
};
use ndarray::array;

const WEATHER: [&str; 2] = ["clear", "rain"];
const TRAFFIC: [&str; 2] = ["light", "heavy"];
const ROAD_CONDITION: [&str; 2] = ["dry", "wet"];
const TRAVEL_TIME: [&str; 3] = ["short", "medium", "long"];

fn main() {
    let (graph, travel_time) = build_model();

    println!("Travel Time Estimation");
    println!("======================\n");

    run_sum_product(graph.clone(), travel_time);

    println!();

    run_max_product(graph, travel_time);
}

/// Builds the factor graph for the travel-time model.
///
/// Weather influences road conditions, while traffic and road conditions
/// jointly influence travel time.
fn build_model() -> (FactorGraph, VariableId) {
    let mut graph = FactorGraph::new();

    let weather = graph
        .add_variable(Variable::discrete("weather", WEATHER))
        .expect("valid weather variable");

    let traffic = graph
        .add_variable(Variable::discrete("traffic", TRAFFIC))
        .expect("valid traffic variable");

    let road_condition = graph
        .add_variable(Variable::discrete("road_condition", ROAD_CONDITION))
        .expect("valid road-condition variable");

    let travel_time = graph
        .add_variable(Variable::discrete("travel_time", TRAVEL_TIME))
        .expect("valid travel-time variable");

    // P(Weather)
    //
    //             clear  rain
    // weather      0.70  0.30
    graph
        .add_factor(UnaryFactor::from_linear(weather, vec![0.70, 0.30]))
        .expect("valid weather prior");

    // P(Traffic)
    //
    //             light  heavy
    // traffic      0.60   0.40
    graph
        .add_factor(UnaryFactor::from_linear(traffic, vec![0.60, 0.40]))
        .expect("valid traffic prior");

    // P(RoadCondition | Weather)
    //
    //             dry   wet
    // clear       0.95  0.05
    // rain        0.20  0.80
    //
    // Axis order: [weather, road_condition]
    graph
        .add_factor(DenseFactor::from_linear(
            vec![weather, road_condition],
            array![[0.95, 0.05], [0.20, 0.80]].into_dyn(),
        ))
        .expect("valid weather-road factor");

    // P(TravelTime | Traffic, RoadCondition)
    //
    //                       short  medium  long
    // light, dry             0.75    0.20  0.05
    // light, wet             0.25    0.55  0.20
    // heavy, dry             0.15    0.60  0.25
    // heavy, wet             0.05    0.35  0.60
    //
    // Axis order: [traffic, road_condition, travel_time]
    graph
        .add_factor(DenseFactor::from_linear(
            vec![traffic, road_condition, travel_time],
            array![
                [[0.75, 0.20, 0.05], [0.25, 0.55, 0.20]],
                [[0.15, 0.60, 0.25], [0.05, 0.35, 0.60]],
            ]
            .into_dyn(),
        ))
        .expect("valid travel-time factor");

    (graph, travel_time)
}

fn run_sum_product(graph: FactorGraph, travel_time: VariableId) {
    let mut state = BeliefState::<LogSumProduct>::from_graph(graph);
    let mut schedule = Synchronous::new();

    let start = Instant::now();
    let result = schedule.run(&mut state, RunOptions::default());
    let elapsed = start.elapsed();

    let belief = normalized_belief::<LogSumProduct>(&state, travel_time);

    println!("Sum-product");
    println!("-----------");
    println!(
        "converged: {}  iterations: {}  residual: {:.3e}  elapsed: {:?}",
        result.converged, result.iterations, result.residual, elapsed
    );

    println!("\nTravel-time marginal:");

    for (label, log_probability) in TRAVEL_TIME.iter().zip(belief.data()) {
        println!("  {label:<6}  {:.4}", log_probability.exp());
    }
}

fn run_max_product(graph: FactorGraph, travel_time: VariableId) {
    let mut state = BeliefState::<LogMaxProduct>::from_graph(graph);
    let mut schedule = Synchronous::new();

    let start = Instant::now();
    let result = schedule.run(&mut state, RunOptions::default());
    let elapsed = start.elapsed();

    let belief = normalized_belief::<LogMaxProduct>(&state, travel_time);

    let (best_index, _) = belief
        .data()
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("travel-time domain is non-empty");

    println!("Max-product");
    println!("-----------");
    println!(
        "converged: {}  iterations: {}  residual: {:.3e}  elapsed: {:?}",
        result.converged, result.iterations, result.residual, elapsed
    );

    println!("\nTravel-time relative scores:");

    for (label, log_score) in TRAVEL_TIME.iter().zip(belief.data()) {
        println!("  {label:<6}  {:.4}", log_score.exp());
    }

    println!("\nBest-supported travel time: {}", TRAVEL_TIME[best_index]);
}

fn normalized_belief<S>(state: &BeliefState<S>, variable: VariableId) -> loopy::UnaryFactor
where
    S: Semiring,
    FactorKind: FactorOps<S> + FactorNormalize<S>,
{
    let belief = state
        .belief(variable)
        .expect("variable exists")
        .expect("belief is available");

    let belief = <FactorKind as FactorNormalize<S>>::normalize(belief);

    let FactorKind::Unary(belief) = belief else {
        panic!("belief over one variable must be unary");
    };

    belief
}
