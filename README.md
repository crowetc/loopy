# Loopy

Loopy is a Rust library for performing **loopy belief propagation (LBP)** on factor graphs. It provides clear abstractions and modular components for working with factorized functions, semirings, and message-passing algorithms.

Factor graphs are used across many domains, including probabilistic inference, optimization, constraint satisfaction, coding theory, robotics, and computer vision. Loopy aims to support this breadth while remaining simple, expressive, and performant.

## Motivation

Many real‑world problems involve a large global function that can be decomposed into smaller, local functions. Factor graphs provide a natural way to represent this structure, and message‑passing algorithms offer efficient methods for performing inference over it.

Loopy belief propagation extends standard belief propagation to factor graphs containing cycles. This makes it useful for a wide range of practical problems where exact inference may be computationally infeasible.

Despite their broad applicability, modular implementations of message-passing algorithms are relatively uncommon—particularly in systems languages that emphasize both safety and performance.

Loopy aims to fill that gap.

The library is built around several core goals:

- **Clear abstractions** — variables, factors, semirings, graphs, and messages are explicit concepts rather than details hidden behind opaque APIs.

- **Modularity** — factor representations, inference algorithms, message-passing schedules, and algebraic choices can evolve independently.

- **Generality** — the underlying abstractions are intended to support more than a single inference algorithm or factor representation.

- **Practicality** — suitable for experimentation and research today, while designed with future production applications in mind.

- **Rust-native design** — leveraging strong typing, memory safety, explicit ownership, and predictable performance.

Loopy is intended to make the mechanics of message passing understandable without sacrificing the architectural flexibility needed for more advanced inference systems.

## Factor Graphs: The Big Picture

A factor graph is a bipartite graphical model that expresses a global problem as a collection of local relationships.

Formally, a factor graph can be written as

![Factor Graph](doc/img/factor_graph.svg)

where:

- <img src="https://latex.codecogs.com/svg.image?V%20=%20%5C%7BX_1,%20X_2,%20%5Cldots,%20X_n%5C%7D"
style="vertical-align: middle;" width="160"> is a set of **variable nodes**

- <img src="https://latex.codecogs.com/svg.image?%5CPhi%20=%20%5C%7B%5Cphi_1,%20%5Cphi_2,%20%5Cldots,%20%5Cphi_m%5C%7D"
style="vertical-align: middle;" width="160"> is a set of **factor nodes**

- <img src="https://latex.codecogs.com/svg.image?E%20%5Csubseteq%20V%20%5Ctimes%20%5CPhi"
style="vertical-align: middle;" width="90"> is the set of **edges** connecting factors to the variables in their scopes

The graph structure indicates which variables participate in which local relationships. Together, those relationships define the global function:


![Global function](doc/img/global_function.svg)



This decomposition enables local computation and efficient propagation of information through the graph.

## What’s a Factor?
A factor is a **function** over one or more variables. Depending on the application, factors may represent probabilities, compatibility scores, costs, energies, constraints, or other relationships.

Each factor has a **scope**: the set of variables on which it depends.

Conceptually, a factor can be written as

![Factor Definition](doc/img/factor_definition.svg)

where <img src="https://latex.codecogs.com/svg.image?\mathcal{K}" style="vertical-align: middle;" width="12">  is the value domain used by the inference system.

Loopy is built around two fundamental factor operations:

- **combine** — aggregates information from multiple factors
- **reduce** — eliminates one or more variables from a factor

These operations are defined by the algebra used for inference.

For example:

| Inference | Combine | Reduce |
|---|---|---|
| Sum-product | Multiply | Sum |
| Max-product | Multiply | Max |
| Constraint satisfaction | AND | OR |

In general:

- `combine` corresponds to the semiring's multiplicative operation
- `reduce` corresponds to the semiring's additive operation

This separation allows the same high-level factor-graph and message-passing machinery to support different forms of inference.

In Loopy, the semiring provides the algebra for interpreting the factor graph: it defines how local functions combine into a global one and how factors relate during message passing. The graph specifies which pieces of the global function exist, while the semiring defines how those pieces interact.

## Loopy Belief Propagation

Loopy belief propagation (LBP) is an iterative message‑passing algorithm used to compute approximate marginal distributions in factor graphs that contain cycles. It extends the standard belief propagation algorithm, which is exact on tree‑structured graphs, to more general graph topologies where exact inference is intractable.

In belief propagation, two kinds of messages are exchanged:

- **Variable-to-factor** — summarizing a variable’s current belief based on all other connected factors.

  ![Variable to Factor](doc/img/lbp_variable_to_factor.svg)

- **Factor-to-variable** — summarizing how a factor constrains a variable, given the other variables in that factor.

  ![Variable to Factor](doc/img/lbp_factor_to_variable.svg)



These messages are updated repeatedly until they converge or until a fixed number of iterations is reached. Once messages stabilize, the approximate marginal distribution for a variable \(X\) is:


![Belief](doc/img/lbp_belief.svg)


Although LBP is not guaranteed to converge on graphs with cycles, it often produces stable and informative approximations in practice. Loopy provides a modular Rust implementation of these message‑passing rules, making it straightforward to explore different graph structures, factor definitions, and semiring choices.

## Core Abstractions

Loopy is organized around a small set of composable abstractions that mirror the structure of a factor graph and the algebra used for inference. These abstractions form the foundation of the library’s API:
- **Variables** — Represent the unknown quantities in a model. Each variable has a domain and participates in one or more factors.
- **Factors** — Functions over one or more variables, implemented using flexible Rust abstractions.
- **Factor graphs** — Connect variables and factors, defining the structure over which messages are exchanged.
- **Semirings** — Provide the algebra that governs how factors interact, how messages are combined, and how inference proceeds.

Together, these components define how a model is represented and how message passing is performed. The library builds on these abstractions to implement loopy belief propagation, custom message‑passing schedules, and future inference algorithms.

## Getting Started

Clone the repository and build the project with Cargo:

```bash
git clone git@github.com:crowetc/loopy.git
cd loopy
cargo build
cargo test
```
