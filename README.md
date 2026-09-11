# Loopy

Loopy is a Rust library for performing **loopy belief propagation (LBP)** on factor graphs. It provides clear, modular abstractions for representing factorized functions and implementing message-passing inference.

Factor graphs are used across many domains, including probabilistic inference, optimization, constraint satisfaction, coding theory, robotics, and computer vision. Loopy aims to support this breadth while remaining simple, expressive, and performant.

## Motivation

Many real-world problems involve a large global function that can be decomposed into smaller local functions. Factor graphs provide a natural representation for this structure, while message-passing algorithms provide an efficient way to propagate information through it.

Belief propagation performs exact inference on tree-structured factor graphs. Loopy belief propagation extends the same message-passing rules to graphs containing cycles, making it useful for problems where exact inference is computationally impractical.

Despite their broad applicability, modular implementations of factor-graph inference are relatively uncommon—particularly in systems languages that emphasize both safety and performance.

Loopy aims to fill that gap.

The library is built around several core goals:

- **Clear abstractions** — variables, factors, semirings, graphs, and messages are explicit concepts rather than details hidden behind opaque APIs.

- **Modularity** — factor representations, inference algorithms, message-passing schedules, and algebraic choices can evolve independently.

- **Generality** — the underlying abstractions are intended to support more than a single inference algorithm or factor representation.

- **Practicality** — suitable for experimentation and research today, while designed with future production applications in mind.

- **Rust-native design** — leveraging strong typing, memory safety, explicit ownership, and predictable performance.

Loopy is intended to make the mechanics of message passing understandable without sacrificing the architectural flexibility needed for more advanced inference systems.

## Factor Graphs: The Big Picture

A factor graph is a bipartite graphical model that expresses a global function as a collection of local relationships.

Formally, a factor graph can be written as

![Factor Graph](docs/img/factor_graph.svg)

where:

- <img src="https://latex.codecogs.com/svg.image?V%20=%20%5C%7BX_1,%20X_2,%20%5Cldots,%20X_n%5C%7D"
style="vertical-align: middle;" width="160"> is a set of **variable nodes**

- <img src="https://latex.codecogs.com/svg.image?%5CPhi%20=%20%5C%7B%5Cphi_1,%20%5Cphi_2,%20%5Cldots,%20%5Cphi_m%5C%7D"
style="vertical-align: middle;" width="160"> is a set of **factor nodes**

- <img src="https://latex.codecogs.com/svg.image?E%20%5Csubseteq%20V%20%5Ctimes%20%5CPhi"
style="vertical-align: middle;" width="90"> is the set of **edges** connecting factors to the variables in their scopes

The graph structure indicates which variables participate in which local relationships. Together, those relationships define the global function:


![Global function](docs/img/global_function.svg)



This decomposition enables local computation and efficient propagation of information through the graph.

## What’s a Factor?
A factor is a **function** over zero or more variables. Depending on the application, factors may represent probabilities, compatibility scores, costs, energies, constraints, or other relationships.

Each factor has a **scope**: the set of variables on which it depends.

Conceptually, a factor can be written as

![Factor Definition](docs/img/factor_definition.svg)

where <img src="https://latex.codecogs.com/svg.image?\mathcal{K}" style="vertical-align: middle;" width="12">  is the value domain used by the inference system.

Loopy is built around two fundamental factor operations:

- **combine** — aggregates information from multiple factors
- **reduce** — eliminates one or more variables from a factor

Their behavior is determined by the semiring used for inference:

| Inference | Combine | Reduce |
|---|---|---|
| Sum-product | Multiply | Sum |
| Max-product | Multiply | Max |
| Constraint satisfaction | AND | OR |

In semiring terms, `combine` is the multiplicative operation and `reduce` is the additive operation. This allows the same factor and graph abstractions to support different forms of inference.

## Loopy Belief Propagation

Loopy belief propagation is an iterative message-passing algorithm for performing approximate inference on factor graphs that contain cycles.

Two kinds of messages are exchanged.

- **Variable-to-factor** — summarizes the information arriving at a variable from all neighboring factors except the destination factor.

  ![Variable to Factor](docs/img/lbp_variable_to_factor.svg)

- **Factor-to-variable** — summarizes how a factor influences a variable after incorporating information from the other variables in its scope.

  ![Factor to Variable](docs/img/lbp_factor_to_variable.svg)



These messages are updated repeatedly until they converge or another stopping condition is reached. Once messages stabilize, the approximate marginal distribution for a variable \(X\) is:


![Belief](docs/img/lbp_belief.svg)


Although LBP is not guaranteed to converge on graphs with cycles, it often produces stable and informative approximations in practice. Loopy provides a modular Rust implementation of these message‑passing rules, making it straightforward to explore different graph structures, factor representations, and semiring choices.

## Core Abstractions

Loopy is organized around a small set of composable abstractions that mirror factor-graph inference:

- **Variables** — Represent unknown quantities and their domains.
- **Factors** — Represent local functions over variables.
- **Factor graphs** — Connect variables and factors, defining the structure over which messages are exchanged.
- **Semirings** — Define the algebra used by factor operations and message passing.
- **Schedules** — Determine how messages are updated during iterative inference.

Together, these components separate model representation, factor algebra, and message-passing strategy while keeping the mechanics of inference explicit.

## Getting Started

Clone the repository and build the project with Cargo:

```bash
git clone git@github.com:crowetc/loopy.git
cd loopy
cargo build
cargo test
```

## Contact

Loopy is an ongoing project reflecting my interests in probabilistic inference, high-performance systems programming, and algorithm development.

Questions, feedback, and discussion are always welcome.

**tim.crowe.dev@proton.me**