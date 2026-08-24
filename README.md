# Loopy

Loopy is a Rust library for performing **loopy belief propagation (LBP)** on factor graphs.

The project is being developed as an educational exploration of probabilistic inference, with an emphasis on flexible factor abstractions, factor graphs, and message-passing algorithms.

## What is Loopy?

Loopy is built around a few core abstractions:

- **Variables** represent the unknown quantities in a graphical model.
- **Factors** represent functions over one or more variables.
- **Factor graphs** connect variables to the factors in which they participate.
- **Semirings** provide the algebra used when combining and reducing factors.

These abstractions form the foundation for implementing message passing and, ultimately, loopy belief propagation.

## Getting Started

Clone the repository and build the project with Cargo:

```bash
git clone git@github.com:crowetc/loopy.git
cd loopy
cargo build
cargo test
```
