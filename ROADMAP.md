# Loopy Roadmap

Loopy is under active development. The immediate goal is a clean, tested, and well-documented implementation of discrete loopy belief propagation. Longer-term development will expand Loopy's inference capabilities, scheduling models, performance, and interoperability while stabilizing the public API.

Issue links will be added as work is planned.

## 0.1 — Discrete LBP

**Status:** In progress

Establish the core discrete belief propagation library and a usable initial API.

- [x] Discrete factor representations and algebra
- [x] Factor graph construction
- [x] Message infrastructure
- [x] Synchronous message passing
- [x] Sum-product and max-product inference
- [x] Belief computation and convergence control
- [x] Unit and integration tests
- [x] Factor operation benchmarks
- [x] Simple example
- [x] Continuous integration
- [ ] Harden discrete inference semantics and edge-case behavior ([#TBD](#))
- [ ] Expand loopy-graph inference tests ([#TBD](#))
- [ ] Add a worked example containing a cycle ([#TBD](#))
- [ ] Review and stabilize the 0.1 public API ([#TBD](#))

## Scheduling

**Status:** Planned

Expand the scheduling model beyond the initial synchronous implementation and establish precise update and convergence semantics.

- [ ] Add message damping ([#TBD](#))
- [ ] Add serial belief propagation ([#TBD](#))
- [ ] Add residual belief propagation ([#TBD](#))
- [ ] Support configurable message schedules ([#TBD](#))
- [ ] Support configurable convergence metrics ([#TBD](#))
- [ ] Add inference progress and convergence diagnostics ([#TBD](#))

## Performance

**Status:** Early work

Characterize and improve inference performance while preserving clear factor and message abstractions.

- [x] Benchmark core factor operations
- [ ] Characterize message-passing and end-to-end inference performance ([#TBD](#))
- [ ] Characterize scaling with graph size, connectivity, and variable cardinality ([#TBD](#))
- [ ] Reduce avoidable allocations and message-passing overhead ([#TBD](#))
- [ ] Improve factor operation performance based on profiling ([#TBD](#))

## Generalized Inference

**Status:** Future

Extend Loopy to support richer models and inference capabilities beyond discrete belief propagation.

- [ ] Add boolean inference ([#TBD](#))
- [ ] Add logical and constraint factors ([#TBD](#))
- [ ] Add Gaussian inference ([#TBD](#))
- [ ] Support hybrid discrete/continuous inference ([#TBD](#))
- [ ] Support templated and repeated graph structures ([#TBD](#))
- [ ] Add temporal and online inference, including filtering and smoothing ([#TBD](#))
- [ ] Add parameterized and reusable factor models ([#TBD](#))

Generalized inference does not need to be complete for 1.0. Before stabilizing the public API, however, the core abstractions should be exercised against at least one substantially different factor or domain representation.

## Parallelism

**Status:** Future

Explore parallel execution of factor operations and message passing to improve inference performance and scalability.

- [ ] Identify parallelizable inference operations and synchronization boundaries ([#TBD](#))
- [ ] Add parallel factor operations where beneficial ([#TBD](#))
- [ ] Add parallel synchronous message computation ([#TBD](#))
- [ ] Evaluate parallel execution for alternative schedules ([#TBD](#))
- [ ] Characterize parallel scaling and synchronization overhead ([#TBD](#))

## Machine Learning Integration

**Status:** Future

Explore integration between factor-graph inference and learned representations and models.

- [ ] Tensor interoperability ([#TBD](#))
- [ ] Learned factors ([#TBD](#))
- [ ] Embedding-conditioned inference ([#TBD](#))
- [ ] Differentiable inference ([#TBD](#))
- [ ] Structured neural decoding ([#TBD](#))
- [ ] Deep learning framework integration ([#TBD](#))

## Toward 1.0

**Status:** Future

A 1.0 release represents API stability and confidence in Loopy's core abstractions rather than completion of every planned feature.

Before 1.0:

- [ ] Discrete inference semantics are well-defined and thoroughly tested
- [ ] Scheduling and convergence semantics are stable and documented
- [ ] Public error and panic behavior is deliberate and documented
- [ ] Core abstractions are validated against at least one additional factor or domain representation
- [ ] End-to-end performance is characterized and major bottlenecks are understood
- [ ] Public APIs are stable and comprehensively documented
- [ ] Worked examples cover both basic and genuinely loopy inference
- [ ] CI and release processes protect the stable API
- [ ] Loopy has been exercised in meaningful applications

At 1.0, users should be able to depend on Loopy's core belief propagation APIs as stable and well-tested, while additional factor representations, scheduling strategies, inference capabilities, and integrations may continue to evolve.

---

This roadmap describes intended direction rather than a fixed release schedule. Priorities may change as implementation experience exposes new requirements.
