# Project Context

This project supports experiments for the paper:

**Curvature-Based Impurity Design for Split Selection in Streaming Decision Trees**

The goal is to study how impurity-function geometry affects split selection in
streaming decision trees. The project is not primarily about proposing a new
streaming stopping rule. Instead, it keeps the operational split-triggering
mechanism fixed and isolates the role of impurity shape in separating competing
candidate splits.

## Core Idea

At a decision-tree node, two candidate splits `A` and `B` are compared through
their impurity reductions:

```text
F = M_A - M_B
```

The parent impurity term cancels exactly, so the comparison depends only on the
child-node impurity patterns induced by the competing splits.

In a local regime, where child class probabilities are close to the parent
probability `p`, the decision margin is approximately

```text
F ~= -0.5 * g''(p) * (V_A - V_B)
```

Here:

- `g` is the impurity function,
- `g''(p)` is its local curvature,
- `V_A` and `V_B` are variance-type quantities describing how strongly the two
  splits separate child-node class probabilities.

Thus, split discrimination is a second-order effect. For a fixed split geometry,
larger negative curvature `-g''(p)` gives a larger population margin between
competing splits.

## Theoretical Direction

The paper studies impurity functions under symmetry, normalization, and
concavity assumptions:

```text
g(0) = g(1) = 0
g(1/2) = 1
g(p) = g(1-p)
```

Under the classical full-domain formulation, normalized Gini impurity,

```text
g(p) = 4p(1-p)
```

has constant curvature `-g''(p) = 8` and is optimal for maximizing the weakest
available curvature over the whole interval `(0, 1)`.

The restricted-domain viewpoint focuses instead on the operational uncertainty
interval:

```text
[epsilon, 1 - epsilon]
```

The motivation is that nearly pure nodes are often less important for further
splitting. By optimizing curvature only on this central interval, one can design
impurity functions with stronger split-discrimination power where decisions are
actually made.

Important impurity families for experiments include:

- normalized Gini,
- normalized entropy,
- misclassification impurity,
- globally concave restricted quadratic impurities,
- locally concave restricted quadratic impurities,
- globally concave restricted entropy impurities,
- locally concave restricted entropy impurities.

## Experimental Goals

Experiments should test whether the curvature-based predictions are visible in
streaming decision trees.

Useful quantities to measure include:

- accuracy over a stream,
- number and timing of split checks,
- selected split attributes,
- empirical gain margins,
- wrong split rate when the population-best split is known,
- split reliability as a function of sample size,
- margin behavior as parent probability changes,
- behavior under label noise,
- behavior as `epsilon` changes for restricted-domain impurities.

The preferred experimental style is controlled comparison: keep the tree,
streaming protocol, and split-triggering heuristic fixed, then vary only the
impurity function.

## Synthetic Data Direction

Synthetic streams should make the true split geometry inspectable or
controllable. Random-tree streams are useful for VFDT-style experiments, but
additional generators may be needed for direct tests of the theory.

Good synthetic settings include:

- a fixed hidden random tree, inspired by Domingos-style random tree generators,
- controlled binary splits with known child probabilities,
- paired candidate attributes with known variance terms `V_A` and `V_B`,
- streams where parent probability `p` can be swept across the interval,
- streams with controllable label noise and irrelevant features.

## Code Design Direction

Impurity implementations should expose a common interface suitable for both
tree training and geometric analysis:

```python
value(p: float) -> float
second_derivative(p: float) -> float
```

If useful later, the interface may also include:

```python
first_derivative(p: float) -> float
curvature(p: float) -> float
```

where `curvature(p)` returns `-second_derivative(p)`.

Tree code should remain simple enough that experiments are easy to interpret.
The important scientific comparison is between impurity measures and split
criteria, not between heavily optimized tree implementations.

## Current Project State

The current codebase contains:

- a basic VFDT-like streaming decision tree,
- binary-feature sufficient statistics,
- Gini and entropy impurity implementations,
- a random-tree synthetic stream generator,
- a fixed-threshold binarizer for numeric attributes,
- prequential evaluation utilities,
- a local virtual environment with dependencies listed in `requirements.txt`.

The implementation is intentionally small. It should evolve toward clearer
experimental instrumentation before becoming more feature-complete.
