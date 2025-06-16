### SoftSort

[SUBTITLE: Reformulation into and modelling in the space of vectors, permutations are vectors as well]

- next we leave the terrytory of manifold optimization. However not entirely in spirit. Just as manifold optimization allowed us to consider the geometry and the relevant spasubspace of the feasible regiion, we might try to ask ourselves, how we could decrease the dimension even further. From the vector representation of the permutations, we knwow, that they can be represented and embedded ina space of dimension _n_. The problem lies, however in the way howto refoermulate the problem to "talk with" our objective function which acce[ts aa permutation vector.

- supposewe are able to transform a permutation vector to permutation matrix with one of the wauys described earlier, we could thenuse it to compte teh objective function, however, we would maybe like to use a gradient based method to then inform te parameter space where to move next t. To do this, we would want the map to be differentiable to provide us  with a gradient that is able to flow thorugh the process back in a kind of backpropagation. If we found such a map, then we could use some simple first order method to solve the problem.

- In search for such a function, we have been inspired by the SoftSort from the study of neural network. This function has been studient as a tool to learn latent [TODO: what does latent mean?] permutaions in the data (i.e. to learn permutations). It is really a differentiable extension of the argsort function that outputs the permutation as a permutation matrix. How close it brings us to the permutation is again controlled by a temperature parameter \tau. 

- HOPE: drastically reduce the dimension of the problem and still obtain good enough algorithm and results
  - QUESTION: is this formulation strong enough to yield satisfying results compared with other high dimensional methods
 
- The basic idea is really quite simple:
  - [TODO(description): Include simple exposition of the method and function here]
  - [TODO(figure): Include a schemantic of the process of how to construct a ]
  - [TODO(theory): What are the most basic fundamental theoretical results that ground the correctness and justify the choice and use of the method]
  - [TODO(practice): Include results where the authors report great success with the method and more importantly where it does not work that well]

> ## SoftSort: A Differentiable Relaxation for Permutation Optimization
> 
> ### 1. Core Concept and Mathematical Description
> 
> **The Fundamental Problem**: You need to optimize over permutations while maintaining differentiability for gradient-based methods. This reduces the problem from optimizing over n×n permutation matrices (n² - n constraints) to optimizing over n-dimensional vectors.
> 
> **The SoftSort Function**: Given a vector s ∈ ℝⁿ and temperature parameter τ > 0:
> 
> ```
> SoftSort_τ(s) = softmax((-d(sort(s)𝟙ᵀ, 𝟙sᵀ))/τ)
> ```
> 
> Where:
> - `sort(s)` sorts s in descending order
> - `d(·,·)` is a distance function (typically |·| or |·|²)
> - `𝟙` is the all-ones vector
> - `softmax` is applied row-wise
> 
> **In matrix form** (as in your implementation):
> ```
> P_ij = exp(-|s_j - s_[i]|/τ) / Σₖ exp(-|s_k - s_[i]|/τ)
> ```
> where s_[i] is the i-th largest element of s.
> 
> ### 2. Key Theoretical Properties
> 
> **Theorem 1 (Main Properties)**:
> 1. **Unimodal Row Stochastic (URS)**: SoftSort_τ(s) produces matrices that are:
>    - Non-negative: all entries ≥ 0
>    - Row-stochastic: rows sum to 1
>    - Have unique row-wise argmax that recovers the true permutation
> 
> 2. **Convergence**: As τ → 0⁺, SoftSort_τ(s) → P_argsort(s) (the true permutation matrix)
> 
> **Interpretation**: The i-th row of SoftSort represents a probability distribution over which element should be in position i.
> 
> ### 3. Dimensionality Reduction and Optimization Framework
> 
> **The Key Insight**: Instead of optimizing over O(n²) variables with complex constraints, you optimize over n unconstrained variables:
> 
> ```
> Original problem: min f(P) subject to P ∈ 𝒫ₙ (permutation matrices)
> Relaxed problem:  min f(SoftSort_τ(s)) over s ∈ ℝⁿ
> ```
> 
> **Gradient Flow**: The mapping s → SoftSort_τ(s) → f(SoftSort_τ(s)) is differentiable, enabling backpropagation through the sorting operation.
> 
> ### 4. Integration with Annealing Schemes
> 
> Your implementation demonstrates a powerful combination with simulated annealing:
> 
> ```
> τ(t) = τ_initial × decay_function(t/T)
> ```
> 
> Where decay_function can be:
> - **Cosine**: τ₀ × 0.5(1 + cos(πt/T))
> - **Exponential**: τ₀ × exp(-αt)
> - **Linear**: τ₀ × (1 - t/T)
> 
> **The annealing strategy**:
> 1. Start with high τ (exploration phase): soft permutations
> 2. Gradually decrease τ (exploitation phase): sharper permutations
> 3. Final τ ≈ 0: nearly discrete permutations
> 
> ### 5. Latent Permutations - General Explanation
> 
> **Latent permutations** refer to hidden orderings in data that aren't directly observable but must be inferred. In your context:
> - The "true" permutation that minimizes your objective is latent
> - SoftSort allows learning this permutation through gradient descent
> - The parameter vector s encodes the permutation implicitly
> 
> ### 6. Mathematical Diagram (as code)
> 
> ```python
> # Mathematical flow of SoftSort optimization
> """
> Input: Objective function f, Initial parameters s₀ ∈ ℝⁿ
> 
>    s ∈ ℝⁿ                    [n parameters]
>      ↓
>    sort(s) + distances       [O(n log n) + O(n²)]
>      ↓
>    SoftSort_τ(s) ∈ ℝⁿˣⁿ     [soft permutation matrix]
>      ↓
>    f(SoftSort_τ(s))         [objective evaluation]
>      ↓
>    ∇_s f                    [gradient w.r.t. s via backprop]
>      ↓
>    s ← s - α∇_s f          [parameter update]
> 
> Temperature annealing: τ = τ(iteration)
> """
> ```
> 
> ### 7. Performance Characteristics and Limitations
> 
> **Strengths**:
> - **Dimension reduction**: n parameters instead of n²
> - **Speed**: O(n²) complexity vs O(Ln²) for Sinkhorn-based methods
> - **Simplicity**: 3 lines of code for core function
> - **Gradient quality**: Smooth gradients throughout optimization
> 
> **Limitations**:
> - **Approximation quality**: Depends heavily on temperature τ
> - **Local minima**: Non-convex optimization landscape
> - **Problem-specific**: Works best when objective is smooth w.r.t. permutations
> 
> **Empirical Performance** (from paper):
> - Sorting tasks: 92% accuracy for n=3, decreasing to 25% for n=15
> - Speed: 6× faster than NeuralSort for n=4000
> - Convergence: Similar number of iterations as competing methods
> 
> ### 8. When to Use SoftSort vs. Manifold Methods
> 
> **Use SoftSort when**:
> - Problem size is moderate (n ≤ 100)
> - You need fast iterations
> - The objective is relatively smooth
> - You can tolerate approximate solutions
> 
> **Use Manifold Methods when**:
> - You need exact feasibility guarantees
> - The problem has additional structure to exploit
> - Accuracy is more important than speed
> 
> ### 9. Connection to Your Approximate Symmetry Problem
> 
> For your specific application to approximate symmetry (minimizing ||PAP^T - A||²):
> - SoftSort provides a natural parameterization for P
> - The annealing scheme helps escape local minima
> - The differentiability enables efficient optimization
> - The penalty term c·diag(P) can be incorporated seamlessly
> 
> This approach transforms your constrained optimization problem into an unconstrained one in ℝⁿ, potentially offering computational advantages over manifold methods while maintaining solution quality through careful temperature annealing.

