# Introduction

# Preliminaries

## Notation

- [TODO: This can be generated from at the end from all the text used and collected from it]

# Approximate Symmetry Problem

[DESC: SHORT CHAPTER INTRODUCTION -> it is not short currently]

- introduce goals
- Goal: Present the problem of approximate symmetries from a different point of view
  - [DESC: WHat is my motivation?] THe main motivation for the sutdy of approximate symmetries comes from the fact that the problems can arrise in real world situations and the strictness and the exactness of the original isomorphism problem cannot capture or rather does not satisgyingly capture the approximativeness of the the problem and the inexactness of the whole matter. What DO I mean by this. The setup is fairly simple./. Imagine we are interested in studying the symmetries of the graph. The reasons why we would want to do this come later [TODO: Introduce the reasons why this is important first of interesting first and provide e tiwht the reasons later]. WT We start with a graph. This graph can be symmetric. The problem of symmetry is solved by the automorphism problem. Automorphism are a solved problem and the enttire automorphism group can be already computed [TODO: include sources that describe the problems of solving automorhphisma dnd automorphism groups; do not forget to include the computational complexity of the problem and the software packages that can be used to solve the problem.]. The exact symmetry is nice, however do consider this simple example (provided by LIu). COnsider a square n.n grid. This grid does have many symmetries (rotational, diagonal, reflexion), However, COnsider the case of removing one edge from the grid. Now all the symmetries are broken and the graph possesis only a single automorphism (namely the identity).  However, looking at the grid without one edge, we do get the distinct feeling that despite the missing edge, it is stile _quite symmetric_. WFollowing this idea, Liu suggest that we can extend the notion of symmetry using the error metrixc E(A,P), or edge count \mathcal{E}(A, P), lea, leading to the normalized symmetry measure S(A).This metric extends the notion of symmetry as symmetriies are the solutions attaing the 0 value, other permutations lead to a solution attaining positive values and we can interpret the smaller values as solutions (or permutations) closer to some symmetry.

  - [TODO(motivation): why is it cool to study approximate symmetries, symmetry of brain, the symmetry (or lack thereof) can provide us with insights about the sate of degradation of the brain]


  - Present the work of Liu, Vandenberghe, Matej Straka and Franta Szcepanik
    - FRantisek Szcepanik showed a combinatorial approach to this problem , we will explore the relaxation of this problem
      - [TODO(literature-review): Include results of the Frantisek Szcepanik thesis]
      - Do find other results he cites

    - [TODO(literature-review): Figure what did Matej Straka do in his thesis and what did he find out]
    - [TODO(literature-review): Vandenberghe]

  - intention to look at the problem from a different angle
    - Liu suggests in his thesis to solve this problem from a combinatorial perspective using local search through the algorithm of simmulated annealing. 

  - introduce the problem in relation to different known studiend problems and how the problem of approximate symmetries naturally arises when following the logic of these problems
    - show how these problems differ and how do they relate
  - 

## Problem overview

(and literature overview)

- introduce the problem of graph symmetries (Why? Oh woe why?)

- quick overview (setup, playground, reiteration)
- goal is to look for approximate symmetries in networks that do not necessarily have one in a _strinct_ sense

- introduce the measure proposed by approximate symmetries proposed by Liu
  - what does it represent?
  - normalized appx symmetry coefficient
  - problems and disadvantages with this definition (it has obvious optimum)

- How Liu suggest to solve this problem? -> Simmulated Annealing
  - Work of Franta Szcepanik
    - [TODO: Include and analyse the results of the work of Franta Sczepanik]

### Graph Isomorphism

- describe the problem of graph isomorphism
- present the combinatorial nature of this problem
  - present how it compares to the problem
- [TODO(problem): Include formal problem description]
- [TODO(search, literature-review): include the recent result that graph isomorphism is quasi-polynomial]
- [TODO: Research? How is the quasipoly result achieved? Are any relaxations involved?]

### Graph Matching Problem (GMP)

- Present the motivation of this problem
- [TODO: Find some canonical, summary report, review paper for GMP]
- definition of the problem, motivation, idea
  - [TODO(problem): Include formal description of the problem]
- generalisation of the graph isomorphism (we allow errors)
- complexity of this problem [TODO(literature): Find citations for the complexity of this problem]


### Quadratic Assignment Problem (QAP)

- Present the original motivation of this problem [TODO: Find landmark study]
- [TODO(problem): Add formal definition of this problem]
  - definition, matrix form, 
- relation to GMP

- add reformulation to GMP -> QAP (mathematical)

- compare the properties of QAP and GMP compare the properties of both of these problems
  - binary QP + LC => NP hard to find global optimumo

#### Linear Assignment Problem

- present the problem, show that it can be solved in poly time using LP, moreover using hungarian algorithm
- [QUESTION: Does this relate to the QAP?]

  
### Representing Permutations

- permutation vectors
  - permutation of [n]
  - the argsort of a vector from R^n
- permutation matrices
- permutation vectors and matrices can be transformed one to each other
  - [TODO(math): How to transform permutatin matrix to permutation vec]
  - [TODO(math): transform permutation vec to permutation matrix]
- honorable mention, representing relaxed permutagedron through the extension polytope using sorting networks

- birkhoff polytope 
  - [TODO:(find the sources for the results of birkhoff)]
  - it has nice and concise representation using linear constraints
- explain how the permutations can be embedded in the space of doubly stochastic metrices
  - which in turn is embedded in the sapce of singly stochastic matrices which(which does not have the save nice properteis)
- explain the dimensionality of the space of permutations and the dimensionality of the space of doubly stochastic matrices

- relax
  - why do we relax? 
    - we hope that we get to exploit the geometry of the space to find solution faster
    - we hope to exploit differentiability and speed of methods of continouous optimization; continouous optimization is awesome because it goes brrrr
  - How do we relax?
    - -> of course the birkhoff polytope; but we can choose GMP or QAP

- relaxed QAP (rQAP) ia a QP with linear constraints
  - !! it is not generally convex (HESSIAN is indefinite)
  - non-convex quadratic optimization is NP-hard
    - we can try to use the tools of continouous optimization to look for at least local optima


### Why relax QAP instead of GMP
- Present the result of [TODO: Insert author] fastQAP and relax at your own risk [paper]
- QAP approximation provides GMP approximation

- Why relax QAP instead of GMP?

### Problems Specific to Approximate Symmetries

#### Choosing a Starting Point

#### Excluding Identity

- approximate symmetry problem is in a sense not well defined since it has the obvious solution of "identity"
  - we want to exclude this identity
    - in iterative local search methods, this is simple (disregard solutions with many fixed points)

- using penalization

- [TODO: Future work, describing the convex hull of permutations]
  - permutations containing identity can be detected through tournaments using LP, one of the questions posed by birkhoff

#### Not Well-Defined Problem

- what is the "best"solution?
  - Is is the solution attaining the least approximate number? counterexample to this would be
  - - >Or is it a solution that might not attain
  - we might want to require 
  - [NOTE(future-work): ]


#### Mapping Back to Permutations

- hungarian algorithm and the "closest permutation"

  >   # Understanding "Closest" Permutation Matrices: A Deeper Look at Distance Metrics
  > 
  > When finding the "closest" permutation matrix to a doubly stochastic matrix, the mathematical notion of "closeness" or "distance" deserves careful explanation. Let me clarify why the formulations I mentioned are meaningful and how they relate to intuitive notions of distance.
  > 
  > ## The Inner Product Formulation
  > 
  > The formulation:
  > 
  > **Maximize ∑ᵢⱼ D[i,j] × P[i,j]**
  > 
  > This represents the inner product (dot product) between matrices D and P, viewing them as vectors in R^(n²). The motivation behind this approach is intuitive: we want to place the 1's in the permutation matrix P at positions where the doubly stochastic matrix D has its largest values.
  > 
  > Think of it this way: the doubly stochastic matrix D assigns weights (between 0 and 1) to all possible position assignments. The permutation matrix P can only select n positions (placing 1's), so we naturally want to select the positions with the highest weights to maximize the total weight captured.
  > 
  > ## The Euclidean Distance Connection
  > 
  > The more standard way to measure distance between matrices is using the Euclidean (Frobenius) distance:
  > 
  > **||D-P||²_F = ∑ᵢⱼ (D[i,j] - P[i,j])²**
  > 
  > Let's expand this squared distance:
  > 
  > **∑ᵢⱼ (D[i,j] - P[i,j])² = ∑ᵢⱼ D[i,j]² + ∑ᵢⱼ P[i,j]² - 2∑ᵢⱼ D[i,j]P[i,j]**
  > 
  > Now, notice that:
  > - ∑ᵢⱼ D[i,j]² is constant (D is fixed)
  > - For any permutation matrix P, ∑ᵢⱼ P[i,j]² = n (since P contains exactly n ones)
  > 
  > Therefore:
  > **minimizing ||D-P||²_F ⟺ maximizing 2∑ᵢⱼ D[i,j]P[i,j] ⟺ maximizing ∑ᵢⱼ D[i,j]P[i,j]**
  > 
  > This shows that maximizing the inner product is equivalent to minimizing the Euclidean distance between the matrices.
  > 
  > ## The Alternate Formulation
  > 
  > When I mentioned:
  > 
  > **Minimize ∑ᵢⱼ (1-D[i,j]) × P[i,j]**
  > 
  > This can be rewritten as:
  > **∑ᵢⱼ (1-D[i,j]) × P[i,j] = ∑ᵢⱼ P[i,j] - ∑ᵢⱼ D[i,j] × P[i,j] = n - ∑ᵢⱼ D[i,j] × P[i,j]**
  > 
  > Since n is constant (the dimension of the matrix), minimizing this expression is equivalent to maximizing ∑ᵢⱼ D[i,j] × P[i,j].
  > 
  > ## Why These Distances Make Sense in Applications
  > 
  > In the context of problems like seriation (as mentioned in the documents you provided), finding the closest permutation matrix helps recover the underlying ordering of objects when given a similarity or distance matrix.
  > 
  > For example:
  > - In genomic sequencing, the doubly stochastic matrix might represent probabilistic relationships between fragments
  > - In archaeology, it might represent similarity between artifacts
  > 
  > The Hungarian algorithm finds the permutation that best preserves these relationships by maximizing the correspondence between high-value entries in D and the positions of 1's in P.
  > 
  > ## Alternative Distance Metrics
  > 
  > While the Euclidean distance is most common, other metrics could be used:
  > - **L₁ norm**: ∑ᵢⱼ |D[i,j] - P[i,j]|
  > - **Matrix approximation error**: measured in spectral or operator norms
  > 
  > Each would lead to different notions of "closest" permutation matrix and potentially different algorithms for finding them. The Hungarian algorithm specifically solves the maximum weight bipartite matching problem, which corresponds to the Euclidean distance minimization.
  > 
  > Does this help clarify the mathematical notion of "closest" and why maximizing the inner product makes intuitive sense as a distance metric?


- The Rearrangement Inequality -> closest permutation for permutation vector

>   - Statement of the Rearrangement Inequality
> The rearrangement inequality states that for any two sequences of real numbers a1,a2,…,ana_1, a_2, \ldots, a_n
> a1​,a2​,…,an​ and b1,b2,…,bnb_1, b_2, \ldots, b_n
> b1​,b2​,…,bn​:
> 
> If a1≤a2≤…≤ana_1 \leq a_2 \leq \ldots \leq a_n
> a1​≤a2​≤…≤an​ and b1≤b2≤…≤bnb_1 \leq b_2 \leq \ldots \leq b_n
> b1​≤b2​≤…≤bn​ are the elements arranged in non-decreasing order, and aσ(1),aσ(2),…,aσ(n)a_{\sigma(1)}, a_{\sigma(2)}, \ldots, a_{\sigma(n)}
> aσ(1)​,aσ(2)​,…,aσ(n)​ is any permutation of the first sequence, then:

> Theorem: For sequences a1≤a2≤…≤ana_1 \leq a_2 \leq \ldots \leq a_n
> a1​≤a2​≤…≤an​ and b1≤b2≤…≤bnb_1 \leq b_2 \leq \ldots \leq b_n
> b1​≤b2​≤…≤bn​, the sum ∑i=1naibσ(i)\sum_{i=1}^n a_i b_{\sigma(i)}
> ∑i=1n​ai​bσ(i)​ is maximized when σ\sigma
> σ is the identity permutation and minimized when σ\sigma
> σ is the reverse permutation.
> a1bn+a2bn−1+…+anb1≤aσ(1)b1+aσ(2)b2+…+aσ(n)bn≤a1b1+a2b2+…+anbna_1 b_n + a_2 b_{n-1} + \ldots + a_n b_1 \leq a_{\sigma(1)} b_1 + a_{\sigma(2)} b_2 + \ldots + a_{\sigma(n)} b_n \leq a_1 b_1 + a_2 b_2 + \ldots + a_n b_na1​bn​+a2​bn−1​+…+an​b1​≤aσ(1)​b1​+aσ(2)​b2​+…+aσ(n)​bn​≤a1​b1​+a2​b2​+…+an​bn​

- von neumann decomposition

  > The Birkhoff-von Neumann decomposition (also called the Birkhoff-von Neumann theorem) is a fundamental result about doubly stochastic matrices that states:
  > Every doubly stochastic matrix can be expressed as a convex combination of permutation matrices.
  > Formal Statement
  > If P is an n × n doubly stochastic matrix, then there exist:
  > 
  > Permutation matrices P₁, P₂, ..., Pₖ
  > Non-negative weights λ₁, λ₂, ..., λₖ with Σλᵢ = 1
  > 
  > Such that: P = λ₁P₁ + λ₂P₂ + ... + λₖPₖ

  > ## **The Original Papers**
  > 
  > ### **1. Garrett Birkhoff (1946) - The Main Theorem**
  > **Birkhoff, Garrett (1946). "Tres observaciones sobre el algebra lineal" [Three observations on linear algebra]. Universidad Nacional de Tucumán, Revista Serie A, Vol. 5, pp. 147-151.**
  > 
  > This is the foundational paper where Birkhoff first stated and proved that every doubly stochastic matrix can be decomposed as a convex combination of permutation matrices.
  > 
  > ### **2. John von Neumann (1953) - Independent Development**
  > **von Neumann, John (1953). "A certain zero-sum two-person game equivalent to the optimal assignment problem." In Contributions to the Theory of Games, Vol. II, Annals of Mathematics Studies. Princeton University Press, pp. 5-12.**

- decomposition with random error


> # Reconstructing Permutations from a Sorting Network Solution
> 
> When we solve an optimization problem using the sorting network formulation, we end up with a point in the permutahedron that needs to be converted back into an actual permutation. Let me walk through this reconstruction process step by step.
> 
> ## What We Have After Optimization
> 
> After solving the optimization problem with the sorting network representation, we have:
> - A point x = (x₁, x₂, ..., xₙ) in the permutahedron
> - This point may not be a vertex of the permutahedron (i.e., may not be an actual permutation)
> - We need to extract a high-quality permutation from this point
> 
> ## Primary Reconstruction Method
> 
> The paper describes several approaches, with the simplest being:
> 
> ### Ordering-Based Recovery
> 1. Take the point x = (x₁, x₂, ..., xₙ) from the permutahedron
> 2. Sort the indices based on the values in x
> 3. The resulting order of indices gives the permutation
> 
> For example, if x = (5.2, 2.1, 8.7, 1.4) for n=4:
> - Sorting gives: x₄ < x₂ < x₁ < x₃
> - So the permutation would be π = (3, 2, 4, 1)
> - This means π(1) = 3, π(2) = 2, π(3) = 4, π(4) = 1
> 
> ## Enhanced Reconstruction with Randomization
> 
> To improve solution quality, the authors used a randomization technique:
> 
> 1. Add random Gaussian noise to each component of x
>    - They specifically used i.i.d. Gaussian noise with variance 0.5
> 2. Sort the indices based on these perturbed values
> 3. Evaluate the resulting permutation in the original objective function
> 4. Repeat multiple times and keep the best permutation found
> 
> This approach is effective because:
> - It allows exploration of nearby permutations in the solution space
> - It's computationally inexpensive compared to solving the original optimization problem
> - It helps overcome potential numerical imprecisions in the solver
> 
> ## Alternative Methods
> 
> The paper also mentions decomposition-based methods:
> 1. Express x as a convex combination of permutations: x = ∑ᵢλᵢπᵢ
> 2. This can be done efficiently in O(n²) time
> 3. Evaluate each permutation πᵢ in the objective function
> 
> However, the authors found this approach generally produced weaker solutions than the randomized approach.
> 
> ## Why This Works
> 
> The intuition behind these reconstruction methods is that:
> 1. The optimization pushes the solution toward good permutations
> 2. Points in the permutahedron that are close to optimal will be near optimal permutations
> 3. By sampling around the solution point, we can find high-quality permutations even if the original point wasn't exactly a permutation
> 
> The paper demonstrates that this overall approach—optimization over the sorting network representation followed by reconstruction—produces high-quality solutions to permutation problems while being computationally much more efficient than traditional methods.

### Summary

- Graph Isomorphism -> Graph Matching -> Quadratic Assignment -> Approximate Symmetries

- Current Solution approach and current solution concepts; include the description of Liu and the algotrightm of Vandenberghe

- describe the algorithm of vandenberge anew because it has mistakes
- inlude the work of simmulated annealing of Franta Sczepanik and the work of Matej Straka


# Optimization Methods

### Interior Point Methods

- Interior-point (or barrier) methods have proved to be as successful for nonlinear optimiza-
tion as for linear programming, and together with active-set SQP methods, they are currently
considered the most powerful algorithms for large-scale nonlinear programming [Nocedal]

 - The terms “interior-point methods” and “barrier methods” are now used interchangeably [Nocedal]

 - The term “interior point” derives from the fact that early barrier methods [98] did
not use slacks and assumed that the initial point x 0 is feasible with respect to the inequality
constraints ci (x) ≥ 0, i ∈ I [Nocedal]
- Most modern interior-point methods are infeasible (they can start from any initial point x 0 ) [Nocedal]

- instead of providing the full description of the algorithm refer to [Nocedal]

- IPOPT is a line-search interior point method
  - [TODO: What is a line-search interior point method and how does it differ from the basic setup]

- [TODO: Create a simple overview of the interior point algorithm and the basic approach maybe using the hartman presentations]

> # Interior Point Methods for Constrained Optimization
> 
> Interior point methods are powerful algorithms for solving constrained optimization problems. Unlike active set methods that move along the boundary of the feasible region, interior point methods travel through the interior of the feasible region toward the optimal solution.
> 
> ## The General Problem Setup
> 
> Let's start with the general form of a constrained optimization problem:
> 
> ```
> minimize    f(x)
> subject to  h(x) = 0     (equality constraints)
>             g(x) ≤ 0     (inequality constraints)
> ```
> 
> Where:
> - f(x) is the objective function (possibly nonconvex)
> - h(x) represents equality constraints
> - g(x) represents inequality constraints
> 
> For quadratic problems specifically, f(x) would take the form f(x) = (1/2)x^T Q x + c^T x, where Q may be indefinite (making the problem nonconvex).
> 
> ## The Barrier Approach
> 
> Interior point methods convert inequality-constrained problems into a sequence of equality-constrained problems using barrier functions. The most common approach uses logarithmic barriers.
> 
> The key idea is to replace inequality constraints g(x) ≤ 0 with penalty terms in the objective function that grow infinitely large as x approaches the boundary of the feasible region. This creates a "barrier" that keeps the iterates inside the feasible region.
> 
> The barrier problem becomes:
> 
> ```
> minimize    f(x) - μ ∑ log(-g_i(x))
> subject to  h(x) = 0
> ```
> 
> Where μ > 0 is the barrier parameter. As μ decreases toward zero, the solution of this barrier problem approaches the solution of the original problem.
> 
> ## The Central Path
> 
> The central path is a key concept in interior point methods. It's the set of minimizers of the barrier problem for different values of μ. The algorithm follows this central path toward the optimal solution as μ → 0.
> 
> In practice, we use a sequence of decreasing values for μ and solve (approximately) the barrier problem at each μ value, using the previous solution as a starting point for the next iteration.
> 
> ## Primal-Dual Interior Point Methods
> 
> Modern interior point methods typically use a primal-dual approach. Instead of directly solving the barrier problem, they solve the KKT (Karush-Kuhn-Tucker) conditions for the barrier problem.
> 
> For our problem, the KKT system looks like:
> 
> ```
> ∇f(x) + ∇h(x)λ + ∇g(x)z = 0
> h(x) = 0
> g(x) + s = 0
> ZSe = μe
> ```
> 
> Where:
> - λ are the Lagrange multipliers for equality constraints
> - z are the Lagrange multipliers for inequality constraints
> - s are slack variables for inequality constraints (g(x) + s = 0, s > 0)
> - Z = diag(z), S = diag(s), e is a vector of ones
> - ZSe = μe represents the complementarity condition
> 
> ## Newton's Method for the KKT System
> 
> The primal-dual interior point method applies Newton's method to this KKT system. Given the current iterate (x, λ, z, s), we compute the Newton direction (Δx, Δλ, Δz, Δs) by solving the linearized KKT system:
> 
> ```
> [ ∇²L    ∇h    ∇g    0  ] [ Δx  ]   [ -∇f - ∇h*λ - ∇g*z ]
> [ ∇h^T   0     0     0  ] [ Δλ  ] = [ -h                 ]
> [ ∇g^T   0     0     I  ] [ Δz  ]   [ -g - s             ]
> [ 0      0     S     Z  ] [ Δs  ]   [ -ZSe + μe          ]
> ```
> 
> Where ∇²L is the Hessian of the Lagrangian function with respect to x.
> 
> ## Handling Nonconvexity
> 
> When the problem is nonconvex (e.g., if Q in a quadratic problem is indefinite), the Hessian ∇²L may not be positive definite. This can lead to Newton directions that don't reduce the objective function or even move outside the feasible region.
> 
> To handle this, interior point methods for nonconvex problems often:
> 
> 1. **Modify the Hessian**: Add a positive definite matrix to make it positive definite
> 2. **Use line search or trust region methods**: Ensure progress toward the solution
> 3. **Apply regularization techniques**: Stabilize the linear system solution
> 
> ## IPOPT's Approach to Nonconvex Problems
> 
> IPOPT (Interior Point OPTimizer) is specifically designed to handle nonconvex problems. Here's how it deals with nonconvexity:
> 
> 1. **Inertia Correction**: IPOPT checks the inertia (number of positive, negative, and zero eigenvalues) of the KKT matrix. If it doesn't have the correct inertia, IPOPT adds a positive definite term to the Hessian block.
> 
> 2. **Globalization Strategy**: IPOPT uses a filter line-search method that considers both the objective function and constraint violation. This helps ensure convergence even for nonconvex problems.
> 
> 3. **Second-Order Correction**: When a step is rejected, IPOPT computes a second-order correction to try to overcome local nonlinearities.
> 
> 4. **Hessian Approximation**: If the exact Hessian isn't available or is too expensive to compute, IPOPT can use quasi-Newton approximations like BFGS or limited-memory BFGS.
> 
> ## For Quadratic Problems Specifically
> 
> When applying interior point methods to nonconvex quadratic problems (where Q is indefinite):
> 
> 1. The objective function is f(x) = (1/2)x^T Q x + c^T x
> 2. The Hessian of the objective is simply Q
> 3. If Q is indefinite, IPOPT will apply inertia correction to ensure the KKT system leads to descent directions
> 
> The Newton system for a quadratic program with linear constraints becomes simpler because the Hessian of the objective is constant (Q) and the constraint Jacobians are also constant.
> 
> ## Practical Implementation Considerations
> 
> When implementing interior point methods for nonconvex problems, several practical issues need attention:
> 
> 1. **Starting Point**: A good interior point is crucial; IPOPT uses a specialized algorithm to find one
> 
> 2. **Barrier Parameter Update**: How quickly to reduce μ affects convergence speed and robustness
> 
> 3. **Step Length Selection**: Choosing an appropriate step length along the Newton direction is critical
> 
> 4. **Termination Criteria**: When to stop the algorithm (e.g., KKT conditions satisfaction, small progress)
> 
> 5. **Handling Degenerate Cases**: Special techniques for problems with degeneracy or ill-conditioning
> 
> ## Example: Simple Nonconvex Quadratic Problem
> 
> Let me illustrate with a simple example of a nonconvex quadratic problem:
> 
> ```
> minimize    (1/2)x^T [ 1  0 ] x + [-2]^T x
>                      [ 0 -1 ]      [ 0]
> subject to  x_1 + x_2 = 1
>             x_1, x_2 ≥ 0
> ```
> 
> Here, Q = [1 0; 0 -1] is indefinite (eigenvalues +1 and -1), making this a nonconvex problem.
> 
> The barrier problem would be:
> ```
> minimize    (1/2)x^T [ 1  0 ] x + [-2]^T x - μ(log(x_1) + log(x_2))
>                      [ 0 -1 ]      [ 0]
> subject to  x_1 + x_2 = 1
> ```
> 
> As IPOPT solves this, it would:
> 1. Apply inertia correction to the Hessian when needed
> 2. Follow the central path as μ decreases
> 3. Use line search to ensure progress
> 4. Eventually converge to a local minimum (since global optimality can't be guaranteed for nonconvex problems)
> 
> ## Conclusion
> 
> Interior point methods, particularly as implemented in solvers like IPOPT, are powerful tools for solving constrained nonconvex quadratic problems. They work by following the central path defined by a sequence of barrier problems with decreasing barrier parameters.
> 
> For nonconvex problems, these methods incorporate special techniques like inertia correction, line search, and regularization to handle the challenges of indefinite Hessians. While they can't guarantee global optimality for nonconvex problems, they often find high-quality local solutions efficiently.
> 
> Would you like me to go deeper into any particular aspect of these methods? For example, I could elaborate more on IPOPT's specific implementation details, the mathematical derivation of the KKT system, or practical examples of these algorithms in action.

> # Derivation of the KKT System in Interior Point Methods
> 
> Let me walk you through how we derive the KKT system used in interior point methods, starting from the original optimization problem and working toward the final system of equations that algorithms like IPOPT solve at each iteration.
> 
> ## Step 1: Starting from the Original Problem
> 
> We begin with our constrained optimization problem:
> 
> ```
> minimize    f(x)
> subject to  h(x) = 0     (equality constraints)
>             g(x) ≤ 0     (inequality constraints)
> ```
> 
> ## Step 2: The Barrier Approach
> 
> To handle inequality constraints using interior point methods, we introduce the logarithmic barrier function. This transforms the problem into:
> 
> ```
> minimize    f(x) - μ ∑ log(-g_i(x))
> subject to  h(x) = 0
> ```
> 
> Where μ > 0 is the barrier parameter that we'll gradually reduce to zero.
> 
> ## Step 3: Introducing Slack Variables
> 
> To make the problem more manageable, we convert inequality constraints into equalities by introducing slack variables s > 0:
> 
> ```
> minimize    f(x) - μ ∑ log(s_i)
> subject to  h(x) = 0
>             g(x) + s = 0
>             s > 0
> ```
> 
> This reformulation maintains the barrier effect (through the logarithmic terms) while allowing us to work with equality constraints.
> 
> ## Step 4: Forming the Lagrangian Function
> 
> The Lagrangian for this barrier problem incorporates dual variables (Lagrange multipliers) for the constraints:
> 
> ```
> L(x,λ,z,s) = f(x) - μ ∑ log(s_i) + λ^T h(x) + z^T (g(x) + s)
> ```
> 
> Where:
> - λ are the Lagrange multipliers for equality constraints h(x) = 0
> - z are the Lagrange multipliers for the constraints g(x) + s = 0
> 
> ## Step 5: First-Order Optimality Conditions
> 
> The first-order necessary conditions for optimality (the KKT conditions) are obtained by taking derivatives of the Lagrangian with respect to all variables and setting them to zero:
> 
> ```
> ∇_x L = ∇f(x) + ∇h(x)λ + ∇g(x)z = 0
> ∇_λ L = h(x) = 0
> ∇_z L = g(x) + s = 0
> ∇_s L = -μs^(-1) + z = 0
> ```
> 
> Where s^(-1) denotes the vector [1/s₁, 1/s₂, ..., 1/s_m].
> 
> ## Step 6: Rearranging the Complementarity Condition
> 
> The last equation ∇_s L = -μs^(-1) + z = 0 can be rearranged to:
> 
> ```
> z_i s_i = μ  for all i
> ```
> 
> This is the perturbed complementarity condition, which we can write in matrix form as:
> 
> ```
> ZSe = μe
> ```
> 
> Where Z = diag(z), S = diag(s), and e is a vector of ones.
> 
> ## Step 7: The Full KKT System
> 
> Putting all conditions together, we get the KKT system for the barrier problem:
> 
> ```
> ∇f(x) + ∇h(x)λ + ∇g(x)z = 0
> h(x) = 0
> g(x) + s = 0
> ZSe = μe
> ```
> 
> This system characterizes the central path as μ varies from a positive value toward zero.
> 
> ## Step 8: Applying Newton's Method to the KKT System
> 
> To solve this nonlinear system at each iteration, we apply Newton's method. We linearize each equation around the current point (x, λ, z, s) to find a direction (Δx, Δλ, Δz, Δs).
> 
> For the first equation:
> ```
> ∇f(x+Δx) + ∇h(x+Δx)λ + ∇h(x+Δx)Δλ + ∇g(x+Δx)z + ∇g(x+Δx)Δz ≈ 0
> ```
> 
> After linearization:
> ```
> ∇f(x) + ∇²f(x)Δx + ∇h(x)λ + ∇²h(x)λΔx + ∇h(x)Δλ + ∇g(x)z + ∇²g(x)zΔx + ∇g(x)Δz ≈ 0
> ```
> 
> Rearranging and noting that the Hessian of the Lagrangian ∇²L = ∇²f(x) + ∑λᵢ∇²hᵢ(x) + ∑zᵢ∇²gᵢ(x):
> ```
> ∇²LΔx + ∇h(x)Δλ + ∇g(x)Δz ≈ -(∇f(x) + ∇h(x)λ + ∇g(x)z)
> ```
> 
> Similarly linearizing the other equations:
> ```
> ∇h(x)ᵀΔx ≈ -h(x)
> ∇g(x)ᵀΔx + Δs ≈ -(g(x) + s)
> ```
> 
> For the complementarity condition ZSe = μe, its linearization requires more care:
> ```
> (Z+ΔZ)(S+ΔS)e ≈ μe
> ```
> 
> Expanding and ignoring the second-order term ΔZ·ΔS:
> ```
> ZSe + ZΔSe + SΔZe ≈ μe
> ```
> 
> Since ZSe = μe already, and noting that ΔZe = Δz and ΔSe = Δs:
> ```
> ZΔs + SΔz ≈ 0
> ```
> 
> Or, if we're targeting a slightly smaller μ^+ < μ for the next iteration:
> ```
> ZΔs + SΔz ≈ μ^+e - μe = (μ^+ - μ)e
> ```
> 
> ## Step 9: The Newton System in Matrix Form
> 
> Combining all the linearized equations, we get the Newton system:
> 
> ```
> [ ∇²L    ∇h    ∇g    0  ] [ Δx  ]   [ -(∇f + ∇h*λ + ∇g*z) ]
> [ ∇h^T   0     0     0  ] [ Δλ  ] = [ -h                  ]
> [ ∇g^T   0     0     I  ] [ Δz  ]   [ -(g + s)            ]
> [ 0      0     S     Z  ] [ Δs  ]   [ (μ^+ - μ)e          ]
> ```
> 
> For a simpler Newton step aiming at the same μ value, the right-hand side of the last row becomes 0 instead of (μ^+ - μ)e.
> 
> ## Step 10: Refining for Implementation
> 
> For implementation in algorithms like IPOPT, the system is often rearranged. The most common form is:
> 
> ```
> [ ∇²L    ∇h    ∇g    0  ] [ Δx  ]   [ -∇f - ∇h*λ - ∇g*z ]
> [ ∇h^T   0     0     0  ] [ Δλ  ] = [ -h                 ]
> [ ∇g^T   0     0     I  ] [ Δz  ]   [ -g - s             ]
> [ 0      0     S     Z  ] [ Δs  ]   [ -ZSe + μ^+e        ]
> ```
> 
> Notice that the right-hand side of the last row has been rewritten as -ZSe + μ^+e. For a quadratic objective function f(x) = (1/2)x^T Q x + c^T x and linear constraints, this system simplifies because ∇²f(x) = Q and the second derivatives of the constraints are zero.
> 
> ## Special Case: Nonconvex Quadratic Problems
> 
> For a nonconvex quadratic problem, where Q has negative eigenvalues, IPOPT performs an inertia correction on the KKT matrix when needed. Specifically, it modifies the ∇²L block (which equals Q for quadratic problems with linear constraints) by adding a positive definite perturbation:
> 
> ```
> ∇²L ← ∇²L + δI
> ```
> 
> This ensures the KKT system yields a direction that makes progress toward a local minimum.
> 
> ## Conclusion
> 
> The KKT system in interior point methods is derived from:
> 1. Transforming the original problem using logarithmic barriers
> 2. Introducing slack variables to handle inequality constraints
> 3. Forming the Lagrangian of the barrier problem
> 4. Taking first-order optimality conditions
> 5. Applying Newton's method to solve these conditions iteratively
> 
> This approach provides a theoretically sound and computationally effective way to solve constrained optimization problems, including nonconvex quadratic programs. The system balances progress toward optimality and constraint satisfaction while navigating through the interior of the feasible region.

> 1. Problem Formulation
> We'll consider problems of the form:
> min⁡x∈Rnf(x)subject togi(x)≤0,i=1,…,mhj(x)=0,j=1,…,p\begin{align}
> \min_{x \in \mathbb{R}^n} \quad & f(x) \\
> \text{subject to} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
> & h_j(x) = 0, \quad j = 1, \ldots, p
> \end{align}x∈Rnmin​subject to​f(x)gi​(x)≤0,i=1,…,mhj​(x)=0,j=1,…,p​​
> Where:
> 
> f(x)f(x)
> f(x) is the objective function (possibly nonconvex)
> 
> gi(x)g_i(x)
> gi​(x) are inequality constraint functions (possibly nonconvex)
> 
> hj(x)h_j(x)
> hj​(x) are equality constraint functions (possibly nonlinear)
> 
> 
> Unlike in convex optimization, we don't have guarantees about global optimality, and the problem may have multiple local minima.
> 2. Interior Point Framework
> Interior point methods work by transforming the constrained problem into a sequence of unconstrained (or more easily constrained) problems. For nonconvex optimization, we'll use a barrier-based approach.
> The key idea is to replace the inequality constraints with a barrier function that prevents the algorithm from leaving the feasible region.
> 2.1. Barrier Function Formulation
> We'll introduce the logarithmic barrier function:
> ϕ(x)=−∑i=1mlog⁡(−gi(x))\phi(x) = -\sum_{i=1}^{m} \log(-g_i(x))ϕ(x)=−i=1∑m​log(−gi​(x))
> This function grows to infinity as we approach the boundary of the feasible region (where any gi(x)g_i(x)
> gi​(x) approaches zero from below).
> 
> We then formulate the barrier problem:
> min⁡x∈Rnf(x)+μϕ(x)subject tohj(x)=0,j=1,…,p\begin{align}
> \min_{x \in \mathbb{R}^n} \quad & f(x) + \mu \phi(x) \\
> \text{subject to} \quad & h_j(x) = 0, \quad j = 1, \ldots, p
> \end{align}x∈Rnmin​subject to​f(x)+μϕ(x)hj​(x)=0,j=1,…,p​​
> Where μ>0\mu > 0
> μ>0 is a barrier parameter that will be gradually reduced to zero.

> Overview https://coin-or.github.io/Ipopt/
> 
> Ipopt (Interior Point Optimizer, pronounced "Eye-Pea-Opt") is an open source software package for large-scale nonlinear optimization. It can be used to solve general nonlinear programming problems of the form
> 
> minx∈Rns.t.f(x)gL≤g(x)≤gUxL≤x≤xU,(NLP)
> 
> where x∈Rn are the optimization variables (possibly with lower and upper bounds, xL∈(R∪{−∞})n and xU∈(R∪{+∞})n) with xL≤xU, f:Rn→R is the objective function, and g:Rn→Rm are the general nonlinear constraints. The functions f(x) and g(x) can be linear or nonlinear and convex or non-convex (but should be twice continuously differentiable). The constraint functions, g(x), have lower and upper bounds, gL∈(R∪{−∞})m and gU∈(R∪{+∞})m with gL≤gU. Note that equality constraints of the form gi(x)=g¯i can be specified by setting gLi=gUi=g¯i.



- [TODO: Extract the basic idea ]

- We are able to exploit hessian sparsity

- explain the basic idea and motivation behind the interior point method (noceda, hartman presentation)
- why it is interesting 

#### fast QAP

- [TODO: Provide a description of the algorithm since in the original paper there is a mistake]
- Tie back to the results of Hartman, Pidnebesna


#### Algorithmic Choices
[SUBTITLE: Using more sophisticated algorithms]

- We use IPOPT interior point solver
  - open source, widely available, has many options (maybe too many)
  - [TODO: Different choices of linear systems solution algorithms]
    - [TODO: Explain how the "ma57" algorithm from IPOPT work]

### Manifold
[SUBTITLE: Exploiting the geometry of the search space.]

- manifold optimization allows us to only consider the space we are working on, potentially decreasing the problem dimension and leading to faster algorithms

#### Manifold optimization Overview

> Drawing on the provided sources, "Step 0 in optimization" lays out the fundamental definition of an optimization problem before considering any additional structure that might inform algorithmic approaches.
> According to Source
> , "Step 0 in optimization" starts with:
> •
> A set $S$
> .
> •
> A function $f: S \rightarrow \mathbb{R}$

> Step 0 in optimization
> It starts with a set 𝑆𝑆 and a function 𝑓𝑓: 𝑆𝑆 → 𝐑𝐑. We want to compute:
> min
> 𝑥𝑥∈𝑆𝑆 𝑓𝑓 𝑥𝑥
> These bare objects fully specify the problem.
> Any additional structure on 𝑆𝑆 and 𝑓𝑓 may (and should) be exploited
> for algorithmic purposes but is not part of the problem.
> Classical unconstrained optimization
> The search space is a linear space, e.g., 𝑆𝑆 = 𝐑𝐑𝑛𝑛:
> min
> 𝑥𝑥∈𝐑𝐑𝑛𝑛 𝑓𝑓 𝑥𝑥
> We can choose to turn 𝐑𝐑𝑛𝑛 into a Euclidean space: 𝑢𝑢, 𝑣𝑣 = 𝑢𝑢⊤𝑣𝑣.
> If 𝑓𝑓 is differentiable, we have a gradient grad𝑓𝑓 and Hessian Hess𝑓𝑓.
> We can build algorithms with them: gradient descent, Newton’s...
> This course: optimization on manifolds
> We target applications where 𝑆𝑆 = ℳ is a smooth manifold:
> min
> 𝑥𝑥∈ℳ 𝑓𝑓 𝑥𝑥
> We can choose to turn ℳ into a Riemannian manifold.
> If 𝑓𝑓 is differentiable, we have a Riemannian gradient and Hessian.
> We can build algorithms with them: gradient descent, Newton’s...

> .
> •
> The objective is to compute the minimum of $f(x)$ over all $x$ in the set $S$, expressed as $\min_{x \in S} f(x)$
> .
> These two components, the set $S$ and the function $f$, fully specify the problem itself
> . Any additional structure on $S$ or $f$, while potentially exploitable for developing algorithms, is not considered part of the problem definition at this initial stage
> .
> Within this foundational framework of minimizing a function $f$ over a set $S$, the sources then introduce different contexts based on the nature of the set $S$:
> 1.
> Classical unconstrained optimization: In this setting, the search space $S$ is a linear space, such as $\mathbb{R}^n$
> . For example, one might seek $\min_{x \in \mathbb{R}^n} f(x)$. Here, $S = \mathbb{R}^n$. In this context, one can introduce additional structure like a Euclidean space inner product on $\mathbb{R}^n$. If $f$ is differentiable, concepts like the gradient (grad$f$) and Hessian (Hess$f$) exist, which allow for the construction of algorithms like gradient descent and Newton's methods
> .
> 2.
> Optimization on manifolds: This is the specific focus of the course mentioned in sources
> . Here, the target application involves situations where the set $S$ is a smooth manifold, denoted as $\mathcal{M}$. The problem is formulated as $\min_{x \in \mathcal{M}} f(x)$. Similar to the classical case, one can endow $\mathcal{M}$ with additional structure, specifically turning it into a Riemannian manifold. If $f: \mathcal{M} \rightarrow \mathbb{R}$ is differentiable (a notion that requires careful definition for manifolds), corresponding Riemannian gradient and Hessian concepts can be defined to build optimization algorithms such as gradient descent and Newton's methods adapted for the manifold setting. The unit sphere $S^{n-1}$ is given as a concrete example of a smooth manifold where such optimization problems arise
> .
> The paper by Douik and Hassibi
> , which discusses manifold optimization, also aligns with this fundamental definition of finding an extreme point of a function over a domain $D \subset \mathbb{R}^n$. Riemannian optimization, as described in this paper, aims to transform constrained Euclidean optimization problems (where the search set $D$ is constrained, i.e., $D \subset \mathbb{R}^n$) into an unconstrained optimization problem over a restricted search space, which is a Riemannian manifold. This reformulates the problem back into the core "Step 0" structure, but with the set $S$ being the manifold itself, allowing algorithms to exploit its geometric structure. This is particularly advantageous because the manifold may be of lower dimension than the ambient Euclidean space
> .
> In essence, "Step 0 in optimization" is the universally applicable initial statement of the optimization problem: minimize a function over a set. The subsequent steps in algorithm development, whether for classical Euclidean space or for manifolds, involve leveraging the specific mathematical structure of that set $S$ (e.g., linear space, smooth manifold) and the function $f$ (e.g., differentiability) to define notions like gradients and Hessians, which are necessary to construct practical algorithms
.

> Manifold Optimization Overview
> Manifold optimization extends classical optimization techniques to problems where solutions must lie on a curved geometric space (a manifold) rather than in a flat Euclidean space. This is particularly useful in many applications where constraints naturally form manifolds, such as orthogonality constraints, fixed-rank matrices, or positive definite matrices.
> Key Differences from Classical Optimization
> The fundamental difference between classical optimization and manifold optimization is how we conceptualize the search space and movement through it. In manifold optimization:
> 
> Geometry matters: The manifold's curvature affects how we measure distances, define directions, and move along the surface.
> Tangent spaces: At each point on a manifold, we work with a local linearization called the tangent space, which approximates the manifold at that point.
> Retractions: After moving in the tangent space, we need to "project" back onto the manifold using operations called retractions.
> Riemannian metrics: These define how to measure distances and angles on the manifold, replacing the standard Euclidean inner product.
> 
> How Manopt Framework Adapts Classical Algorithms
> Manopt (and similar frameworks) adapts classical optimization algorithms by redefining their core operations to work on manifolds:
> Gradient Descent
> Classical: Move in the direction of the negative gradient.
> On Manifolds:
> 
> Compute the Riemannian gradient (projection of Euclidean gradient onto the tangent space)
> Move along this direction in the tangent space
> Use a retraction to map back to the manifold
> 
> Trust-Region Methods
> Classical: Build a quadratic model in a "trusted" region and find the minimizer.
> On Manifolds:
> 
> Define the trust region on the tangent space
> Solve the trust-region subproblem in the tangent space
> Retract the solution back to the manifold
> Update the trust region size based on model fidelity
> 
> Conjugate Gradient Methods
> Classical: Build conjugate directions in Euclidean space.
> On Manifolds:
> 
> Transport previous search directions to the current tangent space (using vector transport)
> Ensure conjugacy with respect to the Riemannian metric
> Combine with retraction steps
> 
> Specific Challenges in Manifold Optimization
> 
> Vector Transport: Moving vectors between different tangent spaces requires special operations.
> Second-Order Information: The Riemannian Hessian incorporates both the Euclidean Hessian and the manifold's curvature.
> Line Search: Requires special care since straight lines don't generally stay on the manifold.
> Interior Point Methods: These become more complex on manifolds, as the concept of "interior" depends on the manifold's boundary structure.

> Manifold Optimization Over the Set of Doubly Stochastic Matrices: A Second-Order Geometry
> This paper by Ahmed Douik and Babak Hassibi introduces a novel geometric framework for solving optimization problems where the variable of interest is a doubly stochastic matrix. The core innovation is reformulating these constrained optimization problems as unconstrained problems over specialized manifolds.
> Key Contributions
> The authors introduce three new Riemannian manifolds:
> 
> Doubly stochastic multinomial manifold (DPn): For doubly stochastic matrices (positive entries with rows and columns summing to 1)
> Symmetric multinomial manifold (SPn): For symmetric doubly stochastic matrices
> Definite multinomial manifold (SP+n): For positive definite symmetric doubly stochastic matrices
> 
> These manifolds generalize the simplex (multinomial manifold) to more complex constraint structures while preserving mathematical elegance.
> Why This Matters
> Traditional approaches to these optimization problems typically use interior-point methods, which become prohibitively slow in high dimensions due to the "curse of dimensionality." The Riemannian approach exploits the lower-dimensional structure of the constraint set, leading to more efficient algorithms.
> Technical Framework
> For each manifold, the paper develops a complete second-order geometric structure by deriving:
> 
> The tangent space definition
> Orthogonal projections onto the tangent space
> Riemannian gradient (relating Euclidean and Riemannian gradients)
> Riemannian Hessian operators
> Efficient retraction mappings (to move from tangent space back to the manifold)
> 
> Each manifold is endowed with the Fisher information metric, which gives them a differentiable structure invariant to coordinate system choices.
> Algorithms and Performance
> The authors formulate both first-order (line search/gradient descent) and second-order (Newton's method) optimization algorithms for these manifolds and analyze their computational complexity. Through extensive simulations, they demonstrate that their approach significantly outperforms:
> 
> Generic convex solvers (like CVX)
> Specialized algorithms for these types of problems
> 
> The performance gap widens dramatically as the problem dimension increases, with speed improvements of up to two orders of magnitude in high dimensions.
> Applications
> The framework is particularly valuable for clustering applications, where recovering the structure of a graph from a similarity matrix often involves optimizing over doubly stochastic matrices. The paper demonstrates the efficacy of their approach on both convex and non-convex clustering problems.

> Retraction in Manifold Optimization
> A retraction is a fundamental concept in optimization on manifolds that provides a way to "move" from a point on a manifold along a tangent direction while staying on the manifold.
> The Basic Idea
> When working with constrained optimization problems on manifolds like the doubly-stochastic matrices, we face a challenge: if we move in a tangent direction, we typically leave the manifold (violating our constraints). A retraction solves this problem by mapping points from the tangent space back onto the manifold.
> Mathematically, a retraction at a point PP
> P on manifold M\mathcal{M}
> M is a smooth mapping RP:TPM→MR_P: T_P\mathcal{M} \rightarrow \mathcal{M}
> RP​:TP​M→M from the tangent space to the manifold that satisfies two key properties:
> 
> 
> RP(0)=PR_P(0) = P
> RP​(0)=P (the zero tangent vector maps to the original point)
> 
> dRP(0)=idTPMdR_P(0) = \text{id}_{T_P\mathcal{M}}
> dRP​(0)=idTP​M​ (the differential of RPR_P
> RP​ at 0 is the identity mapping)
> 
> 
> The second property ensures that the retraction locally preserves the direction of the tangent vector.
> Why Retractions Matter
> Retractions are crucial because:
> 
> They allow us to convert unconstrained optimization in the tangent space to constrained optimization on the manifold
> They're often computationally more efficient than using the exact exponential map (which would follow geodesics)
> They maintain the constraints that define the manifold

> 1. Projection-Based Retraction
> Given a point P∈ΩnP \in \Omega_n
> P∈Ωn​ and a tangent direction H∈TPΩnH \in T_P\Omega_n
> H∈TP​Ωn​, a simple retraction is:
> 
> RP(H)=ProjΩn(P+H)R_P(H) = \text{Proj}_{\Omega_n}(P + H)
> RP​(H)=ProjΩn​​(P+H)
> where ProjΩn\text{Proj}_{\Omega_n}
> ProjΩn​​ denotes the projection onto the set of doubly-stochastic matrices. This projection can be computed using the Sinkhorn-Knopp algorithm, which iteratively normalizes rows and columns until convergence.

> The Exponentiation Problem on Manifolds
> Overview
> The exponentiation problem refers to finding and computing the exponential map on a manifold. In the context of the doubly-stochastic matrix manifold, this is a fundamental challenge that bridges the gap between the tangent space and the manifold itself.
> Definition and Purpose
> The exponential map at a point PP
> P on a manifold M\mathcal{M}
> M, denoted as ExpP\text{Exp}_P
> ExpP​, maps tangent vectors to points on the manifold by following geodesics (shortest paths):
> 
> ExpP:TPM→M\text{Exp}_P: T_P\mathcal{M} \rightarrow \mathcal{M}ExpP​:TP​M→M
> For a tangent vector HH
> H, ExpP(H)\text{Exp}_P(H)
> ExpP​(H) represents the point reached by traveling along the geodesic starting at PP
> P in the direction of HH
> H for a distance of ∥H∥\|H\|
> ∥H∥.
> 
> Why It's Important
> The exponential map is crucial for:
> 
> Implementing optimization algorithms on manifolds
> Moving along geodesic curves (optimal paths)
> Defining Riemannian distances and metrics
> Translating between the linear tangent space and the curved manifold
> 
> The Challenge for Doubly-Stochastic Matrices
> For the manifold of doubly-stochastic matrices, the exponentiation problem is particularly challenging because:
> 
> The manifold has a complex structure due to both equality constraints (row/column sums) and inequality constraints (non-negativity)
> Computing true geodesics requires solving differential equations with these constraints
> Even small steps in certain tangent directions can lead to constraint violations
> 
> Approaches to Exponentiation
> 1. Exact Exponential Map
> For some manifolds, closed-form solutions exist for the exponential map. However, for doubly-stochastic matrices, no simple closed-form expression is known due to the interaction between constraints.

> Practical Solutions
> In practice, the exponentiation problem is often addressed by using retractions instead of the true exponential map:
> 
> First-order Retraction: Use a simpler mapping that approximates the exponential map to first order:
> 
> Move in the tangent direction: P′=P+HP' = P + H
> P′=P+H
> Project back to the manifold: RP(H)=ProjΩn(P′)R_P(H) = \text{Proj}_{\Omega_n}(P')
> RP​(H)=ProjΩn​​(P′)

> Illustrative Example
> Consider a doubly-stochastic matrix PP
> P and a tangent direction HH
> H. To approximate ExpP(H)\text{Exp}_P(H)
> ExpP​(H):
> 
> 
> Ensure HH
> H has row and column sums of zero (it's in the tangent space)
> 
> Form P′=P+HP' = P + H
> P′=P+H (likely not doubly-stochastic)
> 
> Apply a threshold to prevent negative values: Pij′′=max⁡(Pij′,ϵ)P''_{ij} = \max(P'_{ij}, \epsilon)
> Pij′′​=max(Pij′​,ϵ)
> Use Sinkhorn algorithm to rescale P′′P''
> P′′ until it's doubly-stochastic
> 
> 
> This gives an approximation to the exponential map that's computationally feasible.

> Connection to Optimization
> In Riemannian optimization algorithms for doubly-stochastic matrices, the exponentiation problem appears when updating the current iterate:
> Pk+1=RPk(−α⋅gradf(Pk))P_{k+1} = R_{P_k}(-\alpha \cdot \text{grad}f(P_k))Pk+1​=RPk​​(−α⋅gradf(Pk​))
> where RR
> R is a retraction that approximates the exponential map, and α\alpha
> α is the step size.

> Historical Motivation
> The name "exponential" comes from its origin in Lie group theory, where for matrix groups, it literally is the matrix exponential:
> exp⁡(X)=∑k=0∞Xkk!=I+X+X22!+X33!+⋯\exp(X) = \sum_{k=0}^{\infty} \frac{X^k}{k!} = I + X + \frac{X^2}{2!} + \frac{X^3}{3!} + \cdotsexp(X)=k=0∑∞​k!Xk​=I+X+2!X2​+3!X3​+⋯

> From Lie Groups to General Manifolds
> This concept was generalized from Lie groups to arbitrary Riemannian manifolds, retaining the name "exponential" due to its similar properties:
> 
> It maps a vector space (tangent space) to a curved space (manifold)
> At small scales, it behaves similarly to the exponential series
> The paths it generates satisfy differential equations similar to those with exponential solutions

> Geodesics: The Shortest Paths on Manifolds
> Fundamental Definition
> A geodesic is the generalization of a "straight line" to curved spaces. On a Riemannian manifold, a geodesic is locally the shortest path between points as measured by the Riemannian metric.


> ## 1. Riemannian Gradient Computation
> 
> **Yes, the Riemannian gradient can be computed explicitly and is uniquely determined.** Here's how it works:
> 
> - Given a function f: M → ℝ on a Riemannian manifold M, the Riemannian gradient grad_M f(x) is the unique tangent vector at x such that:
>   ```
>   ⟨grad_M f(x), v⟩_x = Df(x)[v]
>   ```
>   for all tangent vectors v ∈ T_x M, where ⟨·,·⟩_x is the Riemannian metric and Df(x)[v] is the directional derivative.
> 
> - **In practice**: You compute the Euclidean gradient ∇f(x) in the ambient space, then project it onto the tangent space:
>   ```
>   grad_M f(x) = P_{T_x M}(∇f(x))
>   ```
> 
> - **Uniqueness**: The Riemannian gradient is uniquely determined by the choice of Riemannian metric on the manifold. Different metrics give different Riemannian gradients.
> 
> ## 2. Moving Along Geodesics
> 
> This is where the **exponential map** becomes crucial:
> 
> - After computing the Riemannian gradient grad_M f(x), we want to move in the direction -grad_M f(x) (for minimization)
> - The exponential map Exp_x: T_x M → M takes a tangent vector and follows the geodesic for that "distance"
> - **Gradient descent step**: 
>   ```
>   x_{k+1} = Exp_{x_k}(-α · grad_M f(x_k))
>   ```
>   where α is the step size
> 
> The key insight is that the Riemannian gradient points in the direction of steepest ascent *on the manifold*, and the exponential map ensures we stay on the manifold while moving in that direction.
> 
> ## 3. Exponential and Logarithm Maps
> 
> These are **geometric mappings**, not the usual exp/log functions:
> 
> - **Exponential map** Exp_x(v): Takes a tangent vector v ∈ T_x M and returns the point on M reached by following the geodesic starting at x in direction v for "time" ||v||
> - **Logarithm map** Log_x(y): The inverse operation - given two points x, y on M, returns the tangent vector v such that Exp_x(v) = y
> 
> Think of it this way: if you're standing at point x on a curved surface and want to walk to point y, the logarithm map tells you which direction and how far to walk (in the tangent plane), while the exponential map actually takes you there along the curved surface.
> 
> ## 4. Manifolds Without Known Geodesics
> 
> This is where the theory meets practical reality:
> 
> **When geodesics are unknown or expensive to compute**, we use **retractions** instead of exponential maps:
> 
> - A retraction R_x: T_x M → M is any smooth mapping that satisfies:
>   - R_x(0) = x
>   - DR_x(0) = identity (first-order approximation to exponential map)
> 
> **For the doubly stochastic manifold example:**
> - The Sinkhorn projection is indeed a retraction, not the true exponential map
> - It projects points back onto the manifold, but not necessarily via the shortest geodesic path
> - **This is still valid manifold optimization!** Many practical algorithms use retractions
> 
> **Why retractions work:**
> - They preserve the first-order optimality conditions
> - Convergence guarantees still hold (though rates may differ)
> - Much more computationally tractable
> 
> **To move along true geodesics, you would need:**
> - The exponential map (which requires solving geodesic ODEs)
> - Or at least a good approximation to it
> 
> The beauty of Riemannian optimization is that it provides a principled framework that works both with exact geodesics (when available) and practical approximations (retractions) when they're not. The Sinkhorn projection for doubly stochastic matrices is a perfect example of how this theory translates into practical, efficient algorithms.

> Excellent follow-up questions! These get to some deep and subtle aspects of Riemannian geometry. Let me address each one:

> ## 1. Multiple Riemannian Metrics
> 
> **Yes, there can be infinitely many Riemannian metrics on the same manifold!** This is a crucial point:
> 
> **Example - The sphere S²:**
> - **Standard metric**: Inherited from embedding in ℝ³ (the usual round metric)
> - **Scaled metric**: g₂ = c·g₁ where c > 0 is a constant
> - **Warped metrics**: Different curvature in different regions
> 
> **Impact on optimization:**
> - Different metrics give different Riemannian gradients
> - Different convergence rates and paths to the solution
> - **The final solution often differs!** (except for convex problems)
> 
> **How to choose a metric:**
> - **Natural/canonical choice**: Use the metric inherited from the ambient space (most common)
> - **Problem-specific**: Choose based on the geometry you want to respect
> - **Computational**: Choose for numerical efficiency
> 
> **For projection from ambient space:**
> This corresponds to the **canonical metric** - the one inherited from the ambient Euclidean space. When you project ∇f(x) onto the tangent space, you're implicitly using the ambient inner product.
> 
> ## 2. Distance Along Tangent Vectors
> 
> When I say "follow a tangent vector for that distance," the **distance is ||v||** - the norm of the tangent vector:
> 
> - Given tangent vector v ∈ T_x M
> - ||v|| is computed using the Riemannian metric: ||v|| = √⟨v,v⟩_x
> - Exp_x(v) follows the geodesic starting at x in direction v/||v|| for arc-length ||v||
> 
> **Think of it like this:** v encodes both direction (v/||v||) and how far to travel (||v||).
> 
> ## 3. Etymology of "Exponential" and "Logarithm"
> 
> The names come from **Lie group theory**:
> 
> - For matrix Lie groups (like SO(n), the rotation matrices), the exponential map is literally the matrix exponential: Exp(A) = e^A = I + A + A²/2! + ...
> - The logarithm map is the matrix logarithm (inverse of matrix exponential)
> - This naming was then generalized to all Riemannian manifolds, even when there's no literal exponential function involved
> 
> **Historical note:** Lie groups are both manifolds and groups, so the exponential map connects the group structure with the geometric structure.
> 
> ## 4. The "Time" Parameter
> 
> This comes from thinking of geodesics as **particle trajectories**:
> 
> - A geodesic γ(t) is a curve parameterized by time t
> - γ(0) = x (starting point)
> - γ'(0) = v (initial velocity)
> - γ(1) = Exp_x(v) (where we end up after "time" 1)
> 
> So ||v|| represents both the speed and the total time traveled. If we travel with velocity v for time 1, we cover distance ||v||.
> 
> ## 5. Higher-Order Retractions
> 
> **Absolutely!** Retractions can be designed with higher-order accuracy:
> 
> **Order of a retraction R_x:**
> - **1st order**: DR_x(0) = I (matches exponential map to first order)
> - **2nd order**: D²R_x(0,0) = D²Exp_x(0,0) (matches second derivatives)
> - And so on...
> 
> **Examples:**
> - **Cayley transform**: Often 2nd order accurate
> - **Padé approximants**: Can achieve arbitrary order
> - **Truncated geodesic series**: Computational approximation to any order
> 
> **Trade-off:** Higher order usually means more computational cost per step.
> 
> ## 6. Retractions vs. True Geodesics in Practice
> 
> **Retractions provide guarantees, just different ones:**
> 
> **What retractions guarantee:**
> - ✅ Stay on the manifold
> - ✅ First-order optimality conditions preserved
> - ✅ Convergence to critical points (under standard assumptions)
> - ✅ Local convergence rates often preserved
> 
> **What they don't guarantee:**
> - ❌ Shortest path between points
> - ❌ Exact geodesic motion
> - ❌ Sometimes: global convergence rates may be slower
> 
> **Practical implications:**
> - **"Move far enough"**: Step size control still works - you can take larger/smaller steps
> - **"Where we move"**: You reach a point on the manifold, just not via the geodesic
> - **Performance**: Often nearly as good as true geodesics, much cheaper to compute
> 
> **Bottom line:** For most optimization problems, a good retraction is practically indistinguishable from the exponential map, especially with adaptive step sizes. The Sinkhorn projection for doubly stochastic matrices is a perfect example - it works beautifully in practice despite not being the true exponential map.
> 
> The theory is elegant with true geodesics, but the practice is often more efficient with well-chosen retractions!

> Excellent question! You're absolutely right that for basic gradient descent, you don't need vector transport - just compute the gradient at each point. But vector transport becomes crucial for more sophisticated optimization algorithms. Let me show you why.
> Where Basic Gradient Descent Doesn't Need Vector Transport
> You're correct for simple gradient descent:
> # No vector transport needed!
> x_k = current_point
> grad_k = riemannian_gradient(x_k)  # Compute fresh at current point
> x_{k+1} = retract(x_k, -alpha * grad_k)
> 
> This works fine and is the most common approach for simple problems.
> Vector can transport improve solution time.

#### Specifics For Doubly Stochastic Manifold

- we use simple projection because we use simple gradient descent algorithms
  - for more complex algorithms such as conjugate gradient vector transport or parallel transport would be needed for DS Manifold 
    - [TODO: CHeck what is available from the paper on DS manifolds]

- mention the Manopt (PyManopt) library is used for optimization

- [TODO: Provide a definition of the tangent space, description]
- [TODO: Provide a definition of how to project to the tangent space]
- [TODO: Question??? WHat is FIsher metric?]


##### Sinkhorn-Knopp algorithm

- [TODO: include formal definition for the SInkhorn-knopp algorithm]
- [TODO: Include landmark paper and description]
  - include discussion for the number of iterations of the sinkhorn algorithm
  - inlcude discussion that the number of iterations decreases as we are closer to the original the space of DS matrices
    - this is the case in iterative gradient based algorithms where we move move a small distance along a set direction (does not move us far)

- [TODO: Show that this is a good retraction for manifold opt; provide sources]


> No, the sources do **not** state that the Birkhoff polytope is embedded in the special orthogonal group. Instead, they present the Birkhoff polytope and the orthogonal group as **two distinct continuous spaces** that have been proposed for relaxing permutation matrices.
> 
> Here's a breakdown of what the sources indicate about each:
> 
> *   **Birkhoff Polytope ($B_n$)**: This is defined as the set of **doubly stochastic matrices**. A doubly stochastic matrix is a square matrix ($X \in \mathbb{R}^{n \times n}$) where all entries are positive ($X_{ij} > 0$), and each row and each column sums to 1. The dimension of the Birkhoff polytope is $(n-1)^2$.
> *   **Special Orthogonal Group ($SO(n)$)**: This consists of all $n \times n$ **orthogonal matrices** ($O$) that satisfy $O^T O = OO^T = I$ and have a **determinant of +1**. The special orthogonal group is a differentiable manifold. Its dimension is $n(n-1)/2$.
> 
> A key distinction highlighted in the sources is their **representation dimension**:
> *   The Birkhoff polytope has a dimension of $(n-1)^2$.
> *   The orthogonal group has a lower dimension of $n(n-1)/2$.
> 
> While **permutation matrices** (which are square binary matrices with exactly one 1 in each row and column) are a subset of both doubly stochastic matrices (and thus lie within the Birkhoff polytope) and orthogonal matrices (and thus lie within the special orthogonal group), the Birkhoff polytope and the special orthogonal group are generally distinct sets with different defining properties. One is not embedded within the other.

- Birkhoff polytope and pecial orthogonal groups are different spaces

> No.  In fact, the Birkhoff–von Neumann theorem tells us that every $n\times n$ doubly stochastic matrix is a convex combination of permutation matrices—and apart from those permutation matrices, none of the interior (or “mixed”) doubly stochastic matrices is even orthogonal, let alone special‐orthogonal.
> 
> * **Orthogonality** requires $P^T P = I$.  A typical doubly stochastic matrix $D$ (with all entries strictly between 0 and 1) does *not* satisfy $D^T D = I$.
> * **Special orthogonal** further demands $\det = +1$.  Among the permutation matrices, exactly those corresponding to even permutations lie in $\mathrm{SO}(n)$; the others have determinant $-1$.
> * Thus
> 
>   $$
>     \{\text{doubly stochastic matrices}\}\;\cap\;\mathrm{SO}(n)
>     \;=\;\{\text{even permutation matrices}\},
>   $$
> 
>   a finite (zero-dimensional) subset, not the whole Birkhoff polytope.
> 
> In particular, the convex hull of those even permutations (the Birkhoff polytope) is *not* contained in $\mathrm{SO}(n)$—only its vertices that happen to be even permutations lie there.
> As an example consider barycenter, it is a DS matrix but singular


### OP4T
[SUBTITLE: Moving into a better behaved space]

#### Manifold background

> ## Lie Groups: Definition and Properties
> 
> **Definition:** A **Lie group** G is a set that is simultaneously:
> 1. A **smooth manifold** (has coordinates, calculus works)
> 2. A **group** (has multiplication, identity, inverses)
> 3. The group operations are **smooth** (differentiable)
> 
> **Key properties:**
> - **Tangent space at identity**: Called the **Lie algebra** 𝔤
> - **Exponential map**: exp: 𝔤 → G (connects algebra to group)
> - **Logarithm map**: log: G → 𝔤 (inverse of exponential, locally)
> 
> **Examples:**
> - ℝⁿ under addition
> - S¹ = {e^{iθ} : θ ∈ ℝ} under complex multiplication  
> - GL(n) = invertible n×n matrices under matrix multiplication
> - **SO(n)** = special orthogonal group (our main focus)
> 
> ## Special Orthogonal Group SO(n)
> 
> **Definition:**
> ```
> SO(n) = {Q ∈ ℝⁿˣⁿ : QᵀQ = I, det(Q) = +1}
> ```
> 
> **What this means:**
> - **Orthogonal**: QᵀQ = I (columns are orthonormal)
> - **Special**: det(Q) = +1 (orientation-preserving)
> - **Geometric interpretation**: Rotations in n-dimensional space
> 
> **Properties:**
> - **Compact manifold** of dimension n(n-1)/2
> - **Connected** (for n ≥ 2): can rotate from any orientation to any other
> - **Matrix group**: inherits matrix operations
> 
> **Examples:**
> - **SO(2)**: 2D rotations, isomorphic to S¹
>   ```
>   [cos θ  -sin θ]
>   [sin θ   cos θ]
>   ```
> - **SO(3)**: 3D rotations (orientation of rigid bodies)
> 
> ## The Lie Algebra so(n)
> 
> **Definition:** The tangent space at the identity:
> ```
> so(n) = {A ∈ ℝⁿˣⁿ : Aᵀ = -A}
> ```
> 
> **Properties:**
> - **Skew-symmetric matrices**: Aᵀ = -A
> - **Dimension**: n(n-1)/2 (same as SO(n))
> - **Lie bracket**: [A,B] = AB - BA
> 
> **Physical interpretation:** 
> - Elements represent **angular velocities**
> - Generators of infinitesimal rotations
> 
> ## The Exponential Map Connection
> 
> Here's where the naming becomes clear! For matrix Lie groups:
> 
> **Matrix exponential:**
> ```
> exp(A) = I + A + A²/2! + A³/3! + ... = Σ Aᵏ/k!
> ```
> 
> **Key theorem:** If A ∈ so(n), then exp(A) ∈ SO(n)
> 
> **Examples:**
> - **SO(2)**: exp([0 -θ; θ 0]) = [cos θ -sin θ; sin θ cos θ]
> - **SO(3)**: Rodrigues' formula for rotation from axis-angle
> 
> **This is literally the exponential map from Riemannian geometry!**
> 
> ## Applications in Manifold Optimization
> 
> **1. Orthogonality-Constrained Problems:**
> ```
> minimize f(X)  subject to X ∈ SO(n)
> ```
> 
> **Examples:**
> - **Procrustes problem**: Find rotation minimizing ||AQ - B||²
> - **Principal component analysis** with orthogonal factors
> - **Matrix factorizations** with orthogonal constraints
> 
> **2. Concrete Algorithm Steps:**
> 
> Given objective f: SO(n) → ℝ:
> 
> **Step 1 - Riemannian gradient:**
> ```
> grad_SO f(Q) = P_tangent(∇f(Q))
> ```
> where P_tangent projects onto so(n)
> 
> **Step 2 - Exponential map update:**
> ```
> Q_{k+1} = Q_k · exp(-α · Q_k^T ∇f(Q_k))
> ```
> 
> **Why this works:**
> - ∇f(Q) lives in ambient space ℝⁿˣⁿ
> - Q_k^T ∇f(Q_k) projects to so(n) (skew-symmetric)
> - exp(-α · Q_k^T ∇f(Q_k)) ∈ SO(n) automatically
> - Matrix multiplication keeps us in SO(n)
> 
> **3. Practical Benefits:**
> 
> **Exact geodesics:** Unlike many manifolds, we can compute exp(A) exactly for skew-symmetric A using:
> - **Rodrigues' formula** (SO(3))
> - **Trigonometric formulas** (general case)
> - **Efficient algorithms** (scaling and squaring)
> 
> **No projection needed:** The exponential map naturally stays on the manifold, unlike retractions that might need correction.
> 
> ## Example: PCA with Orthogonality
> 
> **Problem:** Minimize ||X - QR||² where Q ∈ SO(n)
> 
> **Traditional approach:** SVD or iterative projection methods
> 
> **Manifold approach:**
> 1. Compute Euclidean gradient: ∇f(Q) = 2(QR - X)R^T
> 2. Project to so(n): A = Q^T∇f(Q) - (Q^T∇f(Q))^T
> 3. Update: Q ← Q·exp(-αA)
> 
> **Advantages:**
> - **Guaranteed orthogonality** at every step
> - **No numerical drift** from constraint violation
> - **Natural geometry** respects the manifold structure
> 
> This is why SO(n) is such a perfect showcase for manifold optimization - it's both geometrically rich and computationally tractable!

### SoftSort

[SUBTITLE: Reformulation into and modelling in the space of vectors, permutations are vectors as well]

- drastically reduce the dimension of the problem
- is this formulation strong enough to yield satisfying results


### Additional considerations

#### Automatic differentiation

[TODO: Explore analytical differentiation vs. automatic differention concretely with approach used by pytorch
  - are there any major performance downsides?
  - what are the memory downsides?
  - how did the field of automatic differentiation change in the past years?
  - what are the disadvantages when cosidering numerical accuracy?
  - do we lose anything in theoretical side of things? When do we want analytical derivatives instead of automatic? Are these differences substantial or only technical?
  - STRETCH: What is symbolic differentiation?
]
> Automatic Differentiation vs. Analytical Derivatives
> Comparing automatic differentiation (AD) with analytical derivatives requires examining both precision and computational efficiency. Based on the foundations of nonlinear optimization as presented in the text, I can offer the following analysis:
> Precision Comparison
> Automatic differentiation is mathematically exact, just like analytical derivatives. Unlike numerical differentiation (such as finite differences), which introduces truncation errors, AD computes the exact same mathematical derivatives that you would calculate by hand, just through a different process.
> AD works by decomposing functions into elementary operations (addition, multiplication, basic functions like exp, log, etc.) and applying the chain rule systematically. Since each elementary operation has a known derivative, the composite derivative is built up precisely.
> For example, when calculating gradients for the Lagrangian function L(x, λ, ν) described in Chapter 5 of your text, both methods would produce identical results in exact arithmetic.
> Speed Comparison
> The efficiency comparison between AD and analytical derivatives is more nuanced:
> When analytical derivatives may be faster:
> 
> For simple functions with straightforward derivatives
> When the analytical form leads to significant algebraic simplifications
> In cases where a specialized implementation can take advantage of problem structure
> 
> When automatic differentiation may be faster:
> 
> For complex functions where manual derivation is cumbersome
> When computing higher-order derivatives (like Hessians needed for Newton methods)
> For large-scale problems with many variables
> When the function definition changes frequently during development
> 
> The reverse mode of AD (backpropagation) is particularly efficient for optimization problems because it computes gradients in time roughly proportional to the function evaluation time, regardless of the number of variables. This property makes it indispensable for problems with many variables, like those discussed in Chapter 6 of your text (ℓ₁ methods for cardinality problems).
> Practical Considerations
> Several factors influence the choice between AD and analytical derivatives:
> 
> Development time: AD dramatically reduces implementation time and eliminates potential human errors in deriving complex expressions.
> Maintainability: When your objective function changes, AD automatically adjusts the derivatives, while analytical derivatives require manual updates.
> Memory requirements: Reverse mode AD typically requires storing the computational graph, which can have significant memory costs for very large problems.
> Sparsity exploitation: Advanced AD systems can exploit sparsity patterns in the derivatives, which is crucial for the specialized matrix problems discussed in Chapters 6 and 7 of your text.
> 
> Application to Nonlinear Optimization
> For the optimization problems described in your textbook, such as:
> 
> Quadratic programming (Chapter 2.1)
> Semidefinite programming (Chapter 2.3)
> KKT conditions (Chapter 4)
> Lagrangian duality (Chapter 5)
> 
> Both approaches can work effectively, but the choice depends on problem specifics. For structured problems like quadratic programming where derivatives have simple analytical forms, hand-derived expressions might be optimal. For complex objective functions or when implementing methods that require higher-order derivatives (like interior point methods mentioned in Chapter 2), AD provides significant advantages.
> Conclusion
> Automatic differentiation is precisely as accurate as analytical derivatives in theory. In practice, it often delivers superior results by eliminating human error. While analytical derivatives may sometimes be more computationally efficient for simple cases with known structures, modern AD implementations often match or exceed their performance, especially for complex problems. The decision ultimately depends on your specific application, but the trend in optimization research is increasingly toward automatic differentiation, particularly for large-scale problems.
> Would you like me to elaborate on how automatic differentiation might apply to any specific optimization methods or problems from your textbook?


# Experimental Results

- [TODO: Add chapter introduction]

## Data Instances Descriptions

- Admit data instances are adapted from the instances used by Pidnebesna, Hartmann

### ER

### BA

### LRM ER Rewired

## Experimental Results

### 

# Conclusion

## Future Work

# Appendix

## Code documentation