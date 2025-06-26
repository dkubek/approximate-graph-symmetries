### Interior Point Method


<!-- [INSERT: Personal Artistic Intro to Interior Point Optimization] -->

- Next in our search from a better algorithm comes in form of the natural solution of the to one of the most successful solvers of non-convex nonlinear programming methods. Om this chapter we will briefly introduce the idea of interior point barrieer methods. We will explain at which parts we might gain an advantage and which parts of the algotithm will prove disadvatageous

- one of the main driving force that we are leaning on here is the relative maturity of the method wiith well developed ecosystem and availability of tools and implementations of algorithms.

 - what follows is a brief excursion and introduction to the interiror point barrier method almost entirely inspired by the exposition in the Nocedal book with slight change in notation.

- Read Nocedal if you are hungry for more

- Interior-point (or barrier) methods have proved to be as successful for nonlinear optimiza-
tion as for linear programming, and together with active-set SQP methods, they are currently
considered the most powerful algorithms for large-scale nonlinear programming [Nocedal]

 - The terms “interior-point methods” and “barrier methods” are now used interchangeably [Nocedal]

- The term “interior point” derives from the fact that early barrier methods [98] did
not use slacks and assumed that the initial point x 0 is feasible with respect to the inequality constraints ci (x) ≥ 0, i ∈ I [Nocedal]

- Most modern interior-point methods are infeasible (they can start from any initial point x 0 ) [Nocedal]

- instead of providing the full description of the algorithm refer to [Nocedal]



<!-- [ ] Problem Setup
  - [ ] Definition
  - include basic definitions and assumptions about the problem -->

> # Interior Point Methods for Constrained Optimization
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


<!-- - [ ] Algorithm Idea
  - describe the main idea of the algorithm informally in a paragraph, at this point in such a way as not to require any prerequisite knowledge and definitions
  - Interior Point Framework -->

> 
> Interior point methods are powerful algorithms for solving constrained optimization problems. Unlike active set methods that move along the boundary of the feasible region, interior point methods travel through the interior of the feasible region toward the optimal solution.
> 

> 2. Interior Point Framework
> Interior point methods work by transforming the constrained problem into a sequence of unconstrained (or more easily constrained) problems. For nonconvex optimization, we'll use a barrier-based approach.
> The key idea is to replace the inequality constraints with a barrier function that prevents the algorithm from leaving the feasible region.


<!-- - [ ] Barrier approach
  - Barrier function optimization
  - [ ] Formal mathematical description -->

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


<!-- - [ ] Central Path
  - [ ] Definition (formal)
      - what is the central path formally
  - Informal description -->

> ## The Central Path
> 
> The central path is a key concept in interior point methods. It's the set of minimizers of the barrier problem for different values of μ. The algorithm follows this central path toward the optimal solution as μ → 0.
> 
> In practice, we use a sequence of decreasing values for μ and solve (approximately) the barrier problem at each μ value, using the previous solution as a starting point for the next iteration.
> 


<!-- - [ ] KKT System
  - [ ] Theorem (KKT)
  - [ ] Derivation of the KKT System -->

[TODO: Add KKT Theorem from Nocedal]

> # Derivation of the KKT System in Interior Point Methods
> 
> Let me walk you through how we derive the KKT system used in interior point methods, starting from the original optimization problem and working toward the final system of equations that algorithms like IPOPT solve at each iteration.
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


<!-- - [ ] Primal-Dual Interior Point Method
  - [ ] Basic idea (informal)
  - [ ] Formal definition
  - [ ] Sidenote about what would it mean to  -->

> ## Primal-Dual Interior Point Methods
> 
> Modern interior point methods typically use a primal-dual approach. Instead of directly solving the barrier problem, they solve the KKT (Karush-Kuhn-Tucker) conditions for the barrier problem.

 - [TODO: What advantages do we gain by solving the Primal-Dual problem instead of just the primal problem?] 
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


<!-- - [ ] Solving the KKT system
  - [ ] Newton's Method for the KKT System
  - [ ] Note about hidden dragons and problems with this system and how much of a problem and pain it is to solve it quickly or approximately -->

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

<!-- Conclusion -->

> ## Conclusion
> 
> Interior point methods, particularly as implemented in solvers like IPOPT, are powerful tools for solving constrained nonconvex quadratic problems. They work by following the central path defined by a sequence of barrier problems with decreasing barrier parameters.
> 
> For nonconvex problems, these methods incorporate special techniques like inertia correction, line search, and regularization to handle the challenges of indefinite Hessians. While they can't guarantee global optimality for nonconvex problems, they often find high-quality local solutions efficiently.

> This approach provides a theoretically sound and computationally effective way to solve constrained optimization problems, including nonconvex quadratic programs. The system balances progress toward optimality and constraint satisfaction while navigating through the interior of the feasible region.
