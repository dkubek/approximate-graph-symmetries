### Interior Point Method

<!-- - We use IPOPT interior point solver -->
> 
> Ipopt (Interior Point Optimizer, pronounced "Eye-Pea-Opt") is an open source software package for large-scale nonlinear optimization. It can be used to solve general nonlinear programming problems of the form
> 
> minx∈Rns.t.f(x)gL≤g(x)≤gUxL≤x≤xU,(NLP)
> 
> where x∈Rn are the optimization variables (possibly with lower and upper bounds, xL∈(R∪{−∞})n and xU∈(R∪{+∞})n) with xL≤xU, f:Rn→R is the objective function, and g:Rn→Rm are the general nonlinear constraints. The functions f(x) and g(x) can be linear or nonlinear and convex or non-convex (but should be twice continuously differentiable). The constraint functions, g(x), have lower and upper bounds, gL∈(R∪{−∞})m and gU∈(R∪{+∞})m with gL≤gU. Note that equality constraints of the form gi(x)=g¯i can be specified by setting gLi=gUi=g¯i.

[TODO: Include reasons why IPOPT is a good solver: open source, mature, wide availability, many options and linear solvers]

  <!-- - open source, widely available, has many options (maybe too many) -->

  <!-- - [TODO: Different choices of linear systems solution algorithms]
    - [TODO: Explain how the "ma57" algorithm from IPOPT work]
    - MUMPS is also good I guess but we have gained practical speedup in 2 orders of magnitute agains the standrad MUMPS sovler using the HSL library -->

> Apart from the cost of evaluating the problem functions and their derivatives, the
work of the interior-point iteration is dominated by the solution of the primal-dual system
(19.12), (19.13). An efﬁcient linear solver, using either sparse factorization or iterative
techniques, is therefore essential for fast solution of large problems. 

  [ TODO: Explain that we use ma57 HSL solever (find citations) but MUMPS open source alternative can be used insted (we however found about 2 orders of magnitude slowdown on our instances)] 

 <!-- - [ ] Interior Point Method -->

- IPOPT is a line-search interior point method
  - [TODO: What is a line-search interior point method and how does it differ from the basic setup]

  <!-- - [ ] IP for Approximate Symmetries
    - IP for quadratic systems -->

> ## For Quadratic Problems Specifically
> 
> When applying interior point methods to nonconvex quadratic problems (where Q is indefinite):
> 
> 1. The objective function is f(x) = (1/2)x^T Q x + c^T x
> 2. The Hessian of the objective is simply Q
> 3. If Q is indefinite, IPOPT will apply inertia correction to ensure the KKT system leads to descent directions
> 
> The Newton system for a quadratic program with linear constraints becomes simpler because the Hessian of the objective is constant (Q) and the constraint Jacobians are also constant.


  <!-- - Special Case : Nonconvex Quadratic Problems -->

  <!-- - Practical Implementation Considerations
  - IPOPT Optimizer
      - Solver specific optimization techniques that make corrections, improvements, special tricks, strategies and approximations to arrive at better solutions and to be more robust and more quick
      - It is a line-search interior point method
        - [TODO: In a sentence what does this mean?] -->

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

  <!-- - [ ] Handling Nonconvexity
      - IPOPT's Approach -->
[TODO: Provide citations for all of the nonconvexity; make it short there need not be much information in this section only overview]

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

 <!-- - [ ] we use analytical gradient computation -->

[TODO: Include analytical gradient and hessian computation for the approximate symmetries problem]


<!-- - [ ] IP for Approximate Symmetries -->

