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

```
\subsection{Barrier Parameter Update}

A crucial component of the algorithm is the strategy for updating the barrier
parameter \(\mu\). Two main approaches have proven effective in practice. The
classical Fiacco-McCormick approach maintains \(\mu\) constant until the KKT
conditions are satisfied to some tolerance, then reduces \(\mu\) by a fixed
factor. Alternatively, adaptive strategies update \(\mu\) at every iteration
based on the progress of the algorithm, often using complementarity-based
heuristics that adjust the reduction rate according to how well the primal and
dual variables are balanced~\cite{nocedal1999numerical}.

The primal-dual framework provides a robust foundation for interior point
methods, combining theoretical soundness with practical efficiency. By
maintaining explicit control over both primal and dual variables and using the
structured Newton system, the algorithm can navigate efficiently through the
interior of the feasible region toward the optimal solution.

\section{Computational Considerations}

While the primal-dual framework provides an elegant theoretical foundation, the
practical success of interior point methods depends critically on our ability
to solve the Newton system efficiently and reliably. This seemingly
straightforward linear algebra problem harbors several computational challenges
that have driven decades of algorithmic innovation.

\subsection{The Computational Challenge}

The most significant computational challenge in interior point methods stems
from the ill-conditioning that inevitably develops as the algorithm approaches
the solution. As the barrier parameter \(\mu\) decreases toward zero, some
elements of the diagonal matrices \(S\) and \(Z\) grow arbitrarily large while
others approach zero. This creates a linear system whose condition number
deteriorates rapidly, potentially leading to inaccurate solutions that
compromise the algorithm's convergence~\cite{nocedal1999numerical}.

More specifically, the matrix \(\Theta = S^{-1}Z\) in the symmetric formulation
of the primal-dual system exhibits this ill-conditioning behavior. Some
diagonal elements of \(\Theta\) diverge to infinity while others converge to
zero as \(\mu \to 0\). Traditional wisdom might suggest that such
ill-conditioning would destroy the accuracy of the computed Newton step, but
surprisingly, the special structure of interior point linear systems often
allows stable direct factorization methods to produce acceptable solutions
despite the poor conditioning.

\subsection{Solution Strategies}

Several approaches have been developed to address these computational
challenges. The most reliable strategy involves using symmetric indefinite
factorization techniques, which can handle the indefinite structure of the
primal-dual matrix while maintaining numerical stability. These methods compute
a factorization of the form \(P^T K P = LBL^T\), where \(P\) represents
permutations for sparsity and stability, \(L\) is lower triangular, and \(B\)
is block diagonal~\cite{nocedal1999numerical}.

For very large problems, iterative methods become attractive alternatives to
direct factorization. However, the ill-conditioning issues that trouble direct
methods become even more problematic for iterative approaches. Successful
iterative methods for interior point systems require sophisticated
preconditioning strategies that cluster the eigenvalues of the coefficient
matrix and remove the artificial ill-conditioning introduced by the barrier
approach.

One elegant approach involves variable scalings that transform the
ill-conditioned terms. For example, the substitution \(\tilde{p}_s = S^{-1/2}
p_s\) can transform the matrix \(\Theta\) into \(S^{1/2} \Theta S^{1/2}\),
which has much better conditioning properties as \(\mu\) approaches
zero~\cite{nocedal1999numerical}.

\subsection{Practical Implications}

These computational challenges have several important practical implications.
First, the choice of linear algebra approach—direct versus iterative—often
depends more on problem structure and size than on theoretical preferences.
Problems with special structure (such as those arising from discretized partial
differential equations) may benefit from specialized iterative methods, while
general nonlinear programs typically favor robust direct methods.

Second, the quality of the linear system solution directly impacts the global
convergence properties of the interior point method. Poor solutions can lead to
steps that make insufficient progress or even cause the algorithm to fail
entirely. This makes the linear algebra component not merely a computational
detail but a core algorithmic concern.

Finally, these challenges explain why interior point methods require more
sophisticated implementation than their elegant theoretical description might
suggest. Successfully handling the transition from well-conditioned systems
early in the optimization to highly ill-conditioned systems near the solution
requires careful attention to numerical details that are often invisible in
high-level algorithmic descriptions.

The computational challenges of interior point methods represent a fascinating
intersection of numerical linear algebra and optimization theory. While these
methods have proven remarkably successful in practice, their computational
demands continue to drive research into more efficient and robust solution
techniques.

\section{Advantages and Limitations}

As with any algorithmic approach, interior point methods come with distinct
advantages and limitations that determine their suitability for different
problem classes. Understanding these trade-offs is essential for making
informed decisions about when to apply these methods in practice.

\subsection{Key Advantages}

The most significant advantage of interior point methods lies in their
exceptional performance on large-scale optimization problems. Numerical
experience demonstrates that these methods frequently outperform active-set SQP
methods on large problems, particularly when the number of free variables is
substantial~\cite{nocedal1999numerical}. This advantage stems from several
factors.

First, interior point methods avoid the combinatorial complexity that plagues
active-set approaches. Rather than explicitly identifying which constraints
should be active at the solution—a process that can require exponential time in
the worst case—interior point methods treat all constraints uniformly
throughout the optimization process. The correct active set emerges naturally
as the barrier parameter decreases, without requiring discrete decisions by the
algorithm.

Second, the linear systems that arise in interior point methods have
predictable block structure that can be exploited by specialized solvers. This
structural regularity enables the development of efficient factorization
techniques and iterative methods that scale well with problem size. Both direct
factorization approaches and projected conjugate gradient methods are
available, allowing practitioners to choose the most appropriate technique for
their specific problem structure~\cite{nocedal1999numerical}.

Third, interior point methods provide strong theoretical guarantees. Under
appropriate regularity conditions, they exhibit polynomial-time complexity for
convex problems and possess well-understood convergence properties for general
nonlinear programs. This theoretical foundation provides confidence in the
algorithm's behavior and enables the development of robust stopping criteria.

Finally, the smooth nature of the central path provides inherent stability.
Unlike active-set methods, which can exhibit cycling behavior or numerical
difficulties when constraints become active or inactive, interior point methods
follow a well-defined continuous trajectory toward the solution.

\subsection{Notable Limitations}

Despite their strengths, interior point methods suffer from several limitations
that can impact their practical effectiveness. The most significant concern is
their sensitivity to problem conditioning and algorithmic parameters. Interior
point methods can be sensitive to the choice of initial point, the scaling of
the problem, and the strategy used for updating the barrier
parameter~\cite{nocedal1999numerical}.

This sensitivity manifests most problematically when iterates approach the
boundary of the feasible region prematurely. In such situations, interior point
methods may experience difficulty escaping the boundary region, leading to slow
convergence or even algorithmic failure. While adaptive strategies for updating
the barrier parameter have reduced this sensitivity, it remains a consideration
for practitioners.

Another limitation stems from the universal treatment of constraints. While the
uniform handling of all constraints is an advantage for problems where many
constraints are active at the solution, it becomes a disadvantage when most
constraints are inactive. Active-set methods can effectively ignore irrelevant
constraints, focusing computational effort on the constraints that actually
matter. In contrast, interior point methods must consider all constraints at
every iteration, potentially making the cost of each iteration excessive for
problems with many inactive constraints~\cite{nocedal1999numerical}.

The robustness gap compared to active-set methods represents another practical
concern. While significant advances continue to be made in interior point
method design and implementation, these methods may not yet match the
robustness of mature active-set codes in challenging problem
instances~\cite{nocedal1999numerical}.

Finally, the specialized linear algebra requirements of interior point methods
can present implementation challenges. The need to handle ill-conditioned
systems and maintain numerical stability as the barrier parameter decreases
requires sophisticated techniques that may not be available in standard linear
algebra libraries.

\subsection{Practical Recommendations}

The choice between interior point and active-set methods often depends on
problem characteristics rather than theoretical preferences. Interior point
methods excel for large-scale problems with dense constraint structure, where
their ability to handle many constraints simultaneously provides clear
advantages. They are particularly well-suited to problems arising from
discretized continuous optimization problems, where the constraint structure
exhibits regularity that can be exploited.

Conversely, active-set methods may be preferable for problems where the optimal
active set is small relative to the total number of constraints, or where
robustness is more important than raw computational speed. The availability of
warm-start capabilities in active-set methods also makes them attractive for
solving sequences of related problems.

Understanding these trade-offs enables practitioners to make informed
algorithmic choices that align with their specific problem requirements and
computational constraints.
```
