## Manifold
[SUBTITLE: Exploiting the geometry of the search space.]


<!-- [INSERT: Personal Artistic Intro to Methods] -->

- Following the mature Interior point we present a relative newcomer to the game [TODO: Include the papers that introduced and popularized manifold optimization] that is manifold optimization.

- manifold optimization allows us to only consider the space we are working on, potentially decreasing the problem dimension and leading to faster algorithms

- comprehensive overview of Riemannian optimization algorithms as an alternative to constrained Euclidean optimization 

<!-- Specific Challenges in Manifold Optimization -->
> Vector Transport: Moving vectors between different tangent spaces requires special operations.
- This is not used in our steepest descent approach
> Second-Order Information: The Riemannian Hessian incorporates both the Euclidean Hessian and the manifold's curvature.
- this is not used in our steepest descent approach either
> Line Search: Requires special care since straight lines don't generally stay on the manifold.

> Manifold Optimization Over the Set of Doubly Stochastic Matrices: A Second-Order Geometry
> This paper by Ahmed Douik and Babak Hassibi introduces a novel geometric framework for solving optimization problems where the variable of interest is a doubly stochastic matrix. The core innovation is reformulating these constrained optimization problems as unconstrained problems over specialized manifolds.
> Key Contributions
> The authors introduce Riemannian manifolds:
> 
> Doubly stochastic multinomial manifold (DPn): For doubly stochastic matrices (positive entries with rows and columns summing to 1)

> Why This Matters
> Traditional approaches to these optimization problems typically use interior-point methods, which become prohibitively slow in high dimensions due to the "curse of dimensionality." The Riemannian approach exploits the lower-dimensional structure of the constraint set, leading to more efficient algorithms.

> Technical Framework
> For each manifold, the paper develops a complete second-order geometric structure by deriving:
> 
> The tangent space definition
> Orthogonal projections onto the tangent space
> Riemannian gradient (relating Euclidean and Riemannian gradients)
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

> Connection to Optimization
> In Riemannian optimization algorithms for doubly-stochastic matrices, the exponentiation problem appears when updating the current iterate:
> Pk+1=RPk(−α⋅gradf(Pk))P_{k+1} = R_{P_k}(-\alpha \cdot \text{grad}f(P_k))Pk+1​=RPk​​(−α⋅gradf(Pk​))
> where RR
> R is a retraction that approximates the exponential map, and α\alpha
> α is the step size.

<!-- - [ ] The Doubly Stochastic Manifold -->

- [TODO: Provide a definition of the tangent space, description DS; extract from paper]
- [TODO: Provide a definition of how to project to the tangent space DS; extract from pape]

> The sources provide a comprehensive discussion of the Doubly Stochastic Multinomial Manifold ($DP_n$) within the context of specific manifolds studied for optimization problems, emphasizing its structure, relevant geometric computations, and applications
> .
> Here's what the sources say about $DP_n$:
> •
> Definition and Context
> ◦
> The set of doubly stochastic matrices ($DP_n$) is defined as the set of square matrices ($n \times n$) with positive entries such such that each column and row sums to 1
> . This can only be true for square matrices
> .
> ◦
> It is identified as an embedded manifold of $\mathbb{R}^{n \times n}$
> . This means it's a smooth subset of a higher-dimensional Euclidean space, allowing for local linear approximations
> .
> ◦
> The problem of optimizing functions where the variable is a doubly stochastic matrix is a key focus of the paper
> . Such frameworks are particularly interesting for clustering applications, where one wishes to recover the structure of a graph given a similarity matrix by minimizing a cost function under the doubly stochastic constraint


> The Doubly Stochastic Multinomial Manifold (DPn) is an embedded manifold of the space of $n \times n$ matrices, $\mathbb{R}^{n \times n}$. It consists of square matrices with positive entries where each column and row sums to 1. The dimension of DPn is $(n-1)^2$
> .

> The set of doubly stochastic matrices, $DP_n$, is an example of such an embedded manifold
> .
> •
> A doubly stochastic matrix is a square matrix ($X \in \mathbb{R}^{n \times n}$) with positive entries ($X_{ij} > 0$) where each row and each column sums to 1 ($X\mathbf{1} = \mathbf{1}$ and $X^T\mathbf{1} = \mathbf{1}$)
> .
> •
> $DP_n$ is an embedded manifold of $\mathbb{R}^{n \times n}$
> . This is supported by the Birkhoff-von Neumann theorem
> .
> •
> The dimension of $DP_n$ is $(n-1)^2$
> . This dimension arises because the constraints for a matrix $X$ to be doubly stochastic ($X\mathbf{1} = \mathbf{1}$ and $X^T\mathbf{1} = \mathbf{1}$) impose $2n-1$ linearly independent equations. In a space of $n^2$ variables (for $X \in \mathbb{R}^{n \times n}$), subtracting these constraints gives $n^2 - (2n-1) = (n-1)^2$ degrees of freedom

> .
> •
> For any point $X$ on $DP_n$, its tangent space $T_X DP_n$ is defined by matrices $Z \in \mathbb{R}^{n \times n}$ such that $Z\mathbf{1} = \mathbf{0}$ and $Z^T\mathbf{1} = \mathbf{0}$
> . These conditions on $Z$ are the linearization of the row and column sum constraints around $X$, analogous to the kernel of the derivative of a local defining function $h(X) = (X\mathbf{1} - \mathbf{1}, X^T\mathbf{1} - \mathbf{1})^T$. The dimension of this tangent space matches the manifold's dimension $(n-1)^2$, confirming its manifold structure.

> .
> The paper by Douik and Hassibi
> , which discusses manifold optimization, also aligns with this fundamental definition of finding an extreme point of a function over a domain $D \subset \mathbb{R}^n$. Riemannian optimization, as described in this paper, aims to transform constrained Euclidean optimization problems (where the search set $D$ is constrained, i.e., $D \subset \mathbb{R}^n$) into an unconstrained optimization problem over a restricted search space, which is a Riemannian manifold. This reformulates the problem back into the core "Step 0" structure, but with the set $S$ being the manifold itself, allowing algorithms to exploit its geometric structure. This is particularly advantageous because the manifold may be of lower dimension than the ambient Euclidean space



<!-- - [ ] Manifold - Concrete form for DS -->

> The paper introduces $DP_n$ as one of three manifolds (doubly stochastic, symmetric, and definite multinomial manifolds) that generalize the simplex (multinomial manifold) to solve convex programs where the variable of interest is a multidimensional probability distribution function
> .
> •
> Manifold Geometry and Dimension
> ◦
> As a consequence of the Birkhoff-von Neumann theorem, $DP_n$ is an embedded manifold of $\mathbb{R}^{n \times n}$
> .
> ◦
> The dimension of $DP_n$ is $(n-1)^2$
> . This dimension arises from the fact that the manifold is generated from $2n-1$ linearly independent equations (row and column sum constraints). The dimension of its tangent space is equal to the manifold's dimension
> .
> •

<!-- - [ ] Tangent Space - Concrete form for DS -->

> Tangent Space ($T_XDP_n$)
> ◦
> For a point $X \in DP_n$, the tangent space $T_XDP_n$ is defined as the set of all matrices $Z \in \mathbb{R}^{n \times n}$ such that $Z\mathbf{1} = \mathbf{0}$ and $Z^T\mathbf{1} = \mathbf{0}$
> . Here, $\mathbf{1}$ is the all-ones vector and $\mathbf{0}$ is the all-zeros vector
> .
> ◦
> These conditions (Z1 = 0 and ZT1 = 0) yield $2n-1$ linearly independent constraints, confirming the $(n-1)^2$ dimension of the tangent space
> .
> ◦
> Tangent vectors are defined as the velocity $c'(0)$ of a smooth curve $c: \mathbb{R} \to \mathcal{M}$ passing through $X$ at $t=0$
> . For embedded submanifolds defined by $h(x)=0$, the tangent space $T_x\mathcal{M}$ is equivalent to the kernel of the derivative of $h$ at $x$ ($ \text{ker } D h(x)$)
> .
> •

> Here's how the tangent space of the DPn manifold can be described, how its projection looks, and how it can be computed:
> Description of the Tangent Space
> For any point $X$ within the Doubly Stochastic Multinomial Manifold (DPn), the tangent space, denoted as $T_X \text{DPn}$, is a linear space defined as the set of all $n \times n$ matrices $Z$ such that their row sums and column sums are all zeros
> . Formally: $T_X \text{DPn} = { Z \in \mathbb{R}^{n \times n} \mid Z\mathbf{1} = \mathbf{0}, Z^T\mathbf{1} = \mathbf{0} }$
> .
> This means that if you have a smooth curve $X(t)$ on the manifold such that $X(0) = X$, its derivative $\dot{X}(t)$ at $t=0$ will be a tangent vector $Z$ satisfying the condition that its row and column sums are zero
> . The dimension of this tangent space is $n^2 - (2n-1) = (n-1)^2$, which matches the dimension of the manifold itself
> .

<!-- - [ ] Projection and Retraction - Concrete form for DS (Sinkhorn projection) -->

> Orthogonal Projection ($\Pi_X$)
> ◦
> The orthogonal projection $\Pi_X: \mathbb{R}^{n \times n} \to T_XDP_n$ maps a point from the ambient space onto the tangent space, orthogonal with respect to the Riemannian metric
> .
> ◦
> The sources provide a detailed expression for $\Pi_X(Z) = Z - (\alpha\mathbf{1}^T + \mathbf{1}\beta^T) \odot X$, where $\alpha$ and $\beta$ are vectors obtained by solving a linear system derived from the tangent space conditions
> . This projection is crucial for relating the Riemannian gradient and Hessian to their Euclidean counterparts
> .
> •

> Retraction ($R_X(\xi_X)$)
> ◦
> A retraction is a smooth mapping from the tangent bundle $T\mathcal{M}$ onto the manifold $\mathcal{M}$ that approximates the exponential map, allowing optimization steps computed in the linear tangent space to be brought back onto the curved manifold
> .
> ◦
> Two retractions are discussed for $DP_n$:
> 1.
> Canonical Retraction: $R_X(\xi_X) = X + \xi_X$
> . This is a "well-defined retraction... provided that $\xi_X$ is in the neighborhood of $0_X$, i.e., $X_{ij} > - (\xi_X)_{ij}$". This retraction is simple but its performance can suffer if the optimal solution has vanishing entries
> .
> 2.
> Alternative Retraction (based on Sinkhorn-Knopp algorithm): $R_X(\xi_X) = P(X \odot \exp(\xi_X \oslash X))$, where $P(\cdot)$ is the projection onto the set of doubly stochastic matrices using the Sinkhorn-Knopp algorithm
> . This retraction is more robust for all $\xi_X \in T DP_n$ and does not require $\xi_X$ to be in a small neighborhood of $X$ but is computationally more expensive. It is noted that the canonical retraction is a first-order approximation of this more complex one.


> Projection to the Tangent Space
> The orthogonal projection operator, $\Pi_X$, maps a matrix $Z$ from the ambient space $\mathbb{R}^{n \times n}$ onto the tangent space $T_X \text{DPn}$
> . This projection is orthogonal with respect to the Riemannian metric defined on the tangent space
> .
> The expression for this orthogonal projection $\Pi_X(Z)$ is given by: $\Pi_X(Z) = Z - (\alpha\mathbf{1}^T + \mathbf{1}\beta^T) \odot X$
> where $\odot$ denotes the Hadamard (element-wise) product, $\mathbf{1}$ is the all-ones vector, and $\alpha, \beta$ are vectors
> .
> The term $(\alpha\mathbf{1}^T + \mathbf{1}\beta^T) \odot X$ represents the projection onto the orthogonal complement of the tangent space, $T_X^\perp \text{DPn}$
> .
> How to Compute the Projection
> To compute $\Pi_X(Z)$, you first need to determine the vectors $\alpha$ and $\beta$
> . These vectors are obtained by solving a system of linear equations derived from the definition of the tangent space and its orthogonal complement
> .
> The exact expressions for $\alpha$ and $\beta$ are
> : $\alpha = (I - XX^T)^\dagger(Z - XZ^T)\mathbf{1}$ $\beta = Z^T\mathbf{1} - X^T\alpha$ where $Y^\dagger$ is the left-pseudo inverse that satisfies $Y^\dagger Y = I$
> .
> For numerical stability in implementation, $\alpha$ and $\beta$ are typically obtained by solving the following linear system
> : $ \begin{pmatrix} Z\mathbf{1} \ Z^T\mathbf{1} \end{pmatrix} = \begin{pmatrix} I & X \ X^T & I \end{pmatrix} \begin{pmatrix} \alpha \ \beta \end{pmatrix} $ This system has infinitely many solutions, but a particular solution can be found because the vector of interest is orthogonal to the null space of the matrix (I X; X^T I).


<!-- - [ ] Riemannian Gradient - Concrete form for DS (using projection) -->

> Riemannian Gradient (grad $f(X)$)
> ◦
> The Riemannian gradient is derived from the Euclidean gradient by projecting a scaled version of the Euclidean gradient onto the tangent space
> .
> ◦
> grad $f(X) = \Pi_X(\text{Grad } f(X) \odot X)$, where $\text{Grad } f(X)$ is the Euclidean gradient
> . This identity is applicable to all three manifolds of interest due to the Fisher information metric used
> .
> ◦
> The explicit expression for the Riemannian gradient on $DP_n$ is provided, involving $\text{Grad } f(X) \odot X$ and the projection terms $\alpha$ and $\beta$
> .
> •

<!-- - [ ] Algorithm Used
  - [ ] Steepest descent
      - [ ] Include complexity and empirical performance of the approach (from paper) -->

- we use simple projection because we use simple gradient descent algorithms
  - for more complex algorithms such as conjugate gradient vector transport or parallel transport would be needed for DS Manifold 
    - [TODO: CHeck what is available from the paper on DS manifolds]

> Optimization Algorithms and Complexity
> ◦
> The paper analyzes the complexity of steepest descent (Algorithm 1) and Newton's method (Algorithm 2) on $DP_n$
> .
> ◦
> Gradient Descent Complexity: For $DP_n$, one iteration has a total complexity of $(16/3)n^3 + 7n^2 + \log(n)\sqrt{n}$
> . This cubic term mostly comes from solving the linear system for $\alpha$ (which is $O((2n)^3)$) and the projection step for retraction (Sinkhorn-Knopp algorithm).

> Simulation results show that optimization algorithms on $DP_n$ (e.g., Conjugate Gradient and Trust Region methods from Manopt toolbox) significantly outperform generic solvers like CVX, with speed gains ranging from 39 to 123 fold for first-order methods and 4 to 11 fold for second-order methods. This efficiency is attributed to exploiting the manifold's geometry.


<!-- Conclusion -->

> In summary, the sources meticulously define $DP_n$ as an embedded manifold, detailing its linear tangent space as the local approximation. They provide explicit formulas for the tangent space, orthogonal projection, Riemannian gradient, and two retraction strategies, all derived from its underlying geometry and the chosen Fisher information metric. The computational complexity of common optimization methods on $DP_n$ is rigorously analyzed, and empirical results confirm its superior performance in high-dimensional optimization problems, particularly in clustering, by leveraging its inherent geometric structure compared to general-purpose solvers.


<!-- - [ ] Manifold optimization for approximate symmetries -->

<!-- - [ ] Manopt (Library for optimization on manifolds)
    -  pytorch for automatic differentiation -->

- mention the Manopt (PyManopt) library is used for optimization

<!-- - Describe how exactly we adapt the manifold optimization framework for the problem of approximate symmetries -->

 <!-- - [  ] note AutoDiff using pytorch is being used -->

```
\section{Manifold Optimization Algorithms}

We now have all the tools needed to generalize classical optimization algorithms
to manifolds: tangent spaces for local linearization, Riemannian gradients for
descent directions, and retractions for movement along the manifold.

\subsection{Riemannian Steepest Descent}

\subsection{Advanced Algorithms}

The framework extends naturally to more sophisticated methods:

\begin{itemize}
  \item \textbf{Riemannian Newton's Method}: Uses second-order
    information via the Riemannian Hessian, requiring concepts of
    parallel transport for vector fields on manifolds
  \item \textbf{Riemannian Trust Region}: Builds quadratic models in
    tangent spaces and uses retractions to return to the manifold
    \cite{manifold_small}
  \item \textbf{Riemannian Conjugate Gradients}: Constructs conjugate
    directions using vector transport to connect tangent spaces at
    different points
\end{itemize}

These advanced methods require additional geometric tools beyond the
scope of this chapter, but they demonstrate the rich algorithmic
possibilities enabled by the manifold optimization framework.

\section{Benefits and Applications}

Manifold optimization offers compelling advantages over classical
constrained optimization approaches, particularly for problems where
constraints naturally define geometric structure.

\subsection{Dimension Reduction}

Perhaps the most significant computational advantage is dimension
reduction. When optimizing over an $n$-dimensional manifold
$\mathcal{M}$ embedded in $\mathbb{R}^d$ with $n < d$, we effectively
reduce the problem dimension from $d$ to $n$. This reduction can be dramatic:

\begin{itemize}
  \item The Stiefel manifold $\text{St}(p, k)$ of $p \times k$
    orthogonal matrices has dimension $pk - k(k+1)/2$, significantly
    less than the $pk$ ambient dimension
  \item Spheres $\mathcal{S}^{d-1}$ have dimension $d-1$ instead of $d$
  \item Fixed-rank matrix manifolds have dimension much smaller than
    the full matrix space
\end{itemize}

This dimension reduction translates directly to computational savings
in gradient computations, storage requirements, and algorithmic complexity.

\subsection{Geometric Structure Preservation}

Beyond dimension reduction, manifold optimization offers several
structural advantages:

\begin{itemize}
  \item \textbf{Automatic constraint satisfaction}: All iterates
    remain feasible by construction
  \item \textbf{Natural parameterization}: The manifold structure
    reflects the problem's inherent geometry
  \item \textbf{Numerical stability}: Working directly on the
    constraint manifold often improves conditioning
  \item \textbf{Algorithmic elegance}: Many algorithms simplify when
    constraints are treated geometrically rather than algebraically
\end{itemize}

\subsection{Applications Preview}

The manifold optimization framework applies to numerous important
problem classes. One particularly relevant example for this thesis
involves doubly-stochastic matrices—matrices where all row and column
sums equal one. These matrices form a complex manifold with both
equality constraints (sum conditions) and inequality constraints
(non-negativity), making them ideal candidates for the geometric
optimization approaches developed in this chapter.

The intersection of manifold optimization with discrete optimization
problems, particularly those involving permutation-like structures,
represents a fertile area where the dimension reduction and
constraint-handling benefits of geometric methods can provide
substantial computational advantages.

The tools developed in this chapter—tangent spaces, Riemannian
gradients, and retractions—form the mathematical foundation for
tackling such challenging optimization problems while respecting
their inherent geometric structure.
```
