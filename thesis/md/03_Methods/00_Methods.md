# Methods

- what follows here in this chapter is a description of the methods I was using for optimization of the approximate symmetry problem. The aim is not to provide a deep and and in depthexposition to the theory behind baic methods, but the general idea an neccessary framework to understand the concepts, implications , motivation and beuty of the individual methods. While we know that the explanations will be lacking in places, the we will make sure to poiint the reader to the correct resourcs that will opefully satiate her Neugierigkeit.

- the chapther will be organized as follows, firstly we present the QSA algorithm which is the starting point of this entire journey as well as the starting point of out rhinking process regarding the search for a better, more optimized, more robus, nmore efficient and in short better (for the lack of better term) algorithm. Next we think about the natural step whenm viewing the problem in the lens of constrained optimization and thatwe can utilize the robust and mature environment of constraint programming solvers. We utilize the InteriorPoint solver in hope of beating the performance of the simple Frank-Wolfe algorithm. Next, we try to incorporate the geometry of the space fo double stochastic matrices into aout problem .Until now, we have thought about the constraints as more or less black box feasible-space defining equations. The feasible requion tha of the Birkhoff polytope however has a a special structure and that is that of a smooth surface - manifold.We continue in the discussion of feasible reqion geometry and manifold optimization in the next chapter as well, where we consider a slightly different relaxation of the set of all permutations. We relax the permutations into the space of all orthogonal matrices using. Using the more efficient parametrization of matrices in this space we will be able to reduce the problem dimension and gain new isight on the problem of optimizing over permutations. Last but not least, we will try to reduce the dimension of the problem even further. Borrowing from the research of neural networks we will try to reduce the problem of optimizing over permutation matrices back to vectors through differentiable mappings.

### QSA

- the QSA method is the original method as well as my starting point in the journey of styduing optimization problems of approximate symmetries over on the relaxed space of permutation matrices the birkhoff polytope. Introduced b yVogelstein, this algorithm was innitial proposed as a fast algorithm for solving QAP problems, usefeul lfor solving the GMP. 

- In the following exposition we will briefly describe the algorithm in it's entirety, since the original paper contain a mistake and the adjusted objective function of the approximate symmetry problem results in a slightly different formulation of the algorithm. The algorithm is adapted almost verbatim from the Vogelstein paper.

- It is also the algotorithm Harmtna and pidnebesna chose in their study of approxximate symmetries. This exposition also helps because this algorithm will provide us with a benchmark or somethin. It is the starting point and it is also the goal we want to overcome overtake ans surpass aeither in terms of performance, quality of results or speed.

- As can be seen int the following chapters, the algorithm is quaite simple and it is the result of astraightforward application of the Frank-WOlfe algorithm [TODO: INsret citation] . This simplicity brings the tough task or quest of exceptional performance. However it is stil;l a first order method. And we might surpass it later with algorithmthat uses more information from the problem.


```latex
\documentclass{article} % Or use `report` or `book` for a thesis structure with \chapter
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath, amssymb, mathtools} % For math
\usepackage{bm} % For bold math symbols (vectors and matrices)
\usepackage{algorithm} % For floating algorithm environment
\usepackage{algpseudocode} % For typesetting algorithms (algorithmicx style)
\usepackage{geometry} % For page layout, if needed
\geometry{a4paper, margin=1in} % Example page layout

% Custom operators
\DeclareMathOperator{\tr}{tr}
\DeclareMathOperator*{\argmin}{arg\,min}
\DeclareMathOperator*{\argmax}{arg\,max}

% Macros for sets (optional, for consistency)
\newcommand{\setD}{\mathcal{D}_n}
\newcommand{\setP}{\mathcal{P}_n}

% Define a title for the section/chapter
\newcommand{\chaptertitle}{The Quadratic Symmetry Approximator (QSA) Algorithm}

\begin{document}

% If using `report` or `book` class, use \chapter{\chaptertitle}
% If using `article` class, use \section{\chaptertitle}
\section*{\chaptertitle} % Using \section* for unnumbered section in article, adjust as needed

The Quadratic Symmetry Approximator (QSA) algorithm is an iterative method designed to find an approximate solution to a specific quadratic assignment-like problem. It is based on the Frank-Wolfe algorithm \cite{Frank1956}, also known as the conditional gradient method, which is well-suited for optimization problems with convex feasible sets, such as the set of doubly stochastic matrices.

\subsection*{Problem Formulation}

The classical Quadratic Assignment Problem (QAP) seeks to find a permutation matrix $\bm{P} \in \setP$ (where $\setP$ is the set of $n \times n$ permutation matrices) that minimizes an objective function, often of the form $f(\bm{P}) = \tr(\bm{A} \bm{P} \bm{B}^T \bm{P}^T)$, where $\bm{A}$ and $\bm{B}$ are given square matrices \cite{Koopmans1957}. This formulation is central to problems like graph matching, where $\bm{A}$ and $\bm{B}$ might represent adjacency matrices of two graphs, and the objective is equivalent to minimizing $-\tr(\bm{A} \bm{P} \bm{B}^T \bm{P}^T)$ to maximize edge overlap \cite{Vogelstein2014}.

The QSA algorithm addresses a related but distinct objective function:
\begin{equation} \label{eq:qsa_objective}
f(\bm{P}) = -\tr(\bm{A} \bm{P} \bm{A} \bm{P}^T) + \tr(\text{diag}(\bm{c}) \bm{P})
\end{equation}
where $\bm{A}$ is an $n \times n$ symmetric matrix (i.e., $\bm{A} = \bm{A}^T$), $\bm{P}$ is the $n \times n}$ matrix variable, and $\bm{c}$ is an $n$-dimensional vector. The first term, $-\tr(\bm{A} \bm{P} \bm{A} \bm{P}^T)$, captures a quadratic interaction related to structural alignment, similar to the QAP but using the matrix $\bm{A}$ in both roles. The second term, $\tr(\text{diag}(\bm{c}) \bm{P}) = \sum_{i=1}^n c_i P_{ii}$, serves as a linear penalty or reward for the diagonal elements of $\bm{P}$. This term can be used, for example, to penalize ($c_i < 0$) or encourage ($c_i > 0$) fixed points in the assignment represented by $\bm{P}$.

Due to the combinatorial nature of optimizing over $\setP$, the QSA algorithm operates on a relaxed feasible set: the Birkhoff polytope $\setD$, which is the set of $n \times n$ doubly stochastic matrices. A matrix $\bm{P} \in \setD$ satisfies $\bm{P}_{ij} \ge 0$ for all $i,j$, $\sum_{j=1}^n P_{ij} = 1$ for all $i$, and $\sum_{i=1}^n P_{ij} = 1$ for all $j$. The objective function \eqref{eq:qsa_objective} is not necessarily convex, making the relaxed problem an indefinite quadratic program. The Frank-Wolfe algorithm applied to this problem seeks a stationary point.

\subsection*{The QSA Algorithm}
The QSA algorithm iteratively refines an estimate $\bm{P}$ of the solution. Each iteration involves computing the gradient of the objective function, solving a linear assignment problem to find a search direction, determining an optimal step size along this direction, and updating the current solution. The algorithm is detailed in Algorithm~\ref{alg:qsa}.

\begin{algorithm}[H]
\caption{QSA for finding a local optimum of the relaxed problem}
\label{alg:qsa}
\begin{algorithmic}[1]
\Require Symmetric matrix $\bm{A} \in \mathbb{R}^{n \times n}$, penalty vector $\bm{c} \in \mathbb{R}^n$, initial doubly stochastic matrix $\bm{P}^{(0)} \in \setD$, maximum number of iterations `max_iter`, tolerance $\epsilon_{\text{tol}} > 0$.
\Ensure An estimated doubly stochastic matrix $\bm{P} \in \setD$.

\State $\bm{P} \leftarrow \bm{P}^{(0)}$
\For{$k = 0, \dots, \text{max\_iter}-1$}
    \State $\bm{P}_{\text{prev}} \leftarrow \bm{P}$
    \State \Comment{Compute Gradient}
        \State $\nabla f(\bm{P}) \leftarrow -2 \bm{A} \bm{P} \bm{A} + \text{diag}(\bm{c})$
    \State \Comment{Solve Linear Subproblem (Linear Assignment Problem)}
        \State $\bm{Q}^{(k)} \leftarrow \argmin_{\bm{Q} \in \setD} \tr((\nabla f(\bm{P}))^T \bm{Q})$
        \Comment{$\bm{Q}^{(k)}$ will be a permutation matrix.}
    \State \Comment{Determine Search Direction}
        \State $\bm{R}^{(k)} \leftarrow \bm{Q}^{(k)} - \bm{P}$
    \State \Comment{Line Search: Minimize $g(\alpha) = f(\bm{P} + \alpha \bm{R}^{(k)})$ for $\alpha \in [0, 1]$}
        \State Let $g(\alpha) = a_0 \alpha^2 + b_0 \alpha + f(\bm{P})$, where
        \State $a_0 \leftarrow -\tr(\bm{A} \bm{R}^{(k)} \bm{A} (\bm{R}^{(k)})^T)$
        \State $b_0 \leftarrow -\tr(\bm{A} \bm{P} \bm{A} (\bm{R}^{(k)})^T) - \tr(\bm{A} \bm{R}^{(k)} \bm{A} \bm{P}^T) + \tr(\text{diag}(\bm{c}) \bm{R}^{(k)})$
        \If{$|a_0| < \epsilon_a$} \Comment{$\epsilon_a$ is a small tolerance for $a_0 \approx 0$}
            \If{$b_0 > 0$}
                \State $\alpha^{(k)} \leftarrow 0$
            \Else
                \State $\alpha^{(k)} \leftarrow 1$
            \EndIf
        \ElsIf{$a_0 > 0$} \Comment{Parabola opens upwards}
            \State $\alpha_{\text{vertex}} \leftarrow -b_0 / (2 a_0)$
            \State $\alpha^{(k)} \leftarrow \max(0, \min(1, \alpha_{\text{vertex}}))$
        \Else \Comment{$a_0 < 0$, parabola opens downwards, minimum on boundary}
            \If{$(a_0 + b_0) > 0$} \Comment{Equivalent to $g(1) > g(0)$}
                 \State $\alpha^{(k)} \leftarrow 0$
            \Else
                 \State $\alpha^{(k)} \leftarrow 1$
            \EndIf
        \EndIf
    \State \Comment{Update Iterate}
        \State $\bm{P} \leftarrow \bm{P} + \alpha^{(k)} \bm{R}^{(k)}$
    \State \Comment{Check Convergence}
        \State $\text{change} \leftarrow \|\bm{P} - \bm{P}_{\text{prev}}\|_F / \sqrt{n}$
        \If{$\text{change} < \epsilon_{\text{tol}}$}
            \State \textbf{break}
        \EndIf
\EndFor
\State \Return $\bm{P}$
\end{algorithmic}
\end{algorithm}

The linear subproblem in Step 6 is a Linear Assignment Problem (LAP), which can be solved efficiently by algorithms such as the Hungarian algorithm or the Jonker-Volgenant algorithm \cite{Kuhn1955, Jonker1987}. The solution $\bm{Q}^{(k)}$ to this LAP will always be a permutation matrix, which is a vertex of the Birkhoff polytope $\setD$.

\subsection*{Initial Position and Multiple Restarts}
The QSA algorithm, like many non-convex optimization methods, may converge to a local optimum that depends on the initial doubly stochastic matrix $\bm{P}^{(0)}$. Several strategies can be employed for choosing $\bm{P}^{(0)}$:
\begin{itemize}
    \item \textbf{Barycenter:} The matrix $\bm{J}_n$, where all entries are $1/n$. This represents the most non-informative starting point.
    \item \textbf{Random Permutation Matrix:} A matrix chosen uniformly at random from $\setP$.
    \item \textbf{Random Doubly Stochastic Matrix:} Generated, for instance, by applying a few iterations of the Sinkhorn-Knopp algorithm to a random matrix with positive entries, or by taking convex combinations of random permutation matrices.
    \item \textbf{User-Supplied Matrix:} A specific $\bm{P}^{(0)}$ based on prior knowledge or a previous estimation.
\end{itemize}
Due to the dependence on initialization, a common practice is to employ a multiple restart strategy. This involves running the QSA algorithm multiple times (e.g., 5 times as considered in practical applications for this work) from different initial positions $\bm{P}^{(0)}$, typically chosen randomly (e.g., random doubly stochastic matrices or random permutations). The final solution is then selected as the one that yields the lowest objective function value $f(\bm{P})$ among all runs.

\subsection*{Projection to Permutation Matrices}
The QSA algorithm (Algorithm~\ref{alg:qsa}) outputs a doubly stochastic matrix $\bm{P} \in \setD$, which represents a "soft" assignment. In many applications, a discrete assignment in the form of a permutation matrix $\bm{P}_{\text{perm}} \in \setP$ is required. The doubly stochastic matrix $\bm{P}$ obtained from QSA can be projected onto the set of permutation matrices.

A standard method for this projection is to solve another Linear Assignment Problem:
\begin{equation} \label{eq:projection_lap}
\bm{P}_{\text{perm}} = \argmax_{\bm{X} \in \setP} \tr(\bm{P}^T \bm{X})
\end{equation}
This finds the permutation matrix $\bm{X}$ that is "closest" to $\bm{P}$ in the sense of maximizing the inner product, which is equivalent to minimizing the Frobenius norm $\|\bm{P} - \bm{X}\|_F^2$ subject to $\bm{X} \in \setP$. This final LAP can again be solved using standard algorithms. Other projection or rounding schemes might also be applicable depending on the specific requirements of the problem.

% --- Bibliography (example) ---
% In a real thesis, you'd use a .bib file and a bibliography style.
\begin{thebibliography}{9}
\bibitem{Frank1956}
Frank, M., \& Wolfe, P. (1956). An algorithm for quadratic programming. \textit{Naval Research Logistics Quarterly}, 3(1-2), 95-110.

\bibitem{Koopmans1957}
Koopmans, T. C., \& Beckmann, M. (1957). Assignment problems and the location of economic activities. \textit{Econometrica}, 25(1), 53-76.

\bibitem{Vogelstein2014}
Vogelstein, J. T., Conroy, J. M., Lyzinski, V., Podrazik, L. J., Kratzer, S. G., Harley, E. T., ... \& Priebe, C. E. (2014). Fast approximate quadratic programming for large (brain) graph matching. \textit{arXiv preprint arXiv:1112.5507}. (Note: This refers to the paper you provided, cite the published version if available).

\bibitem{Kuhn1955}
Kuhn, H. W. (1955). The Hungarian method for the assignment problem. \textit{Naval Research Logistics Quarterly}, 2(1-2), 83-97.

\bibitem{Jonker1987}
Jonker, R., \& Volgenant, A. (1987). A shortest augmenting path algorithm for dense and sparse linear assignment problems. \textit{Computing}, 38(4), 325-340.

\end{thebibliography}

\end{document}
```

### Interior Point Method

- Next in our search fro a better algorithm comes in form of the natural solution of the to one of the most successful solvers of non-convex nonlinear programming methods. Om this chapter we will briefly introduce the idea of interior point barrieer methods. We will explain at which parts we might gain an advantage and which parts of the algotithm will prove disadvatageous

- one of the main driving force that we are leaning on here is the relative maturityof the method wiith well developed ecosystem and availability of tools and implementatipons of algorithms.

 - what follows is a brief excursion and introduction to the interiror point barrier method almost entirely inspired by the exposition in the Nocedal book with slight change in notation.

- Read nocedal if you are hyngry for more

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

- explain the basic idea and motivation behind the interior point method (nocedal, hartman presentation)
- why it is interesting 

#### Algorithmic Choices
[SUBTITLE: Using more sophisticated algorithms]

- We use IPOPT interior point solver
  - open source, widely available, has many options (maybe too many)
  - [TODO: Different choices of linear systems solution algorithms]
    - [TODO: Explain how the "ma57" algorithm from IPOPT work]
    - MUMPS is also good I guess but we have gained practical speedup in 2 orders of magnitute agains the standrad MUMPS sovler using the HSL library

### Manifold
[SUBTITLE: Exploiting the geometry of the search space.]

- Following the mature Interior point we present a relative newcomer to the game [TODO: Include the papers that introduced and popularized manifold optimization] that is manifold optimization.

- manifold optimization allows us to only consider the space we are working on, potentially decreasing the problem dimension and leading to faster algorithms

- comprehensive overview of Riemannian optimization algorithms as an alternative to constrained Euclidean optimization 


#### Specifics For Doubly Stochastic Manifold

- we use simple projection because we use simple gradient descent algorithms
  - for more complex algorithms such as conjugate gradient vector transport or parallel transport would be needed for DS Manifold 
    - [TODO: CHeck what is available from the paper on DS manifolds]

- mention the Manopt (PyManopt) library is used for optimization

- [TODO: Provide a definition of the tangent space, description]
- [TODO: Provide a definition of how to project to the tangent space]
- [TODO: Question??? WHat is FIsher metric?]

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
> .
> ◦
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
> Orthogonal Projection ($\Pi_X$)
> ◦
> The orthogonal projection $\Pi_X: \mathbb{R}^{n \times n} \to T_XDP_n$ maps a point from the ambient space onto the tangent space, orthogonal with respect to the Riemannian metric
> .
> ◦
> The sources provide a detailed expression for $\Pi_X(Z) = Z - (\alpha\mathbf{1}^T + \mathbf{1}\beta^T) \odot X$, where $\alpha$ and $\beta$ are vectors obtained by solving a linear system derived from the tangent space conditions
> . This projection is crucial for relating the Riemannian gradient and Hessian to their Euclidean counterparts
> .
> •
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
> 

> Optimization Algorithms and Complexity
> ◦
> The paper analyzes the complexity of steepest descent (Algorithm 1) and Newton's method (Algorithm 2) on $DP_n$
> .
> ◦
> Gradient Descent Complexity: For $DP_n$, one iteration has a total complexity of $(16/3)n^3 + 7n^2 + \log(n)\sqrt{n}$
> . This cubic term mostly comes from solving the linear system for $\alpha$ (which is $O((2n)^3)$) and the projection step for retraction (Sinkhorn-Knopp algorithm).

> Simulation results show that optimization algorithms on $DP_n$ (e.g., Conjugate Gradient and Trust Region methods from Manopt toolbox) significantly outperform generic solvers like CVX, with speed gains ranging from 39 to 123 fold for first-order methods and 4 to 11 fold for second-order methods. This efficiency is attributed to exploiting the manifold's geometry.

> In summary, the sources meticulously define $DP_n$ as an embedded manifold, detailing its linear tangent space as the local approximation. They provide explicit formulas for the tangent space, orthogonal projection, Riemannian gradient, and two retraction strategies, all derived from its underlying geometry and the chosen Fisher information metric. The computational complexity of common optimization methods on $DP_n$ is rigorously analyzed, and empirical results confirm its superior performance in high-dimensional optimization problems, particularly in clustering, by leveraging its inherent geometric structure compared to general-purpose solvers.

> The Doubly Stochastic Multinomial Manifold (DPn) is an embedded manifold of the space of $n \times n$ matrices, $\mathbb{R}^{n \times n}$. It consists of square matrices with positive entries where each column and row sums to 1. The dimension of DPn is $(n-1)^2$
> .
> Here's how the tangent space of the DPn manifold can be described, how its projection looks, and how it can be computed:
> Description of the Tangent Space
> For any point $X$ within the Doubly Stochastic Multinomial Manifold (DPn), the tangent space, denoted as $T_X \text{DPn}$, is a linear space defined as the set of all $n \times n$ matrices $Z$ such that their row sums and column sums are all zeros
> . Formally: $T_X \text{DPn} = { Z \in \mathbb{R}^{n \times n} \mid Z\mathbf{1} = \mathbf{0}, Z^T\mathbf{1} = \mathbf{0} }$
> .
> This means that if you have a smooth curve $X(t)$ on the manifold such that $X(0) = X$, its derivative $\dot{X}(t)$ at $t=0$ will be a tangent vector $Z$ satisfying the condition that its row and column sums are zero
> . The dimension of this tangent space is $n^2 - (2n-1) = (n-1)^2$, which matches the dimension of the manifold itself
> .
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


### OT4P
[SUBTITLE: Moving into a better behaved space]

This research introduces OT4P, a novel method for relaxing permutation matrices onto the orthogonal group, offering potential benefits over existing techniques which often rely on the Birkhoff polytope. OT4P employs a temperature-controlled differentiable transformation that maps unconstrained vector space to the orthogonal group, where temperature influences the proximity of the resulting matrices to permutation matrices. This allows for gradient-based optimization of permutation-related problems and, with a re-parameterization trick, facilitates efficient stochastic optimization over latent permutations. The paper presents experimental results demonstrating OT4P's effectiveness in various tasks, including finding mode connectivity in neural networks, inferring neuron identities, and solving permutation synchronization problems.

> ## Overview of OT4P: A Gentle Exposition
> 
> ### Core Idea
> OT4P relaxes the discrete permutation matrix optimization problem by embedding permutation matrices into the continuous space of orthogonal matrices, which can then be parameterized using unconstrained vectors. This enables gradient-based optimization while maintaining key geometric properties.
> 
> ### The Two-Step Framework
> 
> **Step 1: Vector → Orthogonal Matrix**
> - Start with an unconstrained vector in ℝ^(n(n-1)/2) 
> - Transform it to a skew-symmetric matrix A (by computing A - A^T)
> - Apply matrix exponential to get an orthogonal matrix O = exp(A - A^T)
> 
> **Step 2: Moving Towards Permutations**
> - Find the closest permutation matrix P to O using the Hungarian algorithm
> - Move O towards P along a geodesic path, controlled by temperature τ
> - Result: Õ = P(P^T O)^τ, which approaches P as τ → 0
> 
> ### Addressing Your Specific Questions:
> 
> 1. **Special Orthogonal vs Orthogonal Group**: The paper primarily works in SO(n) (special orthogonal group with det=+1). The occasional mention of O(n) is to handle odd permutations (det=-1) through a simple trick using a diagonal matrix D.
> 
> 2. **Skew-Symmetric Transformation**: The mapping A → A - A^T creates skew-symmetric matrices. Yes, this gets all skew-symmetric matrices because any skew-symmetric S can be written as S = (S/2) - (S/2)^T.
> 
> 3. **Matrix Exponential Motivation**: The matrix exponential (Lie exponential) is the natural map from the Lie algebra so(n) (skew-symmetric matrices) to SO(n). It's locally bijective but not globally (similar to how e^x maps ℝ to ℝ^+).
> 
> 4. **Why A - A^T instead of just A**: The operation A - A^T ensures we get a skew-symmetric matrix, which is necessary for exp(·) to produce an orthogonal matrix. This is the composition of two maps: ℝ^(n(n-1)/2) → so(n) → SO(n).
> 
> 5. **Domain U**: This is the region where the mapping is injective (one-to-one). It ensures each orthogonal matrix has a unique representation, avoiding ambiguities in optimization.
> 
> 6. **SO(n) Contents**: SO(n) contains only orthogonal matrices with determinant +1. It's a subset of all orthogonal matrices O(n).
> 
> 7. **Geodesic Movement**: The key insight is that P^T O is near identity when O is near P. The power operation (·)^τ naturally interpolates along geodesics in SO(n). Multiplying by P shifts this interpolation from around I to around P.
> 
> 8. **Gradient Computation**: Exactly! You can now compute gradients with respect to the original vector parameterization and use standard optimizers.
> 
> ### Comparison with Birkhoff Polytope Methods (e.g., Sinkhorn)
> 
> **Key Advantages of OT4P:**
> 1. **Lower dimensional representation**: n(n-1)/2 for OT4P vs (n-1)² for Birkhoff polytope
> 2. **Preserves geometric structure**: Orthogonal matrices maintain inner products, useful for problems requiring geometric invariance
> 3. **No column-sum constraints**: Simpler optimization landscape compared to doubly-stochastic constraints
> 
> **Empirical Evidence:**
> - **Mode connectivity (Table 1)**: OT4P achieves 100% precision across all architectures, while Sinkhorn fails on VGG11 (63.08% precision)
> - **Neuron identity inference (Table 2)**: OT4P consistently finds optimal solutions even in challenging scenarios (5% known neurons), where Gumbel-Sinkhorn struggles
> - **Permutation synchronization (Figure 3)**: OT4P consistently outperforms across all object classes and problem sizes
> 
> ### Why Use OT4P for Approximate Symmetries?
> 
> 1. **Robustness**: The orthogonal group relaxation appears more stable than Birkhoff polytope, avoiding poor local minima
> 2. **Efficiency**: Lower dimensional search space means faster convergence
> 3. **Geometric preservation**: Critical for symmetry-finding applications where you need to preserve distances and angles
> 4. **Temperature control**: Allows smooth transition from continuous relaxation to discrete permutations
> 
> ### Practical Implementation Note
> The computational cost is dominated by eigendecomposition and Hungarian algorithm (both O(n³)). The paper provides efficient gradient computation through clever reparameterization, making it practical for moderate-sized problems.
> 
> This overview should provide a clear foundation for your thesis exposition while maintaining technical accuracy. The method's strength lies in its elegant use of differential geometry to solve a combinatorial problem, making it particularly suitable for applications like approximate symmetry finding where geometric properties matter.

> ## Deep Technical Analysis of OT4P
> 
> ### Step 1: Vector → Orthogonal Matrix (Detailed)
> 
> The first step creates a differentiable map from unconstrained vector space to SO(n):
> 
> **1.1 Vector Representation**
> - Start with a vector **a** ∈ ℝ^(n(n-1)/2)
> - This vector fills the upper triangular part of an n×n matrix A (with zeros on diagonal)
> - Example for n=3: [a₁, a₂, a₃] → A = [[0, a₁, a₂], [0, 0, a₃], [0, 0, 0]]
> 
> **1.2 Skew-Symmetrization**
> - Transform A to skew-symmetric: S = A - A^T
> - This creates S ∈ so(n), the Lie algebra of SO(n)
> - Key property: S^T = -S
> 
> **1.3 Matrix Exponential**
> - Apply matrix exponential: O = exp(S) = I + Σ(S^k/k!)
> - This maps so(n) → SO(n)
> - Result: O^T O = I and det(O) = +1
> 
> **Mathematical Properties:**
> - The mapping ϕ: ℝ^(n(n-1)/2) → SO(n) defined by ϕ(A) = exp(A - A^T) is:
>   - Differentiable everywhere
>   - Surjective (onto)
>   - Injective on domain U = {A | Im(λₖ(A - A^T)) ∈ (-π, π)}
> 
> ### Step 2: Moving Towards Permutations (Detailed)
> 
> **2.1 Finding Closest Permutation**
> - Given O ∈ SO(n), solve: P* = argmax_{P∈Pₙ} ⟨P, O⟩_F
> - This is a linear assignment problem solved by Hungarian algorithm
> - Cost: O(n³)
> 
> **2.2 Geodesic Interpolation**
> The key innovation is moving O towards P along the geodesic:
> 
> 1. **Push to neighborhood of I**: Compute P^T O (this is near I when O is near P)
> 2. **Take logarithm**: log(P^T O) maps to tangent space at I
> 3. **Scale by τ**: τ·log(P^T O) 
> 4. **Exponential map**: exp(τ·log(P^T O)) = (P^T O)^τ
> 5. **Pull back to P**: Õ = P(P^T O)^τ
> 
> **Why this works:**
> - When O ≈ P, then P^T O ≈ I
> - The power operation (·)^τ naturally interpolates eigenvalues
> - As τ → 0, Õ → P
> 
> ### Detailed Answers to Your Questions
> 
> **1. Special Orthogonal vs Orthogonal Group**
> 
> The paper primarily works in SO(n) but must handle O(n) for completeness:
> - **Even permutations**: det(P) = +1, so P ∈ SO(n) directly
> - **Odd permutations**: det(P) = -1, so P ∈ O(n) \ SO(n)
> 
> For odd permutations, they use a clever trick:
> - Find agent P̃ = PD where D = diag(1,...,1,-1)
> - Now P̃ ∈ SO(n) and ∥P̃ - P∥_F is minimized
> - Work with P̃ in SO(n), then multiply by D^T at the end
> 
> **2. Skew-Symmetric Transformation Details**
> 
> The mapping A → A - A^T is carefully chosen:
> - **Completeness**: Every skew-symmetric S can be written as S = B - B^T for some B
>   - Proof: Take B = S/2, then B - B^T = S/2 - S^T/2 = S/2 + S/2 = S
> - **Dimension matching**: Upper triangular has n(n-1)/2 free parameters, same as so(n)
> - **Isomorphism**: This creates a vector space isomorphism between ℝ^(n(n-1)/2) and so(n)
> 
> **3. Matrix Exponential (Lie Exponential) Deep Dive**
> 
> The exponential map exp: so(n) → SO(n) has profound geometric meaning:
> 
> - **Geometric interpretation**: It maps straight lines through 0 in so(n) to geodesics through I in SO(n)
> - **Local diffeomorphism**: Near 0, it's a bijection to a neighborhood of I
> - **Not globally bijective**: Multiple elements in so(n) can map to same O ∈ SO(n)
>   - Example: Rotations by θ and θ + 2π are the same
> 
> The Lie algebra so(n) is indeed the tangent space T_I SO(n) at identity. The exponential map:
> - Takes tangent vectors at I
> - Flows along geodesics for unit time
> - Lands on the manifold SO(n)
> 
> **4. Why ϕ(A) = exp(A - A^T) Instead of Just exp(A)**
> 
> This is subtle but crucial:
> - exp(A) only gives SO(n) if A is skew-symmetric
> - Starting with arbitrary A ∈ ℝ^(n×n) won't guarantee orthogonality
> - The operation A - A^T:
>   - Guarantees skew-symmetry: (A - A^T)^T = A^T - A = -(A - A^T)
>   - Maintains n(n-1)/2 degrees of freedom
>   - Creates a clean parameterization
> 
> **5. Domain U and Injectivity**
> 
> Domain U = {A ∈ ℝ^(n(n-1)/2) | Im(λₖ(A - A^T)) ∈ (-π, π)}:
> - **Injectivity** means one-to-one (each output has at most one input)
> - Outside U, multiple inputs map to same output (like angles θ and θ + 2π)
> - U is the "fundamental domain" where the exponential map is bijective
> - Practically: optimization stays in U most of the time
> 
> **6. Contents of SO(n)**
> 
> SO(n) contains exactly the n×n real matrices O such that:
> - O^T O = I (orthogonality)
> - det(O) = +1 (special)
> 
> Nothing else! It's a smooth manifold of dimension n(n-1)/2 embedded in ℝ^(n×n).
> 
> **7. Geodesic Movement - The Deep Geometry**
> 
> This is the most sophisticated part. Let me break it down:
> 
> **Why P^T O is near I when O is near P:**
> - If O = P + ε (small perturbation), then P^T O = P^T(P + ε) = I + P^T ε
> - Since P is orthogonal, ∥P^T ε∥ = ∥ε∥, so P^T O is I plus small perturbation
> 
> **How (·)^τ moves along geodesics:**
> - For orthogonal matrix M near I, M^τ interpolates eigenvalues
> - If M = QΛQ^T (eigendecomposition), then M^τ = QΛ^τ Q^T
> - This traces a geodesic from I (when τ=0) to M (when τ=1)
> 
> **The shifting mechanism:**
> - Working near I is computationally convenient (logarithm converges)
> - Left multiplication by P is an isometry in SO(n) with bi-invariant metric
> - So P·(geodesic from I to P^T O) = geodesic from P to O
> 
> **In manifold optimization context:**
> - This is geodesic interpolation on SO(n)
> - Temperature τ controls position along geodesic
> - As τ → 0, we approach the endpoint (permutation matrix)
> 
> **8. Gradient Computation Flow**
> 
> The complete chain:
> 1. **Forward**: a → A → exp(A - A^T) → O → P(P^T O)^τ → Õ
> 2. **Compute loss**: L = f(Õ)
> 3. **Backward**: ∂L/∂a via chain rule through all operations
> 
> Key insights for efficiency:
> - Eigendecomposition enables fast computation of (·)^τ and its gradient
> - They cache W_τ = ÕO^T in forward pass for efficient backward pass
> - All operations are differentiable except Hungarian (which is piecewise constant)
> 
> ### Why This Framework is Elegant
> 
> The beauty of OT4P lies in how it exploits the geometric structure:
> 
> 1. **Natural parameterization**: Uses Lie group structure rather than fighting constraints
> 2. **Geodesic interpolation**: Temperature control via natural geometry
> 3. **Bi-invariant metric**: Makes left/right translations isometries
> 4. **Dimension reduction**: Works in minimal dimensional space
> 
> The method essentially says: "Don't optimize over permutations directly. Instead, optimize over orthogonal matrices (which we can parameterize nicely) and project back to permutations when needed."

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

> You're off to a strong start in understanding the OT4P (Orthogonal Group-based Transformation for Permutation Relaxation) framework! Your understanding that OT4P considers permutations as being embedded in the Special Orthogonal Manifold $SO(n)$ is correct. You've also correctly identified that $SO(n)$ is a Lie group that possesses an exponential map, making it a "well-behaved" space for optimization. This underlying structure allows for the design of gradient-based optimization algorithms for problems involving permutations
> .
> Now, let's address your core question: how do you move closer to permutation matrices within this framework?
> The OT4P algorithm, proposed by Guo et al.
> , operates in two main steps to achieve this, which are summarized in Algorithm 1 of the paper
> :
> The OT4P Framework: Two Key Steps
> Step I: Mapping to an Orthogonal Matrix
> The first step is to map an unconstrained vector space $R^{n(n-1)/2}$ to the special orthogonal group $SO(n)$. This is done through a composition of two mappings:
> 1.
> From vector to skew-symmetric matrix: A vector a from the $R^{n(n-1)/2}$ space is transformed into a skew-symmetric matrix A (i.e., $A^T = -A$)
> . This skew-symmetric matrix A belongs to the Lie algebra $so(n)$
> .
> 2.
> From skew-symmetric to orthogonal matrix: The matrix exponential (Lie exponential) expm(·) is then applied to this skew-symmetric matrix $A - A^T$ to map it into the special orthogonal group $SO(n)$
> .
> ◦
> This combined mapping is denoted as $\phi: R^{n(n-1)/2} \to SO(n)$, where $\phi(A) = \text{expm}(A - A^T)$
> .
> ◦
> The mapping $\phi(·)$ is differentiable and surjective, meaning every orthogonal matrix in $SO(n)$ can be reached. It's also injective on a specific domain U
> .
> ◦
> Addressing Boundary Issues: The paper notes that permutation matrices with an eigenvalue of -1 (common for even-length cycles, e.g., for $n=3$, a single swap has a -1 eigenvalue) can lie on the boundary of this domain U
> . To mitigate this, a "shifting" operation is applied by left-multiplying the result of $\phi(A)$ with a random orthogonal matrix $B \in SO(n)$, i.e., $O = BO$. This effectively shifts the boundary of U to ensure that representations of most permutation matrices fall within the interior of U
> .
> Step II: Moving Closer to Permutation Matrices (The Core of Your Question)
> Once you have an orthogonal matrix $O \in SO(n)$ (obtained from Step I), the second crucial step is to move this $O$ towards the closest permutation matrix $P$ along the geodesic, controlled by a temperature parameter $\tau$
> . Here's how it works:
> 1.
> Finding the Closest Permutation Matrix: The first sub-step is to identify the permutation matrix $P \in P_n$ that is closest to the current orthogonal matrix $O$
> . This is formulated as a maximization problem: $\rho(O) := \text{argmax}_{P \in P_n} \langle P, O \rangle_F$
> .
> ◦
> This is a linear assignment problem, which can be efficiently solved using the Hungarian algorithm in cubic time
> . The paper explains how to handle potential negative values in $O$ for this algorithm by subtracting the minimum element
> .
> 2.
> Interpolation Towards $P$ Along the Geodesic: Once the closest permutation matrix $P$ is identified, the core idea is to interpolate the orthogonal matrix $O$ towards $P$
> . This interpolation is performed in a way that respects the manifold's geometry:
> ◦
> The paper utilizes the logarithm map at $P$ ($\text{logm}_P$) to map orthogonal matrices near $P$ to the tangent space $T_P SO(n)$
> . Similarly, an exponential map at $P$ ($\text{expm}_P$) maps elements from $T_P SO(n)$ back onto $SO(n)$ near $P$. These maps act as local inverses and allow for a "linear-like" movement in the tangent space
> .
> ◦
> The proposed transformation $\tilde{O}$ (the output of this step) is given by a specific formula that incorporates $O$, $P$, and the temperature $\tau$
> : $\tilde{O} = P (P^T O)^{\tau}$ [316, Eq. 10]
> ▪
> Role of $P^T O$: Since $P$ is the closest permutation matrix to $O$, $P^T O$ is an orthogonal matrix that is "close" to the identity matrix $I$ (as $P^T P = I$).
> ▪
> Role of $(\cdot)^{\tau}$: Raising $P^T O$ to the power of $\tau$ (where $\tau \in (0, 1]$) effectively moves it along the geodesic in $SO(n)$ towards the identity matrix $I$
> . The matrix power $A^\tau = \text{expm}(\tau \text{logm}(A))$ formally moves $A$ along a geodesic from $I$ towards $A$
> .
> ▪
> Role of the leading $P$: Multiplying by $P$ at the front ($P(\cdot)$) then shifts this result back from the neighborhood of $I$ to the neighborhood of $P$
> .
> ▪
> Temperature Control ($\tau$): The parameter $\tau$ is crucial for controlling how close $\tilde{O}$ gets to $P$
> .
> •
> If $\tau = 1$, then $\tilde{O} = P(P^T O)^1 = PP^T O = O$. The orthogonal matrix $O$ remains unchanged
> .
> •
> As $\tau \to 0^+$, the term $(P^T O)^{\tau}$ approaches $I$ (the identity matrix)
> . Therefore, $\tilde{O} = P \cdot I = P$. This means $\tilde{O}$ converges to the permutation matrix $P$
> .
> 3.
> Handling Odd Permutations: The formula $P(P^T O)^\tau$ works directly if $P$ is an "even" permutation (i.e., $P \in SO(n)$). For "odd" permutations, where $\text{det}(P) = -1$, the paper introduces an "agent" $P̂ = PD$ where $D = \text{diag}({1, \dots, 1, -1})$ to ensure $P̂ \in SO(n)$
> . The interpolation is then performed towards $P̂$, and the result is mapped back to the neighborhood of $P$ by right-multiplying by $D^T$. The final comprehensive mapping is $\psi_{\tau}: SO(n) \to M_P$, where $M_P$ is the manifold that "tightly wraps around the permutation matrices"
> .
> Optimization Process
> The overall OT4P algorithm takes an unconstrained vector $A$ in $R^{n(n-1)/2}$, transforms it into an orthogonal matrix $O$ via $\phi(A)$, and then moves $O$ towards its closest permutation matrix $P$ to obtain $\tilde{O}$ via $\psi_{\tau}(O)$
> . This composite mapping, $\psi_{\tau} \circ \phi$, effectively converts the challenging combinatorial optimization problem over permutation matrices into an unconstrained optimization problem in the Euclidean space $R^{n(n-1)/2}$
> .
> This means that during optimization, when you take a gradient step in the unconstrained $R^{n(n-1)/2}$ space, the OT4P transformation ensures that the resulting matrix remains on the orthogonal group and is biased towards a permutation matrix based on the temperature parameter $\tau$. Lower $\tau$ values encourage the solution to be very close to a discrete permutation matrix
> .
> In contrast to the doubly stochastic multinomial manifold $DP_n$ which has a dimension of $(n-1)^2$
> , the orthogonal group $SO(n)$ has a lower representation dimension of $n(n-1)/2$. This potentially leads to a smaller search space, one of the unique advantages highlighted by the authors.

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

- We use pytorch for automatic differentiation [TODO: Cite pytorch]

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
