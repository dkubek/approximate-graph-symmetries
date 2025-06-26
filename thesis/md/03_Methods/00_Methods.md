# Methods

- what follows here in this chapter is a description of the methods I was using for optimization of the approximate symmetry problem. The aim is not to provide a deep and and in depthexposition to the theory behind baic methods, but the general idea an neccessary framework to understand the concepts, implications , motivation and beuty of the individual methods. While we know that the explanations will be lacking in places, the we will make sure to poiint the reader to the correct resourcs that will opefully satiate her Neugierigkeit.

- the chapther will be organized as follows, firstly we present the QSA algorithm which is the starting point of this entire journey as well as the starting point of out rhinking process regarding the search for a better, more optimized, more robus, nmore efficient and in short better (for the lack of better term) algorithm. Next we think about the natural step whenm viewing the problem in the lens of constrained optimization and thatwe can utilize the robust and mature environment of constraint programming solvers. We utilize the InteriorPoint solver in hope of beating the performance of the simple Frank-Wolfe algorithm. Next, we try to incorporate the geometry of the space fo double stochastic matrices into aout problem .Until now, we have thought about the constraints as more or less black box feasible-space defining equations. The feasible requion tha of the Birkhoff polytope however has a a special structure and that is that of a smooth surface - manifold.We continue in the discussion of feasible reqion geometry and manifold optimization in the next chapter as well, where we consider a slightly different relaxation of the set of all permutations. We relax the permutations into the space of all orthogonal matrices using. Using the more efficient parametrization of matrices in this space we will be able to reduce the problem dimension and gain new isight on the problem of optimizing over permutations. Last but not least, we will try to reduce the dimension of the problem even further. Borrowing from the research of neural networks we will try to reduce the problem of optimizing over permutation matrices back to vectors through differentiable mappings.

### Sinkhorn-Knopp algorithm

- include discussion for the number of iterations of the sinkhorn algorithm
- inlcude discussion that the number of iterations decreases as we are closer to the original the space of DS matrices
    - this is the case in iterative gradient based algorithms where we move move a small distance along a set direction (does not move us far)
    - [ ] Mapping relaxed permutation (from Birkhoff polytope) back to a permutation

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

```

\section{Challenges Specific to Approximate Symmetries}

While approximate symmetries build upon the mathematical foundations of graph matching and quadratic assignment, they introduce unique challenges that distinguish them from general optimization problems. These challenges arise from the self-referential nature of symmetry detection—seeking structure within the same object rather than comparing distinct entities—and from the inherent ambiguity in defining what constitutes the ``best'' approximate symmetry.

\subsection{The Identity Problem and Fixed Points}

The most immediate challenge in approximate symmetry detection is the trivial solution problem. For any graph \(G\) with adjacency matrix \(\bm{A}\), the identity permutation \(\bm{I}_n\) always achieves \(E(\bm{A}, \bm{I}_n) = 0\), representing perfect symmetry. However, this solution provides no information about the graph's structural organization and fails to reveal meaningful symmetries.

\subsubsection{Excluding the Identity Solution}

Several approaches have been developed to address the identity problem:

\textbf{Hard Exclusion:} The most direct approach explicitly removes the identity from the feasible set, formulating the problem as:
\begin{equation}
\min_{\bm{P} \in \mathcal{P}_n \setminus \{\bm{I}_n\}} \|\bm{A} - \bm{P}\bm{A}\bm{P}^T\|_F
\end{equation}

While mathematically clean, this approach complicates optimization algorithms by introducing a non-convex constraint that is difficult to handle in continuous relaxations.

\textbf{Penalization Approaches:} A more practical alternative introduces penalty terms that discourage fixed points. For a penalty vector \(\bm{c} \in \mathbb{R}^n\), the penalized problem becomes:
\begin{equation}
\min_{\bm{P} \in \mathcal{P}_n} \left( -\tr(\bm{A} \bm{P} \bm{A}^T \bm{P}^T) + \tr(\text{diag}(\bm{c}) \bm{P}) \right)
\end{equation}

Setting \(c_i > 0\) penalizes the assignment of vertex \(i\) to itself, effectively discouraging the identity solution. This approach maintains the continuous structure needed for relaxation-based algorithms \cite{pidnebesna2024computing}.

\textbf{Fixed Point Constraints:} An intermediate approach allows at most \(K\) fixed points for some threshold \(K \geq 0\). This flexibility permits local symmetries while excluding the fully trivial solution. In practice, this approach proves particularly useful for networks that genuinely possess partial symmetries affecting only subsets of vertices.

\subsubsection{The Multiple Optima Challenge}

Even after excluding the identity, approximate symmetry problems often admit multiple optimal solutions. A graph may possess several distinct approximate symmetries with identical error values, each revealing different aspects of its structure. This multiplicity raises fundamental questions about solution interpretation and algorithm behavior.

Unlike standard optimization problems where any optimal solution suffices, approximate symmetry applications often require understanding the entire set of near-optimal solutions to gain complete insight into a network's structural organization.

\subsection{Well-definedness and Solution Quality}

The approximate symmetry problem suffers from an inherent ambiguity in defining solution quality. Two permutations with identical edge error counts may reveal vastly different structural insights, leading to questions about which solution better captures the graph's ``true'' symmetry.

\subsubsection{Local versus Global Symmetries}

Consider a network containing both strong local symmetries (e.g., symmetric subgraphs) and weak global symmetries (e.g., approximate overall balance). A permutation that preserves many local structures might achieve the same error count as one that preserves fewer local structures but captures a more global pattern.

This ambiguity reflects a fundamental tension between different notions of symmetry:
\begin{itemize}
\item \textbf{Structural Symmetry:} Permutations that preserve large connected substructures
\item \textbf{Statistical Symmetry:} Permutations that balance global network properties
\item \textbf{Geometric Symmetry:} Permutations reflecting spatial or hierarchical organization
\end{itemize}

No single metric can simultaneously optimize all these criteria, necessitating problem-specific choices about which aspects of symmetry to emphasize.

\subsubsection{Solution Stability and Initialization Dependence}

Optimization algorithms for approximate symmetry detection often exhibit strong sensitivity to initialization. Small changes in starting points can lead to dramatically different solutions, even when the objective function values are nearly identical. This instability complicates both algorithm design and solution interpretation.

The indefinite nature of the relaxed QAP formulation exacerbates this sensitivity, as the optimization landscape typically contains numerous local optima. Understanding and controlling this initialization dependence remains an active area of research \cite{pidnebesna2024computing}.

\subsection{Mapping Back to Permutations}

Continuous relaxation approaches produce doubly stochastic matrices that must be projected onto the set of permutation matrices to obtain discrete solutions. This projection step introduces its own challenges and can significantly impact solution quality.

\subsubsection{The Hungarian Algorithm for Projection}

The standard approach for projecting a doubly stochastic matrix \(\bm{D}\) onto the closest permutation matrix solves:
\begin{equation}
\max_{\bm{P} \in \mathcal{P}_n} \tr(\bm{D}^T \bm{P})
\end{equation}

This linear assignment problem can be solved efficiently using the Hungarian algorithm in \(O(n^3)\) time. The resulting permutation matrix minimizes the Frobenius distance \(\|\bm{D} - \bm{P}\|_F\) among all permutation matrices.

\subsubsection{The Rearrangement Inequality and Vector Projections}

When dealing with permutation vectors rather than matrices, the projection problem takes a different form. Given a real vector \(\bm{v} \in \mathbb{R}^n\), the closest permutation vector under the \(\ell_2\) norm is determined by the rearrangement inequality.

\begin{theorem}[Rearrangement Inequality]
For sequences \(a_1 \leq a_2 \leq \ldots \leq a_n\) and \(b_1 \leq b_2 \leq \ldots \leq b_n\), the sum \(\sum_{i=1}^n a_i b_{\pi(i)}\) is maximized when \(\pi\) is the identity permutation and minimized when \(\pi(i) = n+1-i\).
\end{theorem}

This theorem implies that the permutation closest to a given vector \(\bm{v}\) is simply the argsort of \(\bm{v}\)—the permutation that sorts the vector components. This provides an \(O(n \log n)\) algorithm for vector-based permutation projection.

\subsubsection{Birkhoff-von Neumann Decomposition}

An alternative approach to projection utilizes the Birkhoff-von Neumann decomposition to express doubly stochastic matrices as convex combinations of permutation matrices. This decomposition provides access to multiple permutation candidates, enabling more sophisticated selection criteria beyond simple projection.

However, the decomposition approach typically requires evaluating numerous permutation matrices, making it computationally expensive for large problems. Practical implementations often use randomized sampling or truncated decompositions to balance solution quality with computational efficiency.

\subsection{Algorithmic Implications}

These challenges collectively shape the design of practical approximate symmetry algorithms. Successful approaches must balance several competing objectives:

\begin{itemize}
\item Avoiding trivial solutions while maintaining optimization tractability
\item Providing stable, reproducible results despite initialization sensitivity
\item Handling solution multiplicity in an interpretable manner
\item Efficiently projecting continuous solutions to discrete permutations
\end{itemize}

The interplay of these challenges distinguishes approximate symmetry detection from other combinatorial optimization problems and continues to drive algorithmic innovation in this active research area. Understanding these challenges is essential for both developing new algorithms and interpreting the results of existing methods.

\section{Conclusion}
```
