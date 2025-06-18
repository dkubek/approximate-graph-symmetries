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
