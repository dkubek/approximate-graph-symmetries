# Approximate Symmetry Problem

[DESC: SHORT CHAPTER INTRODUCTION -> it is not short currently]

<!-- [INSERT: Personal Artistic Intro to Notation] -->

- Goal: Present the problem of approximate symmetries from a different point of view
  - [DESC: WHat is my motivation?] THe main motivation for the sutdy of approximate symmetries comes from the fact that the problems can arrise in real world situations and the strictness and the exactness of the original isomorphism problem cannot capture or rather does not satisgyingly capture the approximativeness of the the problem and the inexactness of the whole matter. What DO I mean by this. The setup is fairly simple./. Imagine we are interested in studying the symmetries of the graph. The reasons why we would want to do this come later [TODO: Introduce the reasons why this is important first of interesting first and provide e tiwht the reasons later]. WT We start with a graph. This graph can be symmetric. The problem of symmetry is solved by the automorphism problem. Automorphism are a solved problem and the enttire automorphism group can be already computed [TODO: include sources that describe the problems of solving automorhphisma dnd automorphism groups; do not forget to include the computational complexity of the problem and the software packages that can be used to solve the problem.]. The exact symmetry is nice, however do consider this simple example (provided by LIu). COnsider a square n.n grid. This grid does have many symmetries (rotational, diagonal, reflexion), However, COnsider the case of removing one edge from the grid. Now all the symmetries are broken and the graph possesis only a single automorphism (namely the identity).  However, looking at the grid without one edge, we do get the distinct feeling that despite the missing edge, it is stile _quite symmetric_. WFollowing this idea, Liu suggest that we can extend the notion of symmetry using the error metrixc E(A,P), or edge count \mathcal{E}(A, P), lea, leading to the normalized symmetry measure S(A).This metric extends the notion of symmetry as symmetriies are the solutions attaing the 0 value, other permutations lead to a solution attaining positive values and we can interpret the smaller values as solutions (or permutations) closer to some symmetry.

  - [TODO(motivation): why is it cool to study approximate symmetries, symmetry of brain, the symmetry (or lack thereof) can provide us with insights about the sate of degradation of the brain]


  - Present the work of Liu, Vandenberghe, Matej Straka and Franta Szcepanik
    - FRantisek Szcepanik showed a combinatorial approach to this problem , we will explore the relaxation of this problem
      - [TODO(literature-review): Include results of the Frantisek Szcepanik thesis]

    - [TODO(literature-review): Figure what did Matej Straka do in his thesis and what did he find out]
    - [TODO(literature-review): Vandenberghe: say what was his contribution, the fastQAP algorith which we later used for finding approximate symeetries]

  - intention to look at the problem from a different angle
    - Liu suggests in his thesis to solve this problem from a combinatorial perspective using local search through the algorithm of simmulated annealing. 

  - introduce the problem in relation to different known studied problems and how the problem of approximate symmetries naturally arises when following the logic of these problems
    - show how these problems differ and how do they relate

## Problem overview

(and literature overview)

- introduce the problem of graph symmetries (Why? Oh woe why?)

- quick overview (setup, playground, reiteration)
- goal is to look for approximate symmetries in networks that do not necessarily have one in a _strict_ sense

- introduce the measure proposed by approximate symmetries proposed by Liu
  - what does it represent?
  - normalized appx symmetry coefficient
  - problems and disadvantages with this definition (it has obvious optimum)

### Literature Overview

- How Liu suggest to solve this problem? -> Simmulated Annealing
  - Work of Franta Szcepanik
    - [TODO: Include and analyse the results of the work of Franta Sczepanik]
      > This thesis addresses the fundamental question of whether graph centralities can guide simulated annealing to find better approximate symmetries in complex networks, motivated by the observation that traditional exact symmetry measures are too "fragile" for real-world applications. The primary aim was to enhance simulated annealing algorithms by incorporating graph centralities (such as PageRank, eigenvector centrality, and betweenness) to preferentially align vertices with similar structural properties, based on the theoretical insight that automorphisms preserve centrality values. The results demonstrate that this centrality-guided approach significantly improves symmetry detection on structured networks: betweenness centrality yields substantial improvements on grid graphs, eigenvector centrality and PageRank perform well on scale-free (Barabási-Albert) and protein-protein interaction (Duplication-Divergence) networks respectively, while no improvements occur on completely random (Erdős-Rényi) networks as expected. Notably, the performance gains become more pronounced with increasing graph size, with the guided algorithm reducing mismatched edges by approximately 71 on average for 500-node networks, suggesting that the method scales well and validates the hypothesis that incorporating domain-specific knowledge into metaheuristic optimization can yield substantial practical benefits.


    > # Analysis of "Centralities in Computation of Approximate Symmetries"

    > ## Overview and Research Question

    > The thesis addresses a fundamental limitation in traditional graph symmetry detection: exact automorphisms are "fragile" - even minor changes (like removing a single edge) can eliminate all symmetries. The main research question solved: **Can graph centralities guide simulated annealing to find better approximate symmetries in complex networks?**

    > ## Key Findings and Contributions

    > ### 1. **Dynamic Fixed Points Approach (DFP-SA)**
    > - **Innovation**: Instead of arbitrarily limiting fixed points to 0 or n/2, the algorithm dynamically determines the optimal number by penalizing them in the objective function
    > - **Result**: Performs comparably to parametrized versions while eliminating arbitrary parameter selection
    > - **Technical detail**: Modified objective function SF(A) that reduces the denominator by pairs of fixed points

    > ### 2. **Centrality-Guided Simulated Annealing**
    > - **Core contribution**: Re-implemented the move function to probabilistically select transpositions that align vertices with similar centrality values
    > - **Key insight**: Since automorphisms preserve centralities, vertices with similar centrality values should be mapped to each other
    > - **Performance gains**:
    >   - **Grid graphs**: Substantial improvements, especially with betweenness centrality
    >   - **BA model**: Statistically significant improvements with eigenvector centrality (effect size increases with graph size)
    >   - **DD model**: PageRank guidance yields best results, particularly for larger graphs
    >   - **ER model**: No improvements (as expected - completely random structure)

    > ### 3. **Scaling Effects**
    > - **Critical finding**: Improvements become more pronounced with increasing graph size
    > - **BA model (n=500)**: Eigenvector-guided annealing reduces mismatched edges by ~71 on average
    > - **Implication**: The method scales well and becomes more effective for larger networks

    > ## Optimization Algorithm Details

    > ### Algorithm Modifications:
    > 1. **Similarity Matrix Construction**:
    >    - Computes centrality differences between all vertex pairs
    >    - Uses two parameters (β for division constant, φ for probability smoothing)
    >    - Parameters optimized via grid search on ~400 random graphs

    > 2. **Move Function Enhancement**:
    >    - Time complexity remains O(n) per move
    >    - Probabilistically favors swaps that increase similarity alignment
    >    - Maintains the Metropolis acceptance criterion for escaping local minima

    > 3. **Computational Optimizations**:
    >    - Exploits sparsity of real networks
    >    - Only computes energy changes rather than full objective function
    >    - Achieves practical performance on graphs up to 500 nodes

    > ## Experimental Results Summary

    > | Network Model | Best Centrality | Improvement Level | Key Insight |
    > |--------------|-----------------|-------------------|-------------|
    > | Grid | Betweenness | High | More unique centrality values enable precise alignment |
    > | Erdős-Rényi | None | None | Random structure offers no guidance |
    > | Barabási-Albert | Eigenvector | Moderate-High | Improvements increase with sparsity and size |
    > | Duplication-Divergence | PageRank | Moderate-High | Effective for tree-like sparse structures |

    > ## Limitations and Reviewer Critiques

    > ### Technical Issues:
    > 1. **Mathematical imprecisions**: 
    >    - Incorrect SF(A) definition (denominator issues)
    >    - DD model description errors
    >    - Minor notation inconsistencies

    > 2. **Algorithm limitations**:
    >    - Gradient descent variant computationally infeasible for n>100
    >    - Limited success in returning to known optimal solutions (LRM experiments)
    >    - Underperforms compared to quadratic assignment relaxation methods

    > ### Methodological Constraints:
    > - Parameters (β, φ) may not be globally optimal
    > - Limited to undirected, unweighted graphs
    > - Clustering coefficient consistently underperforms (due to local scope)

    > ## Practical Applications

    > The thesis mentions several application domains:

    > 1. **Brain Networks**: Understanding synchronized neural clusters and collective behavior
    > 2. **Internet/Web Structure**: Network topology analysis
    > 3. **Public Transportation Networks**: System organization patterns
    > 4. **Social Networks**: Identifying influential individuals and information spread
    > 5. **Protein-Protein Interaction Networks**: Understanding biological system evolution
    > 6. **Citation Networks**: Academic impact analysis

    > ## Future Research Directions

    > 1. **Algorithm Extensions**:
    >    - Test on other metaheuristics (evolutionary algorithms, particle swarm)
    >    - Explore convex combinations of centralities
    >    - Extend to directed and weighted graphs

    > 2. **Theoretical Analysis**:
    >    - Understand why improvements increase with graph size
    >    - Analyze orbit structures of computed permutations
    >    - Formal complexity analysis of the guided approach

    > 3. **Practical Applications**:
    >    - Validation on real-world datasets (especially brain connectomes)
    >    - Comparison with state-of-the-art symmetry detection methods
    >    - Integration with existing network analysis tools

    > ## Conclusion for Optimization Research

    > This work successfully demonstrates that domain knowledge (graph centralities) can effectively guide traditional optimization algorithms (simulated annealing) to achieve better results. The approach is particularly valuable for structured networks where the improvements are most significant. For optimization researchers, this suggests that incorporating problem-specific invariants or characteristics into metaheuristic search can yield substantial benefits, especially as problem size increases.

- [TODO: Include and analyse the results of Straka]
    > This thesis addresses the fundamental challenge of detecting and measuring symmetries in complex networks where perfect symmetries (automorphisms) rarely exist, asking how we can effectively quantify approximate symmetrical properties and which algorithms best identify them. The primary aim was to develop and optimize computational methods for finding approximate symmetries—specifically modifying simulated annealing (SA) to exploit network sparsity and reduce complexity from O(n³) to O(n log n), and adapting the Fast Approximate QAP (FAQ) algorithm to avoid trivial solutions with many fixed points. The research demonstrates that modified FAQ consistently finds better approximate symmetries than SA for smaller networks, while SA scales more efficiently for larger networks; reveals complex, non-monotonic relationships between symmetry and other network measures (modularity, clustering coefficient) across various random network models; and shows that real brain connectomes exhibit significantly higher symmetry (S ≈ 0.09) than comparable random networks (S ≈ 0.35), with left-right hemisphere correspondence as the dominant symmetrical pattern. The work successfully bridges the gap between theoretical perfect symmetries and practical approximate symmetries in real-world networks, providing both methodological tools and empirical insights into hidden symmetrical properties of complex systems.

- [TODO: Include and analyse the results of Hartman, Pidnebesna]

### Graph Isomorphism

- describe the problem of graph isomorphism
- present the combinatorial nature of this problem
  - present how it compares to the problem
- [TODO(problem): Include formal problem description]
- [TODO(search, literature-review): include the recent result that graph isomorphism is quasi-polynomial]
- [TODO: Research? How is the quasipoly result achieved? Are any relaxations involved?]

  **1. Babai's Original Paper:**
  László Babai, "Graph Isomorphism in Quasipolynomial Time," arXiv:1512.03547 (2015) - This is the foundational paper announcing the breakthrough showing that Graph Isomorphism can be solved in quasipolynomial ($\exp((\log n)^{O(1)})$) time, improving upon the previous best bound of $\exp(O(\sqrt{n\log n}))$ from Luks (1983).
  https://arxiv.org/abs/1512.03547

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
  - binary QP + LC => NP hard to find global optimum

```latex
\subsection*{Problem Formulation}

The classical Quadratic Assignment Problem (QAP) seeks to find a permutation matrix $\bm{P} \in \setP$ (where $\setP$ is the set of $n \times n$ permutation matrices) that minimizes an objective function, often of the form $f(\bm{P}) = \tr(\bm{A} \bm{P} \bm{B}^T \bm{P}^T)$, where $\bm{A}$ and $\bm{B}$ are given square matrices \cite{Koopmans1957}. This formulation is central to problems like graph matching, where $\bm{A}$ and $\bm{B}$ might represent adjacency matrices of two graphs, and the objective is equivalent to minimizing $-\tr(\bm{A} \bm{P} \bm{B}^T \bm{P}^T)$ to maximize edge overlap \cite{Vogelstein2014}.

The QSA algorithm addresses a related but distinct objective function:
\begin{equation} \label{eq:qsa_objective}
f(\bm{P}) = -\tr(\bm{A} \bm{P} \bm{A} \bm{P}^T) + \tr(\text{diag}(\bm{c}) \bm{P})
\end{equation}
where $\bm{A}$ is an $n \times n$ symmetric matrix (i.e., $\bm{A} = \bm{A}^T$), $\bm{P}$ is the $n \times n}$ matrix variable, and $\bm{c}$ is an $n$-dimensional vector. The first term, $-\tr(\bm{A} \bm{P} \bm{A} \bm{P}^T)$, captures a quadratic interaction related to structural alignment, similar to the QAP but using the matrix $\bm{A}$ in both roles. The second term, $\tr(\text{diag}(\bm{c}) \bm{P}) = \sum_{i=1}^n c_i P_{ii}$, serves as a linear penalty or reward for the diagonal elements of $\bm{P}$. This term can be used, for example, to penalize ($c_i < 0$) or encourage ($c_i > 0$) fixed points in the assignment represented by $\bm{P}$.
```

#### Linear Assignment Problem

- [TODO(problem) : Include formal definition of the problem (latex template)]

- present the problem, show that it can be solved in poly time using LP, moreover using hungarian algorithm
or Linear Assignment Problem solver using Jonker-Volgenant algorithm
  - [TODO: find paper for Jonker-Volgenant
        A shortest augmenting path algorithm for dense and sparse linear assignment problems
    ]
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
