### Literature Overview

  - Present the work of Liu, Vandenberghe, Matej Straka and Franta Szcepanik
    - Frantisek Szcepanik showed a combinatorial approach to this problem , we will explore the relaxation of this problem
      - [TODO(literature-review): Include results of the Frantisek Szcepanik thesis]

    - [TODO(literature-review): Figure what did Matej Straka do in his thesis and what did he find out]
    - [TODO(literature-review): Vandenberghe: say what was his contribution, the fastQAP algorith which we later used for finding approximate symeetries]


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
    >    - Only computes error changes rather than full objective function
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
