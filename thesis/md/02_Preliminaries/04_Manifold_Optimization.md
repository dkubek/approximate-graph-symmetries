## Manifold optimization

<!-- - [ ] Optimization and Structure
    - [ ] General Optimization Problem
    - [ ] Classical (un)constrained optimization vs. manifold optimization
    - [ ] Exploiting Geometry in optimization -->

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

> In essence, "Step 0 in optimization" is the universally applicable initial statement of the optimization problem: minimize a function over a set. The subsequent steps in algorithm development, whether for classical Euclidean space or for manifolds, involve leveraging the specific mathematical structure of that set $S$ (e.g., linear space, smooth manifold) and the function $f$ (e.g., differentiability) to define notions like gradients and Hessians, which are necessary to construct practical algorithms


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

<!-- - [ ] Manifold Definition -->

<!-- - [ ] Embedding Space
    - [ ] Definition -->

[TODO: Extract from Sources]

<!-- - [ ] Submanifold
    - [ ] Definition -->

[TODO: Extract from Sources]

<!-- - [ ] Embedded Submanifold
    - [ ] Definition -->

[TODO: Extract From sources]

<!-- - [ ] Smooth Manifold Definition through Local Defining Functions
    - [ ] Definition
    - NOTE: the definition will not be necessary for us and we will not use it -->

[TODO: Extract from sources]

> A smooth manifold (referred to simply as a "manifold" in the sources, as all discussions pertain to smooth ones which are $C^\infty$ or infinitely differentiable) is a set that, locally around each point, "looks linear" or "can be linearized"
> .
> More precisely, a manifold can be defined through local defining functions
> :
> •
> A subset $\mathcal{M}$ of a linear space $\mathcal{E}$ (e.g., $\mathbb{R}^d$) is an embedded submanifold of dimension $n < d$ if, for every point $x \in \mathcal{M}$, there exists a neighborhood $U$ of $x$ in $\mathcal{E}$ such that:
> ◦
> The intersection of $U$ and $\mathcal{M}$ can be expressed as: $\mathcal{U} \cap \mathcal{M} = {y \in U : h(y) = 0}$
> .
> ◦
> Here, $h: U \to \mathbb{R}^{d-n}$ is a smooth function
> .
> ◦
> The derivative of $h$ at $x$, denoted as $Dh(x)$, must have a rank of $d-n$
> . This condition ensures the "linearity" and the correct dimension of the manifold locally.



<!-- - [ ] Manifold Optimization in General Terms (not DS specific) -->

<!-- - [ ] Tangent Spaces
    - [ ] Definition -->

[TODO: Extract From sources; short introduction - motivation and illustrative informal explanation is key here]

<!-- - [ ] Riemannian Distances and Metrics
    - [ ] Riemannian Metric
        - [ ] Multiple Riemannian Metrics
            - for each manifold there can be multiple (just as for standard euclidean spaces) 
            - there is a natural metric inherited from the euclidean space
    - [ ] Riemannian Distance -->

[TODO: Extract Definitions From Sources; Very short introduction and definition, will not be expanded upon]

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

<!-- - [ ] Riemannian Gradient
    - [ ] Definition
    - [ ] Riemannian Gradient Computation  
        - in practice compute euclidean gradient and project it to tangent space -->

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

<!-- - [ ] Exponential problem 
    - [ ] Geodesic
        - [ ] Definition
        - [ ] Moving along geodesics
            - Time parameter, Distance along tangent vectors interpretation
        - [ ] Manifolds without known geodesic
            - NOTE: some manifold have nice geodesic (Lie groups connection)
    - [ ] Exponential map / Logarithmic map
        -  [ ] Definition of an exponential map / Log map
    - [ ] Exponential map is often difficult to find
        - ODE description and definition
    - [ ] Lie groups
        - [ ] Definition
        - Important: ! have explicit expression for the geodesic
        - the tangent space is "homogeneous" at all places
        - mention the historical origin of the name for "exponential map" -->

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

<!-- - [ ] Retractions
    - [ ] Definition
        - First order approximations to exponential maps
    - [ ] retractions vs true geodesics in practice
        - in practice retractions are often used instead of exponential map -->

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



<!-- - [ ] Practical Implementation  -->

<!-- - [ ] Steepest descent
    - [ ] Algorithm
        - Informal description of the algorithm
        - Formal definition of the algorithm (simple, schematic) -->

[TODO: Insert algorithm pseudocode from sources]


<!-- - [ ] Other algorithms
    - Adaptations of other algorithms are possible (require bit more advanced concept like riemannian hessians and vector/parallel transport)
        - Trust Region
        - Conjugate Gradients -->
  
[TODO: Extract short overview from sources that also other algorithms are possible, but requite more theory]

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

<!-- Conclusion  -->

> Manifold Optimization Overview
> Manifold optimization extends classical optimization techniques to problems where solutions must lie on a curved geometric space (a manifold) rather than in a flat Euclidean space. This is particularly useful in many applications where constraints naturally form manifolds, such as orthogonality constraints, fixed-rank matrices, or positive definite matrices.
> Key Differences from Classical Optimization
> The fundamental difference between classical optimization and manifold optimization is how we conceptualize the search space and movement through it. In manifold optimization:
> 
> Geometry matters: The manifold's curvature affects how we measure distances, define directions, and move along the surface.
> Tangent spaces: At each point on a manifold, we work with a local linearization called the tangent space, which approximates the manifold at that point.
> Retractions: After moving in the tangent space, we need to "project" back onto the manifold using operations called retractions.
> Riemannian metrics: These define how to measure distances and angles on the manifold, replacing the standard Euclidean inner product.
