# Introduction

[
    TODO(task): Write introduction at the very end to say how awesome and important everything I am doing is;
    DEPENDSON: All other chapters being finished;
]

# Preliminaries

[GOAL: ??? ]

[INSERT(write): Personal Artistic Intro to Preliminaries]

[
    TODO(write): The overview of what will be presented in this chapter;
    DEPENDSON: All chapters in Preliminaries being finished
]

## Notation

[GOAL: ??? ]

[INSERT(write): Personal Artistic Intro to Notation]

[
    TODO(task): At the very end, collect all the notation from all the text and compile it here
    DEPENDSON: All chapters in Preliminaries being finished.
]
    * [TODO(task): Unify the notation of all chapters before compiling the Notations chapter]

[INSERT(write): Personal Artistic Segway to the Approximate Symmetry Problem Chapter]

## Problem Overview

[GOAL: ???]

[INSERT: Personal Artistic Intro to Notation]
    - Present the problem in relation to different problems
    - make the problem emerge as a natural consequence

[INSERT: Factual overview of what is contained in this chapter]

[!TODO(writeup): Compile all points I want to include in this chapter] 
    => [TODO(organize): Organize this information into coherent blocks] 
    => [TODO(story): Create a story from this information] 
{
- General Overview and Definitions of the Problems

- [ ] Graph Isomorphism (GI)
- [ ] Graph Matching Problem (GMP)
- [ ] Quadratic Assignment Problem (QAP)
- [ ] Approximate Symmetry Problem (ASP)
    - [ ] Introduce Approximate Symmetry Metrics (relate to GMP)
        - Introduced by Liu
        - [ ] E(A, P)
            - [ ] Informal motivation for the metric
            - [ ] Formal Definition
        - [ ] \mathcal{E}(E, P)
            - [ ] Informal motivation for the metric
            - [ ] Formal Definition
        - [ ] S(A)
            - [ ] Informal motivation for the metric
            - [ ] Formal definition
    - [ ] Definition of AS
    - [ ] Fragility of Symmetries
        - Idea that a small change can result in destruction of great number of symmetries
        - Introduce the fragility of approximate symmetries on an example
            - Square grid example
    - [ ] Relax
      - [ ] Definition of Doubly Stochastic Matrix
      - [ ] Birkhoff polytope
        - [ ] Definition
        - [ ] Cite original Birkhoff result
      - Why relax QAP instead of GMP
        - Relax at your own risk paper.

- [ ] Linear Assignment Problem (LAP)
    - [ ] Informal Problem Motivation
    - [ ] Formal Problem definition
    - [ ] Polynomial complexity of the problem with efficient algorithms
        - [ ] Hungarian Algorithm
        - [ ] Jonker 1987

- [ ] Representing Permutations
    - [ ] Definition
        - Formal definition of a permutation
            - as a function (\pi)
                - [ ] Formal definition
    - overview ways of different how permutation can be represented
        - NOTE: This will be usefull when modeling
        - as a vector (dimension n)
            - explicitly (permutation of {1, 2, ..., n}) (dimension n - 1 if we are smart)
            - implicitly (argsort of a vector from R^n s.t. no two elements are equal) (dimension n)
        - as a permutation matrix 
            - [ ] Formal definition
    - Honorable mention: Through Sorting Networks, as an extension polytope of permutahedron(n log n)
        - this is relaxed way
        - put this in a footnote since I am not using it anywhere in the thesis but I do find it a neat result


- [ ] Problems Specific to Approximate Symmetries
    - [ ] Not Well-definededness of the Problem
        - If we find a 2 vertices of degree 1 with a common neighbour, then switching them presents already a symmetry
        - Is this the kind of symmetry we want to find => It is quite local
        - Maybe we want a more kind of global symmetry (i.e. such that will move around many vertices)
            - we can do this by penalizing fixed points
        - However, ideally, we want both and have a way to choose
        - More generaly we might want a way or a description of how symmetries of a graph look if we allow k-fixed points

}


[INSERT: Personal Artistic Segway to Interior Point]

## Interior Point Optimization

[GOAL: ???]

[INSERT: Personal Artistic Intro to Interior Point Optimization]
    - maturity of the method
    - main source is Nocedal; look there for exposition
    - interior point methods gained popularity after popularity in LP
    - interior point naming

[INSERT: Factual overview of what is contained in this chapter]

[!TODO(writeup): Compile all points I want to include in this chapter] 
    => [TODO(organize): Organize this information into coherent blocks] 
    => [TODO(story): Create a story from this information] 
{
  - [ ] Problem Setup
    - [ ] Definition
    - include basic definitions and assumptions about the problem

  - [ ] Algorithm Idea
    - describe the main idea of the algorithm informally in a paragraph, at this point in such a way as not to require any prerequisite knowledge and definitions
    - Interior Point Framework

  - [ ] Barrier approach
    - Barrier function optimization
    - [ ] Formal mathematical description

  - [ ] Central Path
    - [ ] Definition (formal)
        - what is the central path formally
    - Informal description

  - [ ] KKT System
    - [ ] Theorem (KKT)
    - [ ] Derivation of the KKT System

  - [ ] Primal-Dual Interior Point Method
    - [ ] Basic idea (informal)
    - [ ] Formal definition

  - [ ] Solving the KKT system
    - [ ] Newton's Method for the KKT System
    - [ ] Note about hidden dragons and problems with this system and how much of a problem and pain it is to solve it quickly or approximately


}

[INSERT: Personal Artistic Segway to Manifold Optimization]

## Manifold Optimization

[INSERT: Personal Artistic Intro to Manifold optimization]
    - we can decrease the dimension of the problem

[!TODO(writeup): Compile all points I want to include in this chapter] 
    => [TODO(organize): Organize this information into coherent blocks] 
    => [TODO(story): Create a story from this information] 
{

[
    GOALS:
        - provide an introduction to manifold optimization to someone who is already well-versed in standard optimization techniques
        - present the parallels and differences of classical constrained optimization and manifold optimization
        - provide necessary background information, motivation, concepts and definition enough to understand how the steepest descent algorithm on manifolds works
        - introduce the technical and computational challenges like moving along the geodesic in complex manifolds
        - introduce practical optimization approaches and in used in practice when the exact description is not available (for example cannot compute the geodesic => we have to make do with retractions)
]

- [ ] Optimization and Structure
    - [ ] General Optimization Problem
    - [ ] Classical (un)constrained optimization vs. manifold optimization
    - [ ] Exploiting Geometry in optimization


- [ ] Manifold Definition

    - [ ] Embedding Space
        - [ ] Definition

    - [ ] Submanifold
        - [ ] Definition

    - [ ] Embedded Submanifold
        - [ ] Definition

    - [ ] Smooth Manifold Definition through Local Defining Functions
        - [ ] Definition
        - NOTE: the definition will not be necessary for us and we will not use it


- [ ] Manifold Optimization in General Terms (not DS specific)

    - [ ] Tangent Spaces
        - [ ] Definition

    - [ ] Riemannian Distances and Metrics
        - [ ] Riemannian Metric
            - [ ] Multiple Riemannian Metrics
                - for each manifold there can be multiple (just as for standard euclidean spaces) 
                - there is a natural metric inherited from the euclidean space
        - [ ] Riemannian Distance

    - [ ] Riemannian Gradient
        - [ ] Definition
        - [ ] Riemannian Gradient Computation  
            - in practice compute euclidean gradient and project it to tangent space

    - [ ] Exponential problem 
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
            - mention the historical origin of the name for "exponential map"

    - [ ] Retractions
        - [ ] Definition
            - First order approximations to exponential maps
        - [ ] retractions vs true geodesics in practice
            - in practice retractions are often used instead of exponential map


- [ ] Practical Implementation 

  - [ ] Steepest descent
    - [ ] Algorithm
        - Informal description of the algorithm
        - Formal definition of the algorithm (simple, schematic)

  - [ ] Other algorithms
    - Adaptations of other algorithms are possible (require bit more advanced concept like riemannian hessians and vector/parallel transport)
        - Trust Region
        - Conjugate Gradients

}

[INSERT: Personal Artistic Segway to Manifold Optimization]

## Literature Review

[INSERT: Personal Artistic Intro to Methods]

[
    INSERT: Factual overview of all things contained in this chapter;
    DEPENDSON: All the information in this chapter is done
]

{
- [ ] Literature Review
    - Present work fo Liu, Vandenberghe, Matej Straka, Franta Sczepanik, Pidnebesna, Hartmann
    - [ ] Liu => Franta Sczepanik
        - [ ] Liu
        - [ ] Franta Sczepanik
            - Looks on the problem through combinatorial optimization using local search through the Simulated Annealing algorithm and tries to find more fitting permutations by using descriptors (? Is descriptor the right word) of complex networks
    - [ ] Vandenberghe => Matej Straka => David Hartman, Pidnebesna
        - [ ] Vandenberge fQAP algorithm original algorithm and method to solve relaxed approximate symmetries
        - [ ] Matej Straka
          - looked at improvements of the Liu SA, and Vandenberge fQAP algorithms
        - [ ] Pidnebesna, Hartman
          - QSA algorithm, compiled results on ER graphs, BA, LRM
}

## Summary

[INSERT: Summary of all the things done in this chapter]

[INSERT: Personal Artistic Segway to Methods]


# Methods

[GOAL: ??? ]

[INSERT: Personal Artistic Intro to Methods]

[
    INSERT: Factual overview of all things contained in this chapter (chapter organization);
    DEPENDSON: All the information in this chapter is done
]


- [ ] Optimization Problems specific to AS
    - [ ] Excluding Identity
    - [ ] Choosing a Starting Point
        - [ ] Sinkhorn-Knopp algorithm
            - [TODO: Include landmark paper and description]

[INSERT: Personal Artistic Segway to QSA]

## QSA
[GOAL: ??? ]

[INSERT: Personal Artistic Intro to Methods]

[
    INSERT: Factual overview of all things contained in this chapter;
    DEPENDSON: All the information in this chapter is done
]

[!TODO(writeup): Compile all points I want to include in this chapter] 
    => [TODO(organize): Organize this information into coherent blocks] 
    => [TODO(story): Create a story from this information] 
{
 - [ ] QSA

- [ ] We use analytical gradient computation

}

[INSERT: Personal Artistic Segway to Methods]

## Interior Point Method
[GOAL: ??? ]

[INSERT: Personal Artistic Intro to Methods]

[
    INSERT: Factual overview of all things contained in this chapter;
    DEPENDSON: All the information in this chapter is done
]

[!TODO(writeup): Compile all points I want to include in this chapter] 
    => [TODO(organize): Organize this information into coherent blocks] 
    => [TODO(story): Create a story from this information] 
{
 - [ ] Interior Point Method
  - [ ] IP for Approximate Symmetries
    - IP for quadratic systems
    - Special Case : Nonconvex Quadratic Problems
    - Practical Implementation Considerations
    - IPOPT Optimizer
        - Solver specific optimization techniques that make corrections, improvements, special tricks, strategies and approximations to arrive at better solutions and to be more robust and more quick
        - It is a line-search interior point method
          - [TODO: In a sentence what does this mean?]
    - [ ] Handling Nonconvexity
        - IPOPT's Approach

 - [ ] we use analytical gradient computation
}

[INSERT: Personal Artistic Segway to Methods]

## Manifold

[GOAL:
    - Focus of this chapter is to present the manifold optimization based on the already build groundwork in the Preliminaries chapter
    - the focus is to present the specific context of doubly stochastic matrices and the specific definitions for them
    - focus on specific forms of concepts like manifold, tangent space, projection and riemannian gradient for the DS Manifold
    - Show how to reformulate the original approximate symmetries problem in context of DS Manifold Optimization
    - Present the concrete approach we use
        - Using the python Manopt library with the steepest descent algorithm

 ]

[INSERT: Personal Artistic Intro to Methods]

[
    INSERT: Factual overview of all things contained in this chapter;
    DEPENDSON: All the information in this chapter is done
]

[!TODO(writeup): Compile all points I want to include in this chapter] 
    => [TODO(organize): Organize this information into coherent blocks] 
    => [TODO(story): Create a story from this information] 
{
   - [ ] The Doubly Stochastic Manifold
    - [ ] Manifold - Concrete form for DS
    - [ ] Tangent Space - Concrete form for DS
    - [ ] Projection and Retraction - Concrete form for DS (Sinkhorn projection)
    - [ ] Riemannian Gradient - Concrete form for DS (using projection)
  - [ ] Algorithm Used
    - [ ] Steepest descent
        - [ ] Include complexity and empirical performance of the approach (from paper)
  - [ ] Manifold optimization for approximate symmetries
    - [ ] Manopt (Library for optimization on manifolds)
        -  pytorch for automatic differentiation
    - Describe how exactly we adapt the manifold optimization framework for the problem of approximate symmetries

 - [  ] note AutoDiff using pytorch is being used
}

[INSERT: Personal Artistic Segway to Methods]

## Orthogonal Relaxation

[GOAL: ??? ]

[INSERT: Personal Artistic Intro to OT4P]

[
    INSERT: Factual overview of all things contained in this chapter;
    DEPENDSON: All the information in this chapter is done
]

[!TODO(writeup): Compile all points I want to include in this chapter] 
    => [TODO(organize): Organize this information into coherent blocks] 
    => [TODO(story): Create a story from this information] 
{

 - [ ] Present results of the OT4P paper
 - [ ] Core Idea
 - [ ] Definitions
    - [ ] Skew-Symmetric Matrices
    - [ ] Matrix Exponential, Matrix logarithm
    - [ ] Lie group
    - [ ] Special Orthogonal, Orthogonal Group
 - [ ] Two-Step Framework
    - [ ] The two step framework summary
    - [ ] Step 1
        - [ ] Vector Representation
        - [ ] Mathematical Properties
    - [ ] Step 2
        - [ ] Finding closest permutation
        - [ ] Geodesic Interpolation
    - [ ] Additional considerations
 - [ ] Birkhoff polytope vs. Orthogonal Rleaxation approach
 - [ ] Advantages and comparison to Birkhoff
 - [ ] Implementation issues
    - [ ] Efficiency

 - [ ] Note AutoDiff using pytorch is being used
}

[INSERT: Personal Artistic Segway to Methods]

## SoftSort
[GOAL: ??? ]

[INSERT: Personal Artistic Intro to Methods]

[
    INSERT: Factual overview of all things contained in this chapter;
    DEPENDSON: All the information in this chapter is done
]

[!TODO(writeup): Compile all points I want to include in this chapter] 
    => [TODO(organize): Organize this information into coherent blocks] 
    => [TODO(story): Create a story from this information] 
{

- [ ] Core concepts and mathematical description
- [ ] Definition (SoftSort)
  - [ ] Diagram
- [ ] Theoretical Properties of SoftSort
- [ ] Dimensionality Reduction
- [ ] Integration with Annealing schemes
- [ ] Performance Characteristics and Limitations
- [ ] Connection Approximate Symmetry Problem

 - [ ] Note automatic differentiation using pytorch is being used
}

[INSERT: Personal Artistic Segway to Methods]

## Summary
[GOAL: ??? ]

[
    INSERT: Factual overview of all things contained in this chapter;
    DEPENDSON: All the information in this chapter is done
]

[INSERT: Personal Artistic Segway to Experimental Results]

# Experimental Results

[GOAL: ??? ]

 - [ ] Note on automatic differentiation

[INSERT: Personal Artistic Segway to Experimental Results]

## Graph Instances

[INSERT: Personal Artistic Segway to Graph Instances]

# Conclusion

[INSERT: Personal Artistic Segway to Conclusion]

[INSERT: Factual overview of all results]

## Future Work

[INSERT: Include future work]

# Appendix

## Code documentation