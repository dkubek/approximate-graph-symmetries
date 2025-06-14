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

[GOAL: ??? ]

[INSERT: Personal Artistic Intro to Notation]

[INSERT: Factual overview of what is contained in this chapter]

[!TODO(writeup): Compile all points I want to include in this chapter] 
    => [TODO(organize): Organize this information into coherent blocks] 
    => [TODO(story): Create a story from this information] 
{
- motivation for studying this problem
    - brain symmetries
    - look at problem from different angle
    - problem in relation to different problems
- General overview and definitions of the problems

- Graph Isomorphism (GI)
- Graph Matching Problem (GMP)
- Quadratic Assignment Problem (QAP)
- Approximate Symmetry Problem
    - Introduce Approximate symmetry metrics (relate to GMP)
    - Introduce the fragility of approximate symmetries on an example
    - Relax
      - Why relax GMP or QAP

- Linear Assignment Problem (LAP)

- Representing Permutations
    - overview ways of how permutation can be represented
    - sorting networks

- Problems Specific to Approximate Symmetries
    - Choosing a starting Point
    - Excluding Identity
    - Not Well-definededness of the Problem
    - Mapping relaxed permutation back to a permutation

- Literature Review
    - Present work fo Liu, Vandenberghe, Matej Straka, Franta Sczepanik
    - Liu => Franta Sczepanik
    - Vandenberghe => Matej Straka
    - David Hartman, Pidnebesna
}


[INSERT: Personal Artistic Segway to Manifold Optimization]

## Optimization on Smooth Surfaces

[INSERT: Personal Artistic Intro to Manifold optimization]

[!TODO(writeup): Compile all points I want to include in this chapter] 
    => [TODO(organize): Organize this information into coherent blocks] 
    => [TODO(story): Create a story from this information] 
{
- manifold optimization in general terms (not DS specific)
- stop 0 of optimization
- submanifold
- ambient space
- general optimization problem
- classical (un)costrained optimization vs manifold optimization, 
- smooth manifold definition through local defining functions
- geometry
- tangent spaces
- retractions
    - first order approximations to exponential maps
- riemannian metrics
- vector transport
- riemannian distances and metrics
- Practical libraries - manopt
  - gradient descent
  - trust region
  - conjugate gradients
- exponential problem / exponential map / logarithmic map
    - exponential map is often difficult to find
    - in practice retractions are often used
- geodesic
- moving along geodesics
- Lie groups
- riemannian gradient computation  
  - in practice compute euclidean gradient
-manifolds without known geodesic
- multiple riemannian metrics
- distance along tangent vectors
- time parameter
- higher order retractions
- retractions vs true geodesics in practice

}

[INSERT: Personal Artistic Segway to Manifold Optimization]


## Summary

[INSERT: Summary of all the things done in this chapter]

[INSERT: Personal Artistic Segway to Methods]


# Methods

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
 - [ ] Interior Point Method
 - [ ] Manifold
 - [ ] OT4P
 - [ ] SoftSort
}

[INSERT: Personal Artistic Segway to Methods]

# Experimental Results

[GOAL: ??? ]

 - [ ] Note on automatic differentiation

[INSERT: Personal Artistic Segway to Methods]

# Conclusion

[INSERT: Personal Artistic Segway to Conclusion]

[INSERT: Factual overview of all results]

## Future Work

# Appendix

## Code documentation