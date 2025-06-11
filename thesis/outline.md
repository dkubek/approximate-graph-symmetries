# Introduction

# Preliminaries

## Notation

# Approximate Symmetry Problem

[CHAPTER INTRODUCTION]
- introduce goals
- Goal: Present the problem of approximate symmetries from a different point of view
  - FRantisek Szcepanik showed a combinatorial approach to this problem , we will explore the relaxation of this problem
  - Present the work of Liu, Vandenberghe, Matej Straka and Franta Szcepanik
  - intention to look at the problem from a different angle
  - introduce the problem in relation to different known studiend problems and how the problem of approximate symmetries naturally arises when following the logic of these problems
    - show how these problems differ and how do they relate
  - 

## Problem overview

(and literature overview)

- introduce the problem of graph symmetries (Why)

- quick overview (setup, playground, reiteration)
- goal is to look for approximate symmetries in networks that do not necessarily have one in a _strinct_ sense

- introduce the measure proposed by approximate symmetries proposed by Liu
  - what does it represent?
  - normalized appx symmetry coefficient
  - problems and disadvantages with this definition (it has obvious optimum)

- How Liu suggest to solve this problem? -> Simmulated Annealing
  - Work of Franta Szcepanik
  [TODO: Include and analyse the results of his work]

### Graph Isomorphism

- describe the problem of graph isomorphism
- include the recent result that graph isomorphism is quasi-polynomial
- present the combinatorial nature of this problem
- [TODO: Research? How is the quasipoly result achieved? Are any relaxations involved?]

### Graph Matching Problem (GMP)

- [TODO: Find some canonical, summary report, review paper for GMP]
- definition of the problem, motivation, idea
- generalisation of the graph isomorphism (we allow errors)
- complexity of this problem [TODO: Find citations]


### Quadratic Assignment Problem (QAP)

- definition, matrix form, 
- relation to GMP

- add reformulation to GMP -> QAP (mathematical)

- compare the properties of QAP and GMP compare the properties of both of these problems
  - binary QP + LC => NP hard to find global optimumo
  
### Representing Permutations

- permutation vectors
- permutation matrices

- birkhoff polytope (find the sources for the results of birkhoff)-
- explain how the permutations can be embedded in the space of doubly stochastic metrices
  - which in turn is embedded in the sapce of singly stochastic matrices which(which does not have the save nice properteis)
- explain the dimensionality of the space of permutations and the dimensionality of the space of ddoublu stochastic matrices

- relax
  - why do we relax? 
    - we hope that we get to explot the geometry of the space to find solution faster
    - we hope to exploit differentiability and speed of methods of continouous optimization
  - How do we relax?

- relaxed QAP (rQAP) ia a QP with linear constraints
  - !! it is not generally convex (HESSIAN is indefinite)
  - non-convex quadratic optimization is NP-hard
    - we can try to use the tools of continouous optimization to look for at least local optima


### Why relax QAP instead of GMP
- Present the result of [TODO: Insert author] fastQAP and relax at your own risk
- QAP approximation provides GMP approximation

- Why relax QAP instead of GMP?

### Problems Specific to Approximate Symmetries

#### Excluding Identity

- approximate symmetry problem is in a sense not well defined since it has the obvious solution of "identity"
  - we want to exclude this identity
    - in iterative local search methods, this is simple (disregard solutions with many fixed points)

- using penalization

- [TODO: FUture work, describing the convex hull of permutations]
  - permutations containing identity can be detected through tournaments using LP, one of the questions posed by birkhoff

#### Mapping Back to Permutations

- hungarian algorithm and the "closest permutation"
- von neumann decomposition
- decomposition with random error

### Summary

- Graph Isomorphism -> Graph Matching -> Quadratic Assignment -> Approximate Symmetries

- Current Solution approach and current solution concepts; include the description of Liu and the algotrightm of Vandenberghe

- describe the algorithm of vandenberge anew because it has mistakes
- inlude the work of simmulated annealing of Franta Sczepanik and the work of Matej Straka


# Optimization Methods

# FastQAP

### Interior Point Methods

- explain the basic idea and motivation behind the interior point method (noveda)
- why it is interesting 

### 

# Experimental Results

## Data Instances Descriptions

### ER

### BA

### LRM ER Rewired

## Experimental Results

### 

# Conclusion

## Future Work