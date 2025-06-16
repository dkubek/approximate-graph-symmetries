### Interior Point Method


#### Algorithmic Choices
[SUBTITLE: Using more sophisticated algorithms]

- We use IPOPT interior point solver
  - open source, widely available, has many options (maybe too many)
  - [TODO: Different choices of linear systems solution algorithms]
    - [TODO: Explain how the "ma57" algorithm from IPOPT work]
    - MUMPS is also good I guess but we have gained practical speedup in 2 orders of magnitute agains the standrad MUMPS sovler using the HSL library
