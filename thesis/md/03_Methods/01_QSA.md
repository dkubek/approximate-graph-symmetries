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
