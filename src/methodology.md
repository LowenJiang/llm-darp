\chapter{Methodology}

%Tripartite: Solver,  LLM, PPO
\section{Neural Solver for the Dial-a-Ride Problem}

Since the DARP is NP-hard, exact solvers scale poorly with instance size.
In our framework, routing cost is evaluated repeatedly within the
PPO-based negotiation training loop (covered in section 4.3), making exact methods computationally
prohibitive. We therefore develop a neural combinatorial solver that learns
to construct routes directly, enabling fast approximate evaluation of
routing cost during training.

Following the neural combinatorial optimization framework of
\cite{vinyals2015pointer} and \cite{kool2018attention}, we cast routing
as a sequential decision process. Given a graph $G$ describing the set of
requests, the solver constructs a route
$\tau = (\tau_1, \dots, \tau_T)$ by selecting one node at a time from a
learned distribution over feasible nodes. At each step $t$, an
autoregressive policy $\pi_\theta$ samples the next node
$\tau_t \sim \pi_\theta(\cdot \mid G,\, \tau_{1:t-1})$, conditioned on
the graph and the partial route constructed so far. The probability of a
complete solution under this policy factorizes as:
%
\begin{equation}
    \pi_{\theta}(\tau \mid G)
      = \prod_{t=1}^{T}
        \pi_{\theta}(\tau_t \mid G,\; \tau_{1:t-1}).
    \label{autoregressive}
\end{equation}
%
The route $\tau$  The policy $\pi_\theta$ is parameterized by a Transformer-based
architecture where the graph is encoded via multi-head attention mechanism, and decoded by sequentially querying the graph with dynamic state, yielding the route $\tau$. 
\cite{vinyals2015pointer, kool2018attention}. The parameters $\theta$ are
optimized via the REINFORCE algorithm \cite{williams1992simple} to
minimize the expected total routing cost:
%
\begin{equation}
    \min_{\theta}\;
      \mathbb{E}_{\tau \sim \pi_{\theta}(\cdot \mid G)}
      \left[\,\sum_{t=0}^{|\tau|-1}
        c_{\tau_t,\, \tau_{t+1}} \right].
    \label{ar_objective}
\end{equation}


\subsection{DARP Route Construction}
Unlike the TSP setup in \cite{kool2018attention}, where a route corresponds to a permutation of all nodes, we represent a DARP solution by treating the full trajectory $\tau$ as a concatenation of sub-routes $\tau^l$, each starting and ending at the depot and encoding a single vehicle's path \eqref{eq:concat}, where $\|\cdot$ denotes concatenation.
%
\begin{equation}
    \tau
      = \bigl\|_{l=1}^{\leq K}\,\tau^lb = [\tau^{l}]_{l=1, \dots, K},
    \qquad \tau^l_s \in \mathcal{N}
    \label{eq:concat}
\end{equation}

A sub-route $\tau^l$ is simply the sequential view of the arc variables
$\{x_{ij}^l\}$: $x_{ij}^l = 1$ if and only if node~$j$ immediately
follows node~$i$ in $\tau^l$. A solution $\tau$ is feasible if and only
if the number of sub-routes is no greater than the fleet size~$K$ and
each sub-route satisfies \eqref{eq:darp-constraints}. We note that the time constraints 
in \eqref{eq:c5}--\eqref{eq:c7} are tracked by vehicle clock variable $T$. Each return to
the depot in $\tau$ therefore separates two vehicle routes and resets
the vehicle clock $T \leftarrow 0$, allowing the entire solution to be
decoded in a single autoregressive pass.

At each decoding step, nodes that satisfy the following conditions are considered infeasible and masked by setting their logits to $-\infty$: {\color{red} define the logits}
%
\begin{enumerate}
    \item \textit{Visited.} The node has already been served.
    \item \textit{Precedence.} A dropoff node~$n{+}i$ is masked
          unless its corresponding pickup node~$i$ has been visited
          on the current vehicle's route.
    \item \textit{Time window.} A pickup node~$i$ is masked if the
          vehicle cannot arrive before its latest permissible
          service time, i.e., $T + c_{\tau_t,\, i} > l_i$.
\end{enumerate}

The above conditions are necessary but not sufficient to guarantee
route feasibility, as they do not preclude stranded pickups -- requests
whose corresponding delivery cannot be completed by the same vehicle
within the time window. A complete feasibility check at each step
would require verifying the existence of a Hamiltonian path through
all picked-up but undelivered nodes that respects every remaining
time window, amounting to a PDPTW subproblem at each decoding step.
This would undercut the computational advantage of the neural solver,
so we retain only the efficiently vectorizable masks above and apply
a reconstruction technique to ensure feasibility of the decoded route. 

When the policy selects the depot (i.e., $\tau_t = 0$), the current
vehicle's route is finalized and we remove any stranded pickups. Let $\mathcal{U}$ denote the set of pickups that
have been visited on the current vehicle route $\tau^l$ but whose corresponding
deliveries have not:
%
\begin{equation}
    \mathcal{U} = \bigl\{\, p 
      : p\in \tau^l \cap {P}, (p+n) \notin \tau^l \cap {D} \,\bigr\}
    \label{eq:reconstruction}
\end{equation}
%
Rather than declaring the solution infeasible, we remove
$\mathcal{U}$ from the current route and return these pickups to
the pool of unserved nodes, making them available for assignment to
subsequent vehicles. This continues until all $2n$ nodes
have been served.

\subsection{Policy Architecture}

The policy $\pi_\theta$ follows the encoder--decoder attention model
of~\cite{kool2018attention}, adapted for the DARP. The encoder
maps each problem instance to a set of node embeddings via
multi-head self-attention; the decoder then constructs the visit
sequence~$\tau$ autoregressively by attending over these embeddings
at each step.

% ----------------------------------------------------------------
With a slight abuse of notation, let $n=2N+1$ denote the total number of nodes, and $\mathbf{x}$ for feature vector.  We write $d$ for the embedding dimension,
$M$ for the number of attention heads, $d_k = d/M$ for the
per-head dimension, and $L$ for the number of encoder layers.

% ================================================================
\paragraph{Encoder.}
The encoder runs once per instance and produces node
embeddings $\mathbf{H} \in \mathbb{R}^{n \times d}$.

For each node~$i$, coordinates and time-window endpoints are
normalised to $[0,1]$ via instance-wise min--max scaling. Together
with the demand indicator $q_i \in \{+1, 0, -1\}$ (pickup, depot,
delivery), this yields a five-dimensional feature vector
%
\begin{equation}
    \mathbf{x}_i
      = \bigl[\,\hat{x}_i,\;\hat{y}_i,\;
        \hat{e}_i,\;\hat{l}_i,\;q_i\,\bigr]
      \in \mathbb{R}^{5},
    \label{eq:node-features}
\end{equation}
%
which is projected into the embedding space:
%
\begin{equation}
    \mathbf{h}_i^{(0)}
      = W_{\mathrm{init}}\,\mathbf{x}_i
        + \mathbf{b}_{\mathrm{init}},
    \qquad
    W_{\mathrm{init}} \in \mathbb{R}^{d \times 5}.
    \label{eq:init-embed}
\end{equation}

The initial embeddings are refined by a stack of $L$ identical
transformer layers ~\cite{vaswani2017attention}. Each layer~$l$
applies multi-head self-attention (MHA) followed by a position-wise
feed-forward network (FFN), both wrapped with residual connections ~\cite{he2016deep} and batch normalisation ~\cite{ioffe2015batch}:
%
\begin{align}
    \hat{\mathbf{H}}^{(l)}
      &= \mathrm{BN}\!\bigl(\mathbf{H}^{(l-1)}
         + \mathrm{MHA}^{(l)}(\mathbf{H}^{(l-1)})\bigr),
    \label{eq:enc-mha} \\[4pt]
    \mathbf{H}^{(l)}
      &= \mathrm{BN}\!\bigl(\hat{\mathbf{H}}^{(l)}
         + \mathrm{FFN}^{(l)}(\hat{\mathbf{H}}^{(l)})\bigr).
    \label{eq:enc-ffn}
\end{align}
%
In the MHA sub-layer, queries, keys, and values are obtained from a
single linear projection of the input, split into $M$ heads, and
combined via scaled dot-product attention:
%
\begin{equation}
    \mathrm{Attn}_m^{(l)}
      = \mathrm{softmax}\!\left(
          \frac{\mathbf{Q}_m^{(l)}\,
                (\mathbf{K}_m^{(l)})^\top}
               {\sqrt{d_k}}
        \right) \mathbf{V}_m^{(l)},
    \label{eq:sdp-attn}
\end{equation}
%
with heads concatenated and projected through
$W_O^{(l)} \in \mathbb{R}^{d \times d}$. The FFN consists of two
linear layers with a ReLU activation and hidden dimension
$d_{\mathrm{ff}}$. The final encoder output is
$\mathbf{H} = \mathbf{H}^{(L)} \in \mathbb{R}^{n \times d}$.

% ================================================================
\paragraph{Decoder.}
The decoder constructs the route step by step using a pointer
mechanism ~\cite{vinyals2015pointer}. Before decoding begins,
a single linear projection of~$\mathbf{H}$ is split three ways
to obtain decoder keys, values, and logit keys:
$\mathbf{K}_{\mathrm{dec}},\,
 \mathbf{V}_{\mathrm{dec}},\,
 \mathbf{K}_{\mathrm{logit}} \in \mathbb{R}^{n \times d}$.
The node embeddings~$\mathbf{H}$ are also cached for reuse
in the context computation below.

At decoding step~$t$, the query is assembled from six
$d$-dimensional signals that capture global structure, current
position, temporal state, and constraint information:
%
\begin{equation}
    \mathbf{c}_t
      = \Bigl[\;\underbrace{\bar{\mathbf{h}}}_{\text{graph}}\;;\;
        \underbrace{\mathbf{h}_{c_t}}_{\text{current}}\;;\;
        \underbrace{\mathbf{h}_0}_{\text{depot}}\;;\;
        \underbrace{T_t \cdot
          \mathbf{w}_{\mathrm{time}}}_{\text{time}}\;;\;
        \underbrace{\bar{\mathbf{h}}_t^{\,\neg}}_{\text{masked}}\;;\;
        \underbrace{\bar{\mathbf{h}}_t^{\,\mathcal{P}}}_{\text{pending}}\;
      \Bigr]
      \;\in\; \mathbb{R}^{6d},
    \label{eq:context-concat}
\end{equation}
%
where $\bar{\mathbf{h}} = \frac{1}{n}\sum_i \mathbf{h}_i$ is the
mean node embedding,
$\mathbf{h}_{c_t}$ is the embedding of the current node,
$\mathbf{h}_0$ is the depot embedding,
$\mathbf{w}_{\mathrm{time}} \in \mathbb{R}^d$ is a learnable
parameter that broadcasts the scalar current time~$T_t$ to
embedding dimension,
$\bar{\mathbf{h}}_t^{\,\neg}$ is the mean embedding of currently
infeasible (masked) nodes, and
$\bar{\mathbf{h}}_t^{\,\mathcal{P}}$ is the mean embedding of
picked-up but not yet delivered nodes.
The concatenated vector is then projected by a two-layer MLP:
%
\begin{equation}
    \mathbf{q}_t
      = W_2\,\mathrm{ReLU}(W_1\,\mathbf{c}_t + \mathbf{b}_1)
        + \mathbf{b}_2
      \;\in\; \mathbb{R}^{d},
    \qquad
    W_1 \in \mathbb{R}^{2d \times 6d},\;
    W_2 \in \mathbb{R}^{d \times 2d}.
    \label{eq:context-query}
\end{equation}

During decoding, the query~$\mathbf{q}_t$ is split into $M$
heads, each computing scaled dot-product attention against the
precomputed decoder keys~$\mathbf{K}_{\mathrm{dec}}$. The
feasibility mask~$\mathbf{M}_t$ sets entries of infeasible
nodes to~$-\infty$ before the softmax, yielding attention
weights over all nodes:
%
\begin{equation}
    \boldsymbol{\alpha}_t
      = \mathrm{softmax}\!\left(
          \frac{\mathbf{q}_t\,
                \mathbf{K}_{\mathrm{dec}}^\top}
               {\sqrt{d_k}}
          + \mathbf{M}_t
        \right)
      \;\in\; \mathbb{R}^{n}.
    \label{eq:attn-weights}
\end{equation}
%
These weights are used to compute a weighted combination of
the decoder values~$\mathbf{V}_{\mathrm{dec}}$. The per-head
results are concatenated and projected through
$W_g \in \mathbb{R}^{d \times d}$, producing a single vector
that summarizes the decoder's view of the graph at step~$t$:
%
\begin{equation}
    \mathbf{g}_t
      = W_g\,
        \boldsymbol{\alpha}_t\,
        \mathbf{V}_{\mathrm{dec}}
      \;\in\; \mathbb{R}^{d}.
    \label{eq:glimpse}
\end{equation}
%
The output logits are then computed as the scaled dot product
of~$\mathbf{g}_t$ with a separate set of precomputed logit
keys~$\mathbf{K}_{\mathrm{logit}}$, clipped
to~$[-C,\,C]$ with $C = 10$:
%
\begin{equation}
    u_{t,i}
      = C \cdot \tanh\!\left(
          \frac{\mathbf{g}_t
                \cdot \mathbf{k}_{\mathrm{logit},i}}
               {\sqrt{d}}
        \right),
    \qquad i = 1,\dots,n.
    \label{eq:logits}
\end{equation}
%
Infeasible nodes are masked by setting
$u_{t,i} \leftarrow -\infty$ for all~$i$ where
$[\mathbf{M}_t]_i = -\infty$. The action probability is then
%
\begin{equation}
    \pi_\theta\!\left(j \mid \tau_{1:t-1}\right)
      = \frac{\exp(u_{t,j})}
             {\sum_{k}\exp(u_{t,k})}.
    \label{eq:action-prob}
\end{equation}
%
During training the action is sampled from this distribution;
at test time we use greedy 

%------------

\subsection{REINFORCE Algorithm} 

% Reward design; probability of taking this
REINFORCE is a policy gradient method to directly optimize a parameterized policy by shifting probability onto trajectories with higher reward \cite{sutton1999policy}. The probability of taking trajectory $\tau = \parallel_{l=1}^{k_\tau}[\tau^l]$ under policy $\pi_\theta$ is given by (\ref{autoregressive}), and since we constrain the fleet size to be $K$, the reward is defined as the negative routing cost, penalized by $p_v$ for additional vehicle in use: 
\begin{equation}
    r(\tau) =  -\sum_{t=0}^{|\tau|-1} c_{t,t+1}  - p_v \max\{(k_\tau - K), 0\}
\end{equation}
The REINFORCE gradient estimator of sampling $L$ trajectories $\{\tau^{(1)}, \dots, \tau^{(L)}\}$ is given by (\ref{eq:reinforce}) as the average product of probability gradient and reward. $b^{(i)}$ is the baseline associated with $\tau^{(i)}$ to reduce variance. During each epoch, this baseline is calculated as the greedy rollout of the best performing policy so far. 



\begin{equation}
\label{eq:reinforce}
    \nabla_\theta J(\theta) = \frac{1}{L}\sum_{i=1}^L 
    \left[ \sum_{t=1}^{|\tau^{(i)}|} \nabla_\theta \log \pi_{\theta}(\tau_t^{(i)}\mid G, \tau_{1:t-1}^{(i)}) \right] 
    \left[ r(\tau^{(i)}) - b^{(i)} \right]
\end{equation}


\begin{algorithm}[H]
\caption{Neural Solver Training Algorithm}
\label{alg:reinforce}
\begin{algorithmic}[1]
\Require Epochs $E$, parallel environments $L$, training size $S$, validation size $V$, learning rate $\alpha$, initial policy $\pi_\theta$, request distribution $\mathcal{D}$, initial baseline $\pi_{b} \leftarrow \pi_{\theta}$, fleet size $K$, daily requests $N$
\State Sample fixed validation set $\{G_v^{(j)}\}_{j=1}^{V} \sim \mathcal{D}$
\For{epoch $= 1, \dots, E$}
    \For{step $= 1, \dots, S/L$}
        \For{$i = 1, \dots, L$}
            \State Sample $G^{(i)} \sim \mathcal{D}$
            \State $\tau \leftarrow \varnothing,\; l \leftarrow 0,\; d \leftarrow 0$
            \While{$d < 2N$} \Comment{Trajectory Collection}
                \State $\tau^l \sim \pi_\theta(G^{(i)}, \tau)$
                \State $d \leftarrow d + \texttt{reconstruct}(\tau^l)$ \Comment{Reconstruct Technique}
                \State $l \leftarrow l + 1;\; \tau \leftarrow \tau \| \tau^l$
            \EndWhile
            \State $g^{(i)} \leftarrow \sum_{t=1}^{|\tau^{(i)}|} \nabla_\theta \log \pi_{\theta}(\tau_t^{(i)} \mid G^{(i)}, \tau_{1:t-1}^{(i)})$
            \State $r^{(i)} \leftarrow -\sum_{t=0}^{|\tau|-1} c_{t,t+1} - p_v \max\{k_\tau - K,\; 0\}$
            \State $b^{(i)} \leftarrow r\bigl(\texttt{greedy}(\pi_b, G^{(i)})\bigr)$
        \EndFor
        \State $\nabla_\theta J(\theta) \leftarrow \frac{1}{L} \sum_{i=1}^{L} g^{(i)} \bigl(r^{(i)} - b^{(i)}\bigr)$ \Comment{REINFORCE Update}
        \State $\theta \leftarrow \theta + \alpha \, \nabla_\theta J(\theta)$
    \EndFor
    \State $R_\theta \leftarrow \frac{1}{V} \sum_{j=1}^{V} r\bigl(\texttt{greedy}(\pi_\theta, G_v^{(j)})\bigr)$
    \State $R_b \leftarrow \frac{1}{V} \sum_{j=1}^{V} r\bigl(\texttt{greedy}(\pi_b, G_v^{(j)})\bigr)$
    \If{$R_\theta > R_b$}
        \State $\pi_b \leftarrow \pi_\theta$ \Comment{Baseline Update}
    \EndIf
\EndFor
\end{algorithmic}
\end{algorithm}


\section{LLM-based User Preference Modeling}

% Add a bit intuition

Let $I$ denote the number of travelers in the system and $J$ denote the number of trip types. Each trip type $j \in [J]$ is characterized by a trip origin, trip destination, departure time window, and arrival time window.

Each traveler $i \in [I]$ has an observable profile $u_i$ available to the system, which may include attributes such as age, occupation, and income level. In addition, each traveler has a latent preference regarding schedule shifts that is not observable to the system. We assume there are $L = 4$ possible preference types: (i) flexible for both earlier pickup and later dropoff, (ii) flexible for earlier pickup but inflexible for later dropoff, (iii) flexible for later dropoff but inflexible for earlier pickup, and (iv) inflexible to any schedule shift. Let $\ell_i$ denote the latent schedule-shift preference type of traveler $i \in [I]$.

Let $\mathcal{A} $ denote the set of schedule shifts that the system can propose, where each $a:=(\delta^p, \delta^d) \in \mathcal{A}$ specifies a change in pickup and dropoff times. When traveler $i$ receives a proposed schedule shift $a$ for trip type $j$, the traveler decides whether to accept or reject the proposal according to a latent decision function $H(u_i, \ell_i, j, a)$, which is not observable to the system.

The system observes a dataset of $O$ interaction records. Each data point $o \in [O]$ corresponds to a proposed schedule shift for a traveler and consists of a tuple $(i_o, j_o, a_o, \beta_o)$, where $i_{o}$ is the index of the traveler involved in the interaction, $j_o \in [J]$ is the trip type, $a_o \in \mathcal{A}$ is the proposed schedule shift, and $\beta_o \in \{0,1\}$ indicates whether the traveler rejected ($0$) or accepted ($1$) the proposal.

We propose an LLM-based approach to predict the unobservable schedule-shift preference  for each traveler $i \in [I]$, following \cite{liu2025aligning}. The LLM acts as a simulator to assist inference of traveler flexibility type. Specifically, we use a matrix $\Xi \in \mathbb{R}^{I \times D}$ to represent the embedding matrix of the $I$ travelers, where the $i$-th row $\xi_i$ denotes the $D$-dimensional embedding vector of traveler $i$. We then apply a single feedforward neural network, denoted by $f^{\phi}: \mathbb{R}^D \rightarrow [0,1]^L$, which maps the embedding to traveler type and is shared across all travelers. For each traveler $i \in [I]$, the network takes the embedding vector $\xi_i$ as input and outputs the probabilities that the traveler belongs to each of the $L$ preference types. In particular, the {\em estimated} probability that traveler $i$ has preference type $\ell$ is denoted by $f^{\phi}_{\ell}(\xi_i) \in [0,1]$ for each $\ell \in [L]$. By definition we have $\sum_{\ell=1}^{L} f^{\phi}_{\ell}(\xi_i)=1 $ for all $i\in [I]$.

For each data point $o \in [O]$ and each latent preference $\ell \in [L]$, we use an LLM agent to predict the traveler’s decision (i.e., accept or reject) given the traveler’s observable profile $u_{i_o}$, latent preference $\ell  $, trip type $j_o$, and the proposed time shift $a_o$. We denote the output of the LLM agent as
\begin{align} \label{eq:llm-llm}
    \hat{\beta}_{o\ell} := \mathrm{LLM}(u_{i_o}, \ell, j_o, a_o).
\end{align}
% This is the LLM simulated output of what the acc/rej should be given that the type is ell

\begin{Comment}
More broadly, the LLM agent predicts acceptance probability over all actions $a\in \mathcal{A}$, with the output denoted as $\hat{\beta}^{a}_{o\ell} := \text{LLM}(u_{i_o}, \ell, j_o, a) $. By law of total probability, the acceptance probability of an arbitrary action $a\in \mathcal{A}$ for datapoint $o\in [O]$ can be written as: 
\begin{equation}
\label{eq:total-probability}
    \mathbb{P}(\beta=1\mid a, o) = \sum_{\ell=1}^{L}f_{\ell}^{\phi}(\xi_{i_o})\hat{\beta}_{o\ell}^{a}
\end{equation}

\end{Comment}

We jointly optimize the embedding matrix $\Xi$ and the neural network parameters $\phi$ by maximizing the likelihood function \eqref{eq:llm-obj}.
% Note that f-xi is between 0 and 1. the coefficient measures how type consistency. 

\begin{align} \label{eq:llm-obj}
    \max_{\Xi, \phi} \sum_{o = 1}^O \sum_{\ell = 1}^L \hat{w}_{o\ell}^{\phi} \log(f^{\phi}_{\ell}(\xi_{i_o})) + \alpha_m \mathcal{L}(\Xi).
\end{align}

In \eqref{eq:llm-obj}, the coefficient $\hat{w}_{o_\ell}^{\phi}$ represents a soft assignment weight that measures how consistent latent preference type $\ell$ is with the observed decision $\beta_o$ for data point $o$. These weights are constructed using the LLM predictions and the current estimates of the preference probabilities. The weights $\hat{w}_{n_\ell}^{\theta}$ are defined in \eqref{eq:llm-what}, while the regularization term $\mathcal{L}(\Xi)$ is defined in \eqref{eq:llm-reg}.

Specifically, the coefficient \eqref{eq:llm-what} assigns positive weight only to those latent preference types whose LLM-predicted decision $\hat{\beta}_{o_\ell}$ matches the observed decision $\beta_o$. Among these consistent preference types, the weights are normalized according to the current model probabilities $f^{\phi}_{\ell}(\xi_{i_o})$. The multiplicative factor involving $\alpha_e$ downweights observations for which all preference types produce the same predicted decision, since such cases provide little information for identifying the traveler’s latent preference.

\begin{align} \label{eq:llm-what}
    \hat{w}_{o\ell}^{\phi} := \frac{f^{\phi}_{\ell}(\xi_{i_o}) \mathds{1}\{\hat{\beta}_{o\ell} = \beta_o\}}{\sum_{\ell' = 1}^L f^{\phi}_{\ell'}(\xi_{i_o}) \mathds{1}\{\hat{\beta}_{o{\ell'}} = \beta_o\}} \times \left( 1 - \alpha_e \mathds{1}\left\{\prod_{\ell' = 1}^L \mathds{1}\{\hat{\beta}_{o{\ell'}} = \beta_o\} = 1 \right\} \right).
\end{align}
The regularization term \eqref{eq:llm-reg} is introduced to prevent any single embedding dimension from dominating the representation. In particular, $\sigma^2(\xi_{\cdot d})$ denotes the variance of the embedding parameters across travelers in dimension $d$, and $\bar{\sigma}^2$ denotes the mean of these variances across all $D$ embedding dimensions. The regularization penalizes large disparities in the variances of different embedding dimensions, encouraging the embedding space to remain balanced and preventing individual dimensions from taking disproportionately large or highly variable values.

\begin{align} \label{eq:llm-reg}
    \mathcal{L}(\Xi) := \sum_{d=1}^D \left(\frac{\sigma^2(\xi_{\cdot d})}{\bar{\sigma}^2} - 1 \right)^2.
\end{align}


\section{Negotiation Scheduling Layer}

% How to encode the state space, action space, reward
We formulate the sequential negotiation process as a Markov decision process (MDP) and train a perturbation policy $\Gamma_{\psi}$ via Proximal Policy Optimization (PPO) to maximize the objective in \eqref{eq:negotiation-objective}. 

The observable state $s_t$ comprises all previously fixed requests together with the incoming one  $(\{\rho_j'\}_{j=1}^{t-1} , \rho_t)$. To obtain a fixed dimensional encoding that captures graph features, we reuse the frozen encoder parameters $\mathrm{Enc}_\theta$ from \eqref{eq:sdp-attn}
to obtain node embeddings: 
\begin{equation}
    \{h_1, h_2, \dots, h_{t-1}, h_t\} = \text{Enc}_\theta(\rho_1', \rho_2',\dots, \rho_{t-1}', \rho_t)
\end{equation}
We aggregate the fixed nodes through mean pooling, where $h_0$ is learned bias that serves as the initial context when no prior requests have been committed. 
\begin{equation}
    \bar{h}_{<t} =h_0 + \mathds{1}\{t>1\} \left( \frac{1}{t-1} \sum_{j=1}^{t-1} h_j \right)
\end{equation}
and concatenate with the current request embedding  to form the state vector: d
\begin{equation}
    s_t = \left[ \bar{h}_{<t} \;\|\; h_t \right]\in \mathbb{R}^{2d_h }
\end{equation}
The policy $\Gamma_\psi \colon \mathbb{R}^{2d} \to \Delta^{|\mathcal{A}|}$ maps the state to a distribution over actions.  Rather than applying a hard mask that sets infeasible logits to~$-\infty$, we introduce a \emph{soft mask} that accounts for the uncertain trip type~$\ell$:
\begin{equation}
\label{eq:ppo-policy-action}
    \Gamma_\psi(a_t\mid s_t) = \text{softmax}\left(\text{FFN}(s_t)  + \kappa \log\mathbb{P}(\beta =1\mid a_t, \rho_t) \right) 
\end{equation}
where the acceptance probability~$\mathbb{P}(\beta=1 \mid a_t, \rho_t)$ is defined in~\eqref{eq:total-probability} and $\kappa \in \mathbb{R_+}$ controls the strength of the acceptability prior.
The stepwise reward is defined as the reduction in routing cost by taking the action $a_t$ compared to non-action ($\rho_t' = \rho$), plus the patience penalty, and $\mathcal{C}(\cdot)$ is the trained neural solver in 4.1.
\begin{equation}
    R(s_t, a_t) = \mathcal{C}(\{\rho_j'\}_{j=1}^{t-1}, \rho_t ) - \mathcal{C}(\{\rho_{j}'\}_{j=1}^{t}) - \omega \|\rho_t'-\rho\|_{1}
\end{equation}
We use separate networks of the same architecture for actor  $\Gamma_\psi(a_t\mid s_t)$ and critic $V_\varphi(s_t)$. The GAE (Generalized Advantage Estimator) \cite{schulman2015high} advantage function is calculated as \eqref{eq:gae}, where $\gamma, \lambda$ are the discount factor and GAE trace-decay parameter respectively. 
\begin{equation}
\label{eq:gae}
\hat{A}^{\text{GAE}(\gamma,\lambda)}_t = \sum_{k=0}^{\infty} (\gamma \lambda)^k \, \delta_{t+k}, \quad \delta_t = R(s_t, a_t) + \gamma V_\varphi(s_{t+1}) - V_\varphi(s_t)
\end{equation}
And the PPO objective optimizes the clipped surrogate loss \cite{schulman2017proximal}:
\begin{equation}
\label{eq:ppo-loss}
    L^{\text{PPO}}(\psi) = \mathbb{E}_t \left[ \min \left( \frac{\Gamma_\psi(a_t \mid s_t)}{\Gamma_{\psi_{\text{old}}}(a_t \mid s_t)} \hat{A}_t^{\text{GAE}}, \; \text{clip}\left(\frac{\Gamma_\psi(a_t \mid s_t)}{\Gamma_{\psi_{\text{old}}}(a_t \mid s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t^{\text{GAE}} \right) \right]
\end{equation}
where $\epsilon$ is the clipping threshold. The critic parameters~$\varphi$ are updated independently by
minimizing the value function loss
\begin{equation}
\label{eq:vf-loss}
    L^{\text{VF}}(\varphi) = \mathbb{E}_t \left[
      \bigl(V_\varphi(s_t) - \hat{R}_t\bigr)^2
    \right],
\end{equation}
where $\hat{R}_t = \hat{A}_t^{\text{GAE}} +
V_{\varphi_{\text{old}}}(s_t)$ is the target return. The entire LLM-enhanced negotiation policy training process is summarized in algo 2.

\begin{algorithm}[H]
\caption{Neural Solver Training Algorithm}
\begin{algorithmic}[1]
\Require   Number of Policy Iterations I, number of trajectories per iteration $L$, Initial user embedding $\Xi$, embedding update iterations m, Buffer $B \leftarrow \emptyset$,  Initial policy network $\Gamma_\psi$,  initial value network $V_\varphi$, clipping rate $\epsilon$, learning rate $\alpha$

\For {Iterations in 1, \dots, I}
    \State Run policy $\Gamma_\psi$ over L monte-carlo trajectories, load dataset in buffer $B$
    \State 
\EndFor

    
\end{algorithmic}
\end{algorithm}

%Define state space, action space, etc. 
% How the LLM part prediction goes into this

% Optimize an equation

% Pseudocode for part 2 and 3