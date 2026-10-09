# THEORY.md — statements, proofs, and the tests that check them

This is the mathematical core of RSC-Rank. Every theorem here has (i) a statement with all assumptions,
(ii) a proof, (iii) the test that falsifies it on thousands of small instances. The LaTeX in
`paper/sections/04_method.tex`, `05_guarantees.tex`, `90_appendix_proofs.tex`, `91_appendix_extra.tex` must match this file (same
numbering). Notation follows the paper.

## 0. Setting and assumptions

**Every statement in this file assumes the setting of this section:** a fixed pointwise teacher $F$, a fixed candidate
set $C(q)$ with $N\ge2$ candidates, $1\le k<N$, cheap predictions $\mu_i\in\mathbb R$ with scales $\sigma_i>0$, the
*strict* stop rule below, the total order with ties broken by ascending identifier, and $1\le r\le\min(k,N-k)$ wherever
$r$ appears. Statements that need an additional assumption (exchangeability, iid, random split) say so.

- A query $q$ with a fixed candidate set $C(q)=\{d_1,\dots,d_N\}$ produced by a fixed first-stage generator $G$; $1\le k<N$.
- A fixed **pointwise** teacher $F$: $s_i=F(q,d_i)\in\mathbb{R}$ depends only on $(q,d_i)$ (not on the other candidates).
- **Total order.** Candidates are ordered by $(s_i\downarrow,\ \mathrm{id}_i\uparrow)$; $s_{(1)}\ge s_{(2)}\ge\dots\ge s_{(N)}$ are the ordered scores and $T_k(q)$ the set of the first $k$ candidates; $\mathcal T_k(q)$ the ordered list.
- **Cheap predictions.** $\mu_i\in\mathbb R$ and $\sigma_i>0$, computed from cheap information only (query, candidate, cheap model). In the main implementation $\sigma_i=\sigma(q)$ is shared across the query.
- **Guardrail.** $U_i(\lambda)=\mu_i+\lambda\sigma_i$, $\lambda\in\mathbb R\cup\{+\infty\}$.
- **Stopping rule (strict).** For a scored set $V\subseteq C(q)$ with $|V|\ge k$, let $\tau(V)=\mathrm{kthmax}_{i\in V}s_i$. Stopping is allowed iff $\tau(V)>\max_{j\notin V}U_j(\lambda)$ (with $\max\emptyset=-\infty$, so $V=C(q)$ always stops and returns the full teacher order). The output is the ordered top-$k$ of $V$ under the total order.
- **Legal policies.** A policy chooses which unscored candidates to score next using only cheap information and the teacher scores already observed; batches of any size; the scored set is nested $V_t\subseteq V_{t+1}$.
- **Losses.** $\ell_1^\pi(q)=\mathbf 1\{\widehat{\mathcal T}^\pi_k(q)\neq\mathcal T_k(q)\}$ (ordered-list mismatch); $m^\pi(q)=|T_k(q)\setminus\widehat T^\pi_k(q)|$; $\ell_r^\pi(q)=\mathbf 1\{m^\pi(q)\ge r\}$ for $1\le r\le\min(k,N-k)$.
- **Unsafe sets.** $\mathcal U_{k,r}(q)=\{V\subsetneq C(q): |V|\ge k,\ |T_k(q)\setminus V|\ge r\}$. A stop at $V$ returns the ordered top-$k$ of $V$; every element of $T_k\cap V$ is in that top-$k$ (it beats every element outside $T_k$ in the total order), so the number of winners missed by a stop at $V$ is exactly $m^\pi(q)=|T_k\setminus V|$. Hence a stop at $V\in\mathcal U_{k,r}$ misses at least $r$ winners ($\ell_r^\pi=1$), and a stop at $V\notin\mathcal U_{k,1}$ contains all of $T_k$ and returns exactly $\mathcal T_k(q)$ (the ordered top-$k$ of a superset of $T_k$ under the same total order is $\mathcal T_k$).

Two elementary facts used repeatedly:

**Fact A.** If $|V|\ge k$ and $V$ contains at most $k-r$ elements of $T_k$ (i.e. $|T_k\setminus V|\ge r$), then $\tau(V)\le s_{(k+r)}$.
*Proof.* The $k$-th largest element of $V$ (in the total order) has overall rank $\ge k+r$ because at most $k-r$ elements of $V$ have rank $\le k$; scores are non-increasing in rank. $\square$

**Fact B.** If $V=C(q)\setminus W$ with $W\subseteq T_k$, $|W|=r$, then $\tau(V)=s_{(k+r)}$.
*Proof.* Removing $r$ elements of rank $\le k$ shifts the $k$-th position of the remaining order to overall rank exactly $k+r$. $\square$

**Lemma C (normalised form).** For $|V|\ge k$ and $V\ne C(q)$ define $\lambda^*(V)=\min_{j\notin V}(\tau(V)-\mu_j)/\sigma_j$. Then the stopping rule holds at $\lambda$ iff $\lambda<\lambda^*(V)$.
*Proof.* $\tau(V)>\mu_j+\lambda\sigma_j\iff\lambda<(\tau(V)-\mu_j)/\sigma_j$ for each $j$ since $\sigma_j>0$; the rule requires it for all $j\notin V$. $\square$

## 1. Theorem 1 (tight rank-sufficient critical value, $r=1$)

Define $A_{k,1}(q)=\max_{i\in T_k(q)}\dfrac{s_{(k+1)}-\mu_i}{\sigma_i}$.

**Theorem 1.** $\lambda\ge A_{k,1}(q)$ iff no $V\in\mathcal U_{k,1}(q)$ satisfies the stopping rule. Equivalently
$\max_{V\in\mathcal U_{k,1}(q)}\lambda^*(V)=A_{k,1}(q)$.

*Proof (sufficiency).* Let $\lambda\ge A_{k,1}$ and $V\in\mathcal U_{k,1}$; pick $w\in T_k\setminus V$. By definition $\lambda\ge(s_{(k+1)}-\mu_w)/\sigma_w$, so $U_w(\lambda)\ge s_{(k+1)}\ge\tau(V)$ (Fact A with $r=1$). Hence $\max_{j\notin V}U_j\ge U_w\ge\tau(V)$ and the strict rule fails.
*(Necessity.)* Let $\lambda<A_{k,1}$ and let $w\in T_k$ attain the max, so $U_w(\lambda)<s_{(k+1)}$. Take $V=C\setminus\{w\}\in\mathcal U_{k,1}$ ($|V|=N-1\ge k$). By Fact B, $\tau(V)=s_{(k+1)}>U_w=\max_{j\notin V}U_j$: the rule holds and the stop misses $w$. The equivalent max-form follows from Lemma C: "no unsafe $V$ stops at $\lambda$" iff $\lambda\ge\lambda^*(V)\ \forall V$ iff $\lambda\ge\max_V\lambda^*(V)$; both thresholds are finite reals, so they coincide. $\square$

*Test:* `tests/test_theorems.py::test_closed_form_equals_bruteforce_max_over_unsafe_sets` (exact float equality with the exponential brute force), `test_iff_exact_rationals`, `test_sufficiency_direction_with_ties_at_boundary`, `test_necessity_direction_realised_by_a_legal_policy`.

## 2. Theorem 2 (the $r$-miss frontier)

Define $A_{k,r}(q)=\mathrm{rthmax}_{i\in T_k(q)}\dfrac{s_{(k+r)}-\mu_i}{\sigma_i}$ for $1\le r\le\min(k,N-k)$.

**Theorem 2.** $\lambda\ge A_{k,r}(q)$ iff no $V\in\mathcal U_{k,r}(q)$ satisfies the stopping rule; equivalently $\max_{V\in\mathcal U_{k,r}}\lambda^*(V)=A_{k,r}(q)$. Moreover $A_{k,r+1}\le A_{k,r}$.

*Proof.* Let $b=s_{(k+r)}$ and $z_i=(b-\mu_i)/\sigma_i$ for $i\in T_k$. (Sufficiency) If $\lambda\ge$ $r$-th largest $z_i$, at most $r-1$ winners have $z_i>\lambda$, i.e. $U_i(\lambda)<b$. Any $V\in\mathcal U_{k,r}$ misses $\ge r$ winners, hence at least one *missed* winner $w\notin V$ has $U_w\ge b\ge\tau(V)$ (Fact A); then $\max_{j\notin V}U_j\ge U_w\ge\tau(V)$ and the strict rule fails. (Necessity) If $\lambda<A_{k,r}$, at least $r$ winners have $U_i<b$; let $W$ be $r$ of them and $V=C\setminus W$ ($|V|=N-r\ge k$ by $r\le N-k$). By Fact B, $\tau(V)=b>\max_{j\in W}U_j=\max_{j\notin V}U_j$: an unsafe stop. The max-form $\max_{V\in\mathcal U_{k,r}}\lambda^*(V)=A_{k,r}$ follows from Lemma C exactly as in Theorem 1 (both sides are finite reals characterising the same set of $\lambda$). Monotonicity: $s_{(k+r+1)}\le s_{(k+r)}$ decreases every $z_i$, and the $(r+1)$-th largest of a vector is $\le$ its $r$-th largest. $\square$

*Test:* `test_closed_form_equals_bruteforce_max_over_unsafe_sets` (all valid $r$), `test_iff_exact_rationals`, `test_frontier_monotone_and_dominance`, `tests/test_scale.py::test_r_miss_frontier_at_scale`.

## 3. Proposition 3 (dominance and same-path call dominance)

Define $B_{\mathrm{top}}(q)=\max_{i\in T_k}(s_i-\mu_i)/\sigma_i$ and $B_{\mathrm{all}}(q)=\max_{i\in C}(s_i-\mu_i)/\sigma_i$.

**Proposition 3.** (a) $A_{k,1}(q)\le B_{\mathrm{top}}(q)\le B_{\mathrm{all}}(q)$ for every query. (b) With the same calibration set and the same order-statistic index, $\lambda_A\le\lambda_{\mathrm{top}}\le\lambda_{\mathrm{all}}$. (c) For a policy whose scoring order does not depend on $\lambda$ (e.g. $\mu$-descending with a fixed batch schedule), the number of teacher calls is non-decreasing in $\lambda$; hence $\mathrm{Calls}_A\le\mathrm{Calls}_{\mathrm{top}}\le\mathrm{Calls}_{\mathrm{all}}$ query by query.

*Proof.* (a) $s_i\ge s_{(k+1)}$ for $i\in T_k$ gives the first inequality termwise; the second enlarges the max. (b) Order statistics are monotone in the pointwise order of the samples. (c) For a fixed scored set $V$ the strict rule holds iff $\lambda<\lambda^*(V)$ (Lemma C), a down-set in $\lambda$: a larger $\lambda$ raises every guardrail, so the set of rounds at which the rule holds *shrinks* (is a subset of the previous one), and the first stopping round can only move later. $\square$
*Test:* `test_frontier_monotone_and_dominance` (a); `test_order_statistics_monotone_in_pointwise_order_prop3b` (b, all three regimes); `tests/test_replay_metrics.py::test_same_path_call_dominance_and_risk` (c); the experiment check `dominance_*` in every replay summary.

Note that $B_{\mathrm{top}}$ and $B_{\mathrm{all}}$ certify *score coverage* ($s_i\le U_i(\lambda)$ for the covered set); Theorem 1 shows coverage is sufficient but not necessary for rank protection, and the gap $B_{\mathrm{top}}-A_{k,1}=\max_i(s_i-\mu_i)/\sigma_i-\max_i(s_{(k+1)}-\mu_i)/\sigma_i$ is driven by the margin $s_i-s_{(k+1)}$ of the winners.

## 4. Proposition 4 (the cost of policy-uniformity)

**Proposition 4.** Let $s_i=1,\mu_i=-c$ for $i\in T_k$ and $s_j=0,\mu_j=0$ for $j\notin T_k$, all $\sigma=1$, $c>1$. Then $A_{k,1}=c$; every $\lambda\ge A_{k,1}$ forces $N$ teacher calls under *every* legal policy, while the winners-first policy stops correctly after exactly $k$ calls with any $\lambda<1$ (its trajectory-specific critical value of Sec. 5 is $A^\pi_{k,1}=-\infty$, so every finite $\lambda\ge A^\pi_{k,1}$ protects it). The ratio of calls is $N/k$.

*Proof.* $A_{k,1}=\max_{i\in T_k}(0-(-c))/1=c$. Let $\lambda\ge c$ and let $V\subsetneq C$ with $|V|\ge k$ be any eligible proper scored set; all scores are $\le1$, so $\tau(V)\le1$ (and $\tau(V)=0$ whenever $V$ holds fewer than $k$ winners). *Case 1: some non-winner $j$ is unscored.* Then $U_j(\lambda)=\lambda\ge c>1\ge\tau(V)$ and the strict rule fails. *Case 2: every non-winner is scored, so some winner $w$ is unscored.* Then $V$ holds at most $k-1$ winners, $\tau(V)=0$, and $U_w(\lambda)=\lambda-c\ge0=\tau(V)$, so the strict rule fails again (equivalently: $V\in\mathcal U_{k,1}$ and $\lambda\ge A_{k,1}$, so Theorem 1 forbids the stop). Hence no proper scored set ever stops and every legal policy makes $N$ calls. For the winners-first order with a first batch of size $k$, every eligible prefix ($|V|\ge k$) already contains $T_k$, so *no* eligible prefix is unsafe and the trajectory-specific threshold of Sec. 5 is $A^\pi_{k,1}=-\infty$; for any $\lambda<1$ the controller stops at $V=T_k$ because $\tau(V)=1>\lambda=\max_{j\notin V}U_j(\lambda)$: $k$ calls and the correct list. $\square$
*Test:* `test_policy_uniform_cost_counterexample_N_over_k`.

This is a lower-bound *example*, not a worst case over all instances; it shows that policy-uniform protection is not free and motivates the fixed-trajectory baseline.

## 5. Trajectory-specific critical value (baseline)

For a fixed policy $\pi$ with nested prefixes $V_1\subset\dots\subset V_m=C$, define $A^\pi_{k,r}(q)=\max\{\lambda^*(V_t): V_t\in\mathcal U_{k,r}\}$ ($-\infty$ if no prefix is unsafe).

**Proposition 5.** $A^\pi_{k,r}(q)\le A_{k,r}(q)$; $\lambda\ge A^\pi_{k,r}(q)$ iff the controller following $\pi$ never stops at an unsafe prefix; the same $\lambda$ gives no protection for a different policy in general.
*Proof.* Prefixes are a subset of the unsafe sets, so the max is smaller. For the iff: by Lemma C, $\lambda\ge A^\pi$ iff no unsafe prefix satisfies the rule. Since prefixes are nested, $|T_k\setminus V_t|$ is non-increasing in $t$; hence every eligible prefix that precedes an $r$-unsafe prefix ($|T_k\setminus V_t|\ge r$) is itself $r$-unsafe (for $r=1$: a safe prefix contains $T_k$ and so do all its supersets). Hence if some $r$-unsafe prefix satisfies the rule, the *first* round at which the rule holds is an $r$-unsafe prefix and the controller stops there missing $\ge r$ winners; if none does, every stop misses fewer than $r$ winners. The last claim is witnessed by Prop. 4 and by `test_trajectory_threshold_below_A_and_protects_only_its_own_trajectory` (both directions of the iff are tested there, including $\lambda$ just below $A^\pi$). $\square$

## 5b. Price of policy-uniformity (definition)

For the deployed policy $\pi$ with its checkpoint schedule, $\mathrm{PoU}(q)=A_{k,1}(q)-A^\pi_{k,1}(q)\ge 0$ is the
*price of policy-uniformity* in threshold units; its calibrated counterpart is $\lambda_A-\lambda_{A^\pi}$ and its
operational counterpart is the difference in teacher calls (experiment E9 reports all three).  $\mathrm{PoU}=0$ on a
query means uniform protection is free there; Prop. 4 shows it can multiply the calls by $N/k$ (a difference of $N-k$ calls).

## 5c. Controllers that depend on $\lambda$ (paper Proposition 5)

A trajectory critical value (Sec. 5) is valid for a trajectory that does not depend on $\lambda$. A schedule that chooses
its batches from the guardrails (the blocker schedule, requiring $k\le b_0\le N$ and $b\ge 1$: first $b_0$ documents in $\mu$ order, then in one round every
unscored $j$ with $(\tau(V)-\mu_j)/\sigma_j\le\lambda$, padded to at least $b$ along the same order) has a trajectory that changes with $\lambda$; the
current blockers are what keeps this state from stopping, not the documents every later certified stop must score ($V^\star$); the
critical value of a trajectory generated at a reference threshold $\lambda_0$ does not protect the schedule re-run at
$\lambda\ne\lambda_0$. Counterexample: $k=1$, $s=(0,1,10)$, $\mu=(10,-5,-10)$, $\sigma\equiv1$, $b_0=b=1$; at $\lambda_0=11$ the
trajectory is $\{d_1\},\{d_1,d_2,d_3\}$ with value $5$; at $\lambda=6$ the controller stops at $\{d_1,d_2\}$ and misses $d_3$.

**Proposition ($\lambda$-dependent controllers).** If controllers $(\pi_\lambda)_\lambda$ err on $q$ iff $\lambda<a(q)$ for a query
statistic $a$, then $a(q)\le A_{k,1}(q)$ and Theorems 6–8 hold for the family with $a$ in place of $A_{k,r}$. For the blocker
schedule with a query-shared $\sigma$ and an eligible first round ($k\le b_0$), with $m_T$ the $\mu$-rank of the last winner, $\tau_0$ the $k$-th score of the first
round and $\mu_{[m]}$ the $m$-th largest prediction (undefined for $b_0<k$; the $k$-grid does not use it): $a^{\mathrm{blk}}=-\infty$ if $m_T\le b_0$; $\lambda^*(P_{b_0})$ if
$b_0<m_T\le b_0+b$; $(\tau_0-\mu_{[m_T]})/\sigma$ otherwise. *Proof.* With a shared $\sigma$ the blockers are the first unscored
documents in $\mu$ order; after the second round every remaining document has $(\tau_0-\mu_j)/\sigma>\lambda$ and $\tau$ can
only grow, so the rule holds; that stop is correct iff the second round reaches $\mu$-rank $m_T$. $\square$
Implementation: `rscrank.core.blocker_critical_value`. Tests: `tests/test_blocker_critical.py` (the counterexample;
re-execution of the controller at every breakpoint of $\lambda$ on random instances with ties); on real data,
`scripts/18_order_free_baselines.py` re-executes the controller at the value and one ulp below on every test query.

*Numbering.* The paper numbers its results in reading order; this file keeps its own numbering.

## 6. Proposition 11 (offline call lower bound)

**Proposition 11.** $C^\star_{\mathrm{cert}}(q)=|T_k\cup\{j:U_j(\lambda)\ge s_{(k)}\}|$ is a lower bound on the calls of any correctly certified early stop: a correct stop has $\tau=s_{(k)}$ and requires all $j$ with $U_j\ge s_{(k)}$ to be scored (else the rule fails) and all of $T_k$ scored (else it is not correct). *Test:* `test_cert_lower_bound_is_a_lower_bound_on_calls_of_correct_early_stops`. (In the paper this statement is Proposition 7(a), "Cost Anatomy".)

## 6b. Proposition 10 (cost anatomy: what the calls are made of, and how close to optimal they are)

Fix a query with $\lambda\ge A_{k,1}(q)$ (a *covered* query; by Thm. 1 every stop is then correct).  Write
$\theta=s_{(k)}$, $U_j=U_j(\lambda)$ and
$$V^\star=T_k\cup\{j: U_j\ge\theta\},\qquad \beta(q)=\#\{j\notin T_k:\ U_j<\theta\ \text{and } j \text{ precedes the last winner in } U\text{-descending order}\}.$$

**Proposition 10.** (a) *Lower bound.* Every legal policy that stops early scores at least $|V^\star|$ documents, and
$V^\star$ itself satisfies the stopping rule (when $|V^\star|<N$): $|V^\star|=C^\star_{\mathrm{cert}}(q)$ is the
instance-optimal cost at this $\lambda$.  (b) *Blocker-first.* The policy that scores documents one at a time in
decreasing $U_j$ (ties by identifier; with a shared $\sigma$ and finite $\lambda$ this is exactly the $\mu$-descending main policy, and the implementation orders by $(\mu_j\downarrow,\mathrm{id}\uparrow)$ in that case so that floating-point ties of $\mu_j+\lambda\sigma$ cannot reorder distinct $\mu$) stops
after exactly $\min(N,\ |V^\star|+\beta(q))$ calls.  (c) *Batching.* With checkpoints $c_1<c_2<\dots$ the cost is the
first checkpoint $\ge$ that number, i.e. at most $\max_i b_i-1$ extra calls.

*Proof.* (a) At a stop $V$ of a covered query, $T_k\subseteq V$ (Thm. 1), hence $\tau(V)=\theta$, and every unscored
$j$ has $U_j<\tau(V)=\theta$; therefore $V\supseteq T_k\cup\{j:U_j\ge\theta\}=V^\star$.  Conversely $\tau(V^\star)=\theta$
and all $j\notin V^\star$ have $U_j<\theta$, so the strict rule holds at $V^\star$.  (b) Let $u_{(1)}\ge u_{(2)}\ge\dots$
be the $U$-descending order and $P_m$ its prefixes.  For $k\le m<N$, $\max_{j\notin P_m}U_j=u_{(m+1)}$ (at $m=N$ the stop is automatic), so a covered
query stops at the first $m$ with $T_k\subseteq P_m$ *and* $u_{(m+1)}<\theta$, i.e. at
$m^\star=\max(m_T,m_\theta)$ where $m_T$ is the position of the last winner and $m_\theta=\#\{j:U_j\ge\theta\}$.  If
$m_T\le m_\theta$ every winner has $U\ge\theta$, so $|V^\star|=m_\theta=m^\star$ and $\beta=0$.  Otherwise
$m^\star=m_T$ and $m_T-|V^\star|$ counts exactly the non-winners in positions $(m_\theta,m_T]$, which is $\beta(q)$.
(c) The controller only evaluates the rule at checkpoints $c_1<c_2<\dots$ (cumulative batch sizes, $b_i=c_i-c_{i-1}$), and the rule, once true at $m^\star$, stays true along the same order: scoring more documents can only raise $\tau$ (the $k$-th largest of a growing set) and can only remove terms from $\max_{j\notin V}U_j$. The cost is therefore the first checkpoint $\ge m^\star$, at most $\max_i b_i-1$ above it. $\square$

**Reading.** $|V^\star|$ is set by the calibrated $\lambda$ and the quality of $\mu$ on *non-winners* (how many
guardrails reach the $k$-th score); $\beta$ is set by the quality of $\mu$ on *winners* (how late a winner appears in
the cheap order); the rest is batching.  The gap between the measured calls and $|V^\star|$ is therefore the most a
smarter scheduler could ever save at the same threshold -- our main policy is within $\beta+$ batching of instance
optimal.  *Test:* `tests/test_theorems.py::test_cost_anatomy_predicts_the_controller_exactly`,
`::test_cert_core_is_a_stopping_state_and_a_lower_bound`,
`::test_cost_anatomy_uses_the_controllers_normalised_comparison_at_lambda_eq_A` (membership of $V^\star$ is decided by the same normalised comparison as the controller, Sec. 8),
`::test_cost_anatomy_and_policies_share_the_identifier_tie_rule` (the "ties by identifier" clause of (b): the
$U$-descending order of the anatomy and the $\mu$ policy use the same natural identifier order, checked with numeric
identifiers and exactly tied $\mu$); experiment E9 reports the decomposition on real data.

## 7. Statistical guarantees

Let $a(q)=A_{k,r}(q)$ be the query-level statistic, computed offline on calibration queries $q_1,\dots,q_n$ with the frozen system; $a_{(1)}\le\dots\le a_{(n)}$, $a_{(n+1)}=+\infty$. The protected event $E(\lambda)=\{A_{k,r}(Q)\le\lambda\}$ is deterministic given the query and, by Theorem 2, implies $\ell^\pi_r(Q)=0$ **for every legal policy simultaneously**.

**Theorem 6 (marginal, exchangeable).** With $j=\lceil(n+1)(1-\alpha)\rceil$ and $\lambda=a_{(j)}$: if $(q_1,\dots,q_n,Q)$ are exchangeable, $\Pr[\exists\pi:\ell^\pi_r(Q)=1]\le\Pr[A_{k,r}(Q)>\lambda]\le\alpha$ (probability over calibration set and $Q$). *Proof.* Standard split-conformal argument: when the $n+1$ values are distinct, the rank of $A(Q)$ among them is uniform on $\{1,\dots,n+1\}$ by exchangeability; $A(Q)>a_{(j)}$ means its rank exceeds $j$, probability $\le 1-j/(n+1)\le\alpha$. With ties, break them by an independent uniform permutation: the strict event $A(Q)>a_{(j)}$ is contained in the tie-broken event "rank $>j$", so the bound is conservative. $\square$ Tests: `test_marginal_index_basic` (index), `test_marginal_coverage_exhaustive_over_ranks` (the coverage bound itself by exhaustive enumeration of the $n+1$ positions, distinct values and ties).

**Theorem 7 (PAC / training-conditional, iid).** Let $j_{\mathrm{pac}}=\min\{j\in\{1..n+1\}:\Pr[\mathrm{Bin}(n,1-\alpha)\ge j]\le\delta\}$ and $\lambda=a_{(j_{\mathrm{pac}})}$. If calibration and future queries are iid, then with probability $\ge1-\delta$ over the calibration draw, $\sup_\pi\Pr_Q[\ell^\pi_r(Q)=1\mid D_{\mathrm{cal}}]\le\Pr_Q[A(Q)>\lambda\mid D_{\mathrm{cal}}]\le\alpha$. *Proof.* Let $q^\star$ be a $(1-\alpha)$-quantile of $A$'s law, so $\Pr[A>q^\star]\le\alpha$ and $\Pr[A<q^\star]\le1-\alpha$. The threshold fails ($\Pr[A>a_{(j)}\mid D_{\mathrm{cal}}]>\alpha$) only if $a_{(j)}<q^\star$, i.e. at least $j$ of the $n$ calibration values lie strictly below $q^\star$; each does so with probability at most $1-\alpha$, so this is an upper-tail event of probability $\le\Pr[\mathrm{Bin}(n,1-\alpha)\ge j]\le\delta$ by the choice of $j$ (the argument holds for every law, discrete or continuous). When it does not fail, $\Pr[A>a_{(j)}]\le\Pr[A>q^\star]\le\alpha$. $\square$ (Wilks-type tolerance limit.) Test: `test_pac_index_definition_and_table`, `test_pac_guarantee_by_simulation`.

**Theorem 8 (finite population, random split).** A fixed pool of $L=n+M$ queries is split uniformly at random into $n$ calibration and $M$ test queries, with all statistics fixed before the split. Let $t=\lfloor\alpha M\rfloor+1$ and $p_j=\Pr[X_j\ge j]$, $X_j\sim\mathrm{Hypergeom}(L,\,M+j-t,\,n)$ (population $L$, "successes" $M+j-t$, draws $n$), (under the standing assumptions $0<\alpha<1$, $M\ge1$ one has $t\le M$, hence $1\le j\le M+j-t\le L$ and the distribution is always well defined). Choose the smallest $j$ with $p_j\le\delta$ (or $j=n+1$, $\lambda=+\infty$). Then $\Pr_{\mathrm{split}}\big[\sup_\pi\frac1M\sum_{q\in D_{\mathrm{test}}}\ell^\pi_r(q)\le\alpha\big]\ge1-\delta$.
*Proof.* Assume distinct values (ties: break them by an independent uniform permutation; the number of test values *strictly* above $a_{(j)}$ is at most the number above the tie-broken position, so the bound is conservative). Let $J$ be the pool rank (ascending) of the $j$-th smallest calibration value. Test values above $a_{(j)}$: $E=M+j-J$ (all pool elements above position $J$ minus the $n-j$ calibration ones). Failure $E\ge t\iff J\le M+j-t\iff$ at least $j$ calibration elements lie among the $M+j-t$ smallest pool elements; the count of calibration elements among a fixed subset of size $M+j-t$ is hypergeometric as stated. $\square$ Test: `test_hypergeometric_identity_exhaustive_distinct_values` (exact equality of the enumerated failure probability and $p_j$), `test_hypergeometric_bound_conservative_with_ties`, `test_finite_index_guarantee_holds_exhaustively`.

*Numbers (main configuration, $\alpha=0.05$, $\delta_{\mathrm{cfg}}=0.05/4$):* $n=1000,M=2452$: indices 951 / 966 / 968; $n=2000,M=5405$: 1901 / 1922 / 1925 (marginal / PAC / finite). Test: `test_finite_population_index_table`.

*Implementation convention (rscrank.calibrate).* The index arithmetic is exact: $\alpha$ is converted to the decimal rational it denotes (`Fraction(repr(alpha))`, e.g. $0.05\mapsto 1/20$) and $j_{\mathrm{marg}}=\lceil(n+1)(1-\alpha)\rceil$, $t=\lfloor\alpha M\rfloor+1$ are evaluated on `Fraction`s with integer ceiling/floor, so products that are integers in exact arithmetic (e.g. $0.05\cdot2000=100$) are never mis-rounded, for every $n$, $M$ and every decimal $\alpha$ (test `test_index_arithmetic_is_exact_for_decimal_alpha`). The binomial and hypergeometric tails themselves are evaluated in float64 by SciPy (an engineering remark, not part of the theorems): for the reported configurations the tails were re-derived with exact big-integer arithmetic and agree with SciPy to $6.5\cdot10^{-16}$, while $|p_j-\delta|\ge4.5\cdot10^{-5}$ at every decisive index of the reported configurations (the smallest margins are the MS MARCO finite-population index at $\alpha=0.01$, $4.5\cdot10^{-5}$, and the $n=2000$ PAC index at $\alpha=0.1$, $4.8\cdot10^{-5}$), so the float evaluation cannot change an index; other $(n,M,\alpha,\delta)$ would need the same check.

**What is not guaranteed (Sec. 9 of the plan).** Not per-query conditional risk; not the conditional risk given early stop ($\Pr[\text{err}\mid E]\le\alpha/\Pr[E]$ only); nothing after a change of $F,G,\mu,\sigma,N,k$, prompt, tokenizer, truncation or a numerical backend that changes the teacher order; nothing across domains (cross-domain results are stress tests); no relevance improvement over the teacher (only the gap bound below).

**Proposition 9 (metric gap).** For any metric $\Phi\in[0,1]$ that depends only on the ordered top-$k$ (with $\Phi_F$, $\Phi_{\mathrm{RSC}}$ the metric of the full teacher's list and of the returned list): $|\Phi_F(q)-\Phi_{\mathrm{RSC}}(q)|\le\ell_1(q)$ for every query. Consequently, when $\lambda$ is calibrated from the $r=1$ statistic $A_{k,1}$: under Theorem 6, $\mathbb E|\Phi_F-\Phi_{\mathrm{RSC}}|\le\alpha$ (expectation over calibration set and $Q$); under Theorem 7, $\mathbb E[|\Phi_F-\Phi_{\mathrm{RSC}}|\mid D_{\mathrm{cal}}]\le\alpha$ with probability $\ge1-\delta$ over the calibration draw; under Theorem 8, $\frac1M\sum_{q\in D_{\mathrm{test}}}|\Phi_F(q)-\Phi_{\mathrm{RSC}}(q)|\le\alpha$ with probability $\ge1-\delta$ over the split ($M$ is the test-set size). *Proof.* If $\ell_1=0$ the lists are identical and the metrics coincide; otherwise the difference is at most 1; average the pointwise bound in the respective regime. $\square$ Test: `test_metric_gap_bound`.

## 8. Implementation note (floating point, part of the theory)

`rscrank.core.can_stop` evaluates the rule in normalised units: stop iff $\lambda<\lambda^*(V)$ computed as `min((tau - mu_j)/sigma_j)`. IEEE rounding is monotone in each argument, so for a $V\in\mathcal U_{k,r}$ with missing winners $W$ ($|W|\ge r$), using $\tau(V)\le s_{(k+r)}$ (Fact A, an exact selection of stored floats, no arithmetic): `fl(fl(tau - mu_w)/sigma_w) <= fl(fl(s_(k+r) - mu_w)/sigma_w)` for every $w\in W$, hence `min_{j notin V} fl(...) <= min_{w in W} fl(fl(s_(k+r) - mu_w)/sigma_w) <= rthmax_{i in T_k} fl(fl(s_(k+r) - mu_i)/sigma_i) = A_float` (the minimum over $\ge r$ of the winners' values is at most their $r$-th largest), where `A_float` is computed by `critical_value` with the same expression on the same float64 inputs. Therefore `lambda >= A_float` can never produce an $r$-unsafe stop in the implementation, and `lambda < A_float` produces one on the witness set $C\setminus W^*$ of the necessity proof — the float controller is *exactly* consistent with the float statistic, for every $r$. Two conditions are part of this statement: both sides must be evaluated in float64 on the *same* stored inputs (`query_statistics` and the controller both go through `np.asarray(..., dtype=float)`), and with the identical expression `(a - mu)/sigma`. (Evaluating the rule as `tau > mu_j + lambda*sigma_j` would not have this property: `mu + A*sigma` can round below `s_(k+1)`.) Tests: `test_controller_is_safe_at_or_above_A_for_random_policies_and_schedules` runs at $\lambda=A_{k,1}$ exactly; `test_controller_is_safe_at_A_kr_for_r_ge_2` runs at $\lambda=A_{k,r}$ exactly for $r\ge2$ and checks that $\lambda=\mathrm{nextafter}(A_{k,r},-\infty)$ errs on the witness.
