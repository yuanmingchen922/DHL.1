## Iteration Version 3

## This Markdown file is the iteration of model, objective function, constraints with detailed mathematics proof and few key notes from the theorem. My formula represents a mathematically rigorous Lagrangian dual of the facility-distribution optimization model

## Author: Mingchen Yuan, my2878@columbia.edu, mingcy@umich.edu, yuanmingchen922@gmail.com, @yuanmingchen922
## Github Link: "https://github.com/yuanmingchen922/DHL.1"


# My overall workflow is illurstrated as below 
Farkas Lemma⇒Weak Duality⇒Strong Duality⇒KKT Conditions⇒Primal–Dual Convergence

# Objective Function Version 3: A gengeral form 
$$
\boxed{
\max_{\lambda,\mu \ge 0} \; \min_{X,Z,Y} \;
\Bigg[
\sum_i f_m^i(1+\alpha_i)Y_i
+\sum_j f_d^j(1+\beta_j)Y_j
+\sum_{i,j}\Big(v_m^i+\frac{3}{2000}d^{MD}_{ij}(1+\gamma_{ij})\Big)X_{ij}
+\sum_{j,c}\frac{Z_{jc}}{6}\Big(p^j+9.75+3.5\frac{d^{DC}_{jc}}{500}\Big)(1+\delta_{jc})
+\sum_{i,j}\lambda_{ij}(X_{ij}-U_{ij}Y_i)
+\sum_{j,c}\mu_{jc}(Z_{jc}-V_{jc}Y_j)
\Bigg]
}$$

A more rigorous layered formatting 
1. Primal Problem: 
$$
\begin{aligned}
\min_{X,Z,Y} \quad & f(X,Z,Y) \\
\text{s.t.} \quad 
& g_{ij}(X,Y) = X_{ij} - U_{ij}Y_i \le 0, \\
& h_{jc}(Z,Y) = Z_{jc} - V_{jc}Y_j \le 0.
\end{aligned}
$$

2. Lagrange Function: 
$$
L(X,Z,Y;\lambda,\mu)
= f(X,Z,Y)+ \sum_{i,j}\lambda_{ij}(X_{ij}-U_{ij}Y_i)+ \sum_{j,c}\mu_{jc}(Z_{jc}-V_{jc}Y_j),
\quad \lambda,\mu \ge 0.
$$

3. Dual Functon: 
$$
G(\lambda,\mu)
= \min_{X,Z,Y} L(X,Z,Y;\lambda,\mu).
$$
The dual function represents the minimum value of the Lagrangian for fixed multipliers(\lambda,\mu), and it provides a lower bound on the optimal value of the primal problem by the Weak Duality Theorem

4. Dual Problem: 
$$
G(\lambda,\mu)
= \min_{X,Z,Y} L(X,Z,Y;\lambda,\mu).
$$
The dual problem seeks the tightest possible lower bound by optimizing over all nonnegative multipliers.
Under convexity and Slater’s condition, strong duality holds: 
$$
\min_{X,Z,Y} f(X,Z,Y) = \max_{\lambda,\mu \ge 0} G(\lambda,\mu)
$$



### Notation and Variables

- **Sets**

  $i, j \in I$: candidate manufacturing and distribution locations  
  $c \in C$: customers  

---

- **Decision Variables**

  $X_{ij} \ge 0$: flow of bottles from manufacturer *i* to distribution center *j*  
  $Z_{jc} \ge 0$: flow of bottles from distribution center *j* to customer *c*  
  $Y_i, Y_j \in [0,1]$: binary open/close decisions for manufacturing or DC sites  

---

- **Upper Bounds**

  $U_{ij} > 0,\quad V_{jc} = D_c > 0$  
  Used to replace the global Big-M constant with tighter local bounds.  

---

- **Parameters**

  $f_m^i, f_d^j, v_m^i, d_{ij}^{MD}, d_{jc}^{DC}, p^j$:  
  represent fixed and variable costs and transportation distances.  

---

- **Scenario Coefficients**

  $\alpha_i, \beta_j, \gamma_{ij}, \delta_{jc}$:  
  scenario perturbation or sensitivity coefficients .  

---

- **Multipliers**

  $\lambda_{ij} \ge 0,\quad \mu_{jc} \ge 0$:  
  outer-layer dual (Lagrange) multipliers.  

---

- **Variable Domains**

  Continuous variables must be non-negative:  
  $X_{ij}, Z_{jc} \ge 0$  

  Location var



### Model Constraints and Lagrangian Relaxation Notes

---

#### **(Demand Satisfaction)**

$$
\sum_j Z_{jc} = D_c, \quad \forall c \in C
$$

Each customer's demand must be satisfied.

---

#### **(Flow Balance)**

$$
\sum_c Z_{jc} = \sum_i X_{ij}, \quad \forall j \in I
$$

Each distribution center (DC) must maintain flow balance — total inbound equals total outbound.

---

#### **(Variable Domains)**

$$
X, Z \ge 0, \quad Y_i, Y_j \in \{0,1\}
$$

All flow variables are non-negative; location variables are binary (or relaxed to [0,1] in LP relaxation).

---

#### **(Linking Constraints to be Relaxed)**

$$
X_{ij} \le U_{ij} Y_i, \quad Z_{jc} \le V_{jc} Y_j
$$

These constraints link flow variables to facility-opening decisions.  
They will be **relaxed** in the Lagrangian formulation.

---

### Notes on Big-M Constraints and Lagrangian Relaxation

In **Version 1**, the Big-M linking constraints were written as:

$$
X_{ij} \le M Y_i, \quad Z_{jc} \le M Y_j
$$

However, in **Version 3**, we introduce **Lagrangian Relaxation**.

The linking constraints are **softened** — they are no longer treated as hard constraints inside the inner minimization problem.  
Instead, they are moved into the objective function with associated **Lagrange multipliers**:

$$
\max_{\lambda,\mu \ge 0}
\Bigg[
\sum_{i,j} \lambda_{ij}(X_{ij} - U_{ij}Y_i)+ \sum_{j,c} \mu_{jc}(Z_{jc} - V_{jc}Y_j)
\Bigg]
$$

The **maximization over** $(\lambda, \mu)$ penalizes any violation of the linking constraints.  
This makes the **inner minimization problem easier to solve**,  
since fewer hard constraints remain and the resulting problem becomes **separable** across indices $(i,j,c)$.

---

**Summary:**  
Lagrangian relaxation replaces rigid Big-M constraints with penalty terms,  
achieving computational efficiency and better decomposability for the inner optimization stage.


# Mathematical Proof for Optimality Conditions

The following parts show mathematical proofs for several conditions that must be satisfied in order to find the optimum among all feasible solutions.

All variables and constraints are stated earlier.  
Denote hard constraints (1–3) as **(H)** and the linking constraints included in the Lagrangian relaxation as **(LNK)**.

---

## **Proof:**

The proof consists of several parts.  
We first define the **Lagrangian Function** and the **Dual Function**,  
then establish **Weak Duality**, **Strong Duality**, and **Complementary Slackness**,  and finally analyze the **Structure of the Inner Minimization**.

---

### **1) Lagrangian and Dual Function**

#### **Lagrangian Function**

Introduce nonnegative multipliers $\lambda_{ij}, \mu_{jc} \ge 0$ for constraints (LNK).  
Then the Lagrangian is defined as:

$$
\begin{aligned}
L(X,Z,Y;\lambda,\mu)
&= \sum_i f_m^i(1+\alpha_i)Y_i+ \sum_j f_d^j(1+\beta_j)Y_j \\
&\quad + \sum_{i,j}\Big(v_m^i+\frac{3}{2000}d_{ij}^{MD}(1+\gamma_{ij})\Big)X_{ij} \\
&\quad + \sum_{j,c}\frac{Z_{jc}}{6}\Big(p^j+9.75+3.5\frac{d_{jc}^{DC}}{500}\Big)(1+\delta_{jc}) \\&\quad + \sum_{i,j}\lambda_{ij}(X_{ij}-U_{ij}Y_i)+ \sum_{j,c}\mu_{jc}(Z_{jc}-V_{jc}Y_j).
\end{aligned}
\tag{L}
$$

---

#### **Dual Function**

For given multipliers $(\lambda,\mu)$,

$$
g(\lambda,\mu)
= \inf_{(X,Z,Y)\text{ s.t. }(H)} L(X,Z,Y;\lambda,\mu).
\tag{g}
$$

This is called the **dual function** — it provides a lower bound to the primal objective value for any $\lambda,\mu \ge 0$.

---

### **Dual Problem (Outer Maximization)**

$$
\max_{\lambda \ge 0, \ \mu \ge 0} \; g(\lambda,\mu).
\tag{D}
$$

This outer maximization searches for the tightest (largest) lower bound among all nonnegative multipliers.

---

### **2) Weak Duality**

Assume $(\bar X,\bar Z,\bar Y)$ a feasible primal solution satisfying both (H) and (LNK).  
Then, by construction:

$$
L(\bar X,\bar Z,\bar Y;\lambda,\mu)
= \text{Cost}(\bar X,\bar Z,\bar Y)+ \sum_{i,j}\lambda_{ij}(\bar X_{ij}-U_{ij}\bar Y_i)+ \sum_{j,c}\mu_{jc}(\bar Z_{jc}-V_{jc}\bar Y_j).
$$

Because $\lambda,\mu \ge 0$ and $\bar X_{ij} \le U_{ij}\bar Y_i, \ \bar Z_{jc} \le V_{jc}\bar Y_j$,

$$
L(\bar X,\bar Z,\bar Y;\lambda,\mu) \le \text{Cost}(\bar X,\bar Z,\bar Y).
$$

Taking the infimum over all feasible primal points gives:

$$
\boxed{
g(\lambda,\mu) \le p^* = \min_{(H),(LNK)} \text{Cost}, \qquad
\max_{\lambda,\mu \ge 0} g(\lambda,\mu) \le p^*.
}
$$

This means every multiplier pair $(\lambda,\mu)$ gives a **valid lower bound** to the primal optimum $p^*$.

---

### **3) Strong Duality and Complementary Slackness — LP Relaxation**

If the binary variables $Y_i,Y_j$ are relaxed to continuous variables in $[0,1]$,  
and if the feasible region satisfies **Slater’s condition**, then it's very obvious that the problem is convex and strong duality:

$$
\min_{(H),(LNK)} \text{Cost}
= \max_{\lambda,\mu \ge 0} g(\lambda,\mu)
= L(X^*,Z^*,Y^*;\lambda^*,\mu^*),
$$

and **complementary slackness** holds:

$$
\lambda_{ij}^*(X_{ij}^* - U_{ij}Y_i^*) = 0, \quad
\mu_{jc}^*(Z_{jc}^* - V_{jc}Y_j^*) = 0.
\tag{CS}
$$

Thus, the saddle point $(X^*,Z^*,Y^*;\lambda^*,\mu^*)$ satisfies both **primal and dual optimality** simultaneously.

---

### **4) Structure of the Inner Minimization**

Expanding $L$ and grouping terms gives:

$$
\begin{aligned}
L &= 
\sum_i 
\underbrace{\Big(
f_m^i(1+\alpha_i)- \sum_j \lambda_{ij} U_{ij}
\Big)}_{\displaystyle =:\phi_i(\lambda)}
Y_i+ 
\sum_j 
\underbrace{\Big(
f_d^j(1+\beta_j)- \sum_c \mu_{jc} V_{jc}
\Big)}_{\displaystyle =:\psi_j(\mu)}
Y_j \\[8pt]
&\quad +
\sum_{i,j} 
\underbrace{\Big(
v_m^i + \frac{3}{2000} d_{ij}^{MD}(1+\gamma_{ij})+ \lambda_{ij}
\Big)}_{\displaystyle =:\tilde{c}_{ij}^X(\lambda)}
X_{ij} \\[8pt]
&\quad +
\sum_{j,c}
\underbrace{\frac{1}{6}
\Big(p^j + 9.75 + 3.5\frac{d_{jc}^{DC}}{500}
\Big)
(1+\delta_{jc})+ \mu_{jc}}_{\displaystyle =:\tilde{c}_{jc}^Z(\mu)}Z_{jc}.
\end{aligned}
$$


Given \((\lambda, \mu)\), the inner problem becomes:

$$
\min \sum_i \phi_i(\lambda)Y_i + \sum_j \psi_j(\mu)Y_j + \sum_{i,j}\tilde{c}^X_{ij}(\lambda)X_{ij} + \sum_{j,c}\tilde{c}^Z_{jc}(\mu)Z_{jc}
\quad \text{s.t. } (H)
$$


This decomposition makes the inner problem separable and computationally easier to solve.

---

#### (A) Optimal Structure for Y

Since in \((H)\), \(Y\) is independent from \(X, Z\) (as (LNK) is relaxed),  
we have:

$$
Y_i^*(\lambda) = \arg \min_{Y_i \in \{0,1\}} \phi_i(\lambda)Y_i =
\begin{cases}
1, & \phi_i(\lambda) < 0,\\
0, & \phi_i(\lambda) > 0,\\
\text{any of } \{0,1\}, & \phi_i(\lambda) = 0.
\end{cases}
$$

Hence,

$$
Y_i^*(\lambda) = 1\{\sum_j \lambda_{ij}U_{ij} > f_m^i(1+\alpha_i)\}
\tag{Y-MFG}
$$

Similarly,

$$
Y_j^*(\mu) = 1\{\sum_c \mu_{jc}V_{jc} > f_d^j(1+\beta_j)\}
\tag{Y-DC}
$$

---

#### (B) Optimal Structure for X and Z

The variables \((X,Z)\) form a **minimum-cost flow** problem with balanced constraints:

$$
\min \sum_{i,j} \tilde{c}^X_{ij}(\lambda)X_{ij} + \sum_{j,c}\tilde{c}^Z_{jc}(\mu)Z_{jc}
\quad \text{s.t. Demand / Balance / Nonnegativity}
$$

The KKT shortest-path conditions indicate that for optimal flows,  
if the reduced cost \(\tilde{c}^X_{ij}(\lambda)\) or \(\tilde{c}^Z_{jc}(\mu)\) is minimal on feasible arcs, then those arcs carry positive flow. 

---

### 5) Dual Ascent (Outer Maximization) — Subgradient Method

Assume \((X^*(\lambda,\mu), Z^*(\lambda,\mu), Y^*(\lambda,\mu))\) be the optimal solution of the inner problem for fixed multipliers \((\lambda, \mu)\).

Then the subgradients of the dual function \(g(\lambda,\mu)\) are:

$$
\partial_{\lambda_{ij}} g(\lambda,\mu) \ni X_{ij}^* - U_{ij}Y_i^*, \qquad
\partial_{\mu_{jc}} g(\lambda,\mu) \ni Z_{jc}^* - V_{jc}Y_j^*
$$

The **dual ascent (subgradient update)** is:

$$
\lambda_{ij}^{k+1} = \big[\lambda_{ij}^k + \tau_k (X_{ij}^k - U_{ij}Y_i^k)\big]_+, \qquad
\mu_{jc}^{k+1} = \big[\mu_{jc}^k + \tau_k (Z_{jc}^k - V_{jc}Y_j^k)\big]_+
$$

where \([\cdot]_+ = \max\{\,\cdot,0\,\}\), and \(\tau_k > 0\) is the step size, diminishing or Polyak step length.
When the LP relaxation satisfies strong duality, the above method guarantees monotonic convergence of \(g(\lambda^k,\mu^k)\) toward the optimal dual value.

---

### 6) KKT Conditions

Define the primal problem (P):

$$
\min \text{Cost}(X,Z,Y) \quad \text{s.t. } (H), (LNK).
$$

The Lagrangian and its dual follow as before.  
Under strong duality, there exists \((X^*,Z^*,Y^*;\lambda^*,\mu^*)\) is the optimal for all solutions iff the following are satisfied :

- **Primal feasibility:** satisfies (H), (LNK)  
- **Dual feasibility:** \(\lambda^*,\mu^* \ge 0\)  
- **Complementary slackness:**

  $$
  \lambda_{ij}^*(X_{ij}^* - U_{ij}Y_i^*) = 0, \quad
  \mu_{jc}^*(Z_{jc}^* - V_{jc}Y_j^*) = 0
  $$

- **Stationarity:**  
  Gradients of the Lagrangian w.r.t. all continuous variables vanish. Equivalently, reduced costs are non-negative in the minimum-cost flow.

Together, these KKT conditions are **necessary and sufficient** for optimality in the LP-relaxed problem.

Proof completed

---

### 7) Practical Optimality Test Formulas

#### **(a) Facility Opening Test**

$$
\sum_j \lambda_{ij}U_{ij}
\begin{cases}> f_m^i(1+\alpha_i) &\Rightarrow Y_i^* = 1,\\< f_m^i(1+\alpha_i) &\Rightarrow Y_i^* = 0,
\end{cases}
\qquad
\sum_c \mu_{jc}V_{jc}
\begin{cases}> f_d^j(1+\beta_j) &\Rightarrow Y_j^* = 1,\\< f_d^j(1+\beta_j) &\Rightarrow Y_j^* = 0.
\end{cases}
$$

---

#### **(b) Effective Arc Costs**

$$
\tilde{c}^X_{ij}(\lambda) = v_m^i + \frac{3}{2000}d^{MD}_{ij}(1+\gamma_{ij}) + \lambda_{ij},
\qquad
\tilde{c}^Z_{jc}(\mu) = \frac{1}{6}\Big(p^j + 9.75 + 3.5\frac{d^{DC}_{jc}}{500}\Big)(1+\delta_{jc}) + \mu_{jc}.
$$

Flows occur only on the arcs with minimum reduced cost  
that satisfy **Demand** and **Balance** constraints.

---

### **(c) Dual Ascent Update**

Use the subgradient formula directly from SubGrad:  
update dual variables using the rule in Ascent until convergence.

---

# 8) MILP vs LP Relaxation

- **MILP (with binary \(Y \in \{0,1\}\):**  
  The dual optimum \(\max g\) gives a **valid lower bound** of the primal objective.  
  This can be used for branch-and-bound or evaluating solution quality.

- **LP Relaxation (with \(Y \in [0,1]\):**  
  When convexity and Slater’s condition hold, **strong duality** exists.  
  The saddle point satisfies KKT and complementary slackness, providing both necessary and sufficient optimality conditions.

---

## **Summary**


$$
\max_{\lambda,\mu \ge 0} \; \min_{(X,Z,Y) \text{ s.t. } (H)} L(X,Z,Y;\lambda,\mu)
$$

Performing Lagrangian relaxation on the linking constraints (LNK)  
transforms the problem into a **dual maximization** framework.

- **Weak Duality** ensures that any multiplier pair \((\lambda,\mu)\) provides a lower bound on the original problem.  
- Under **LP relaxation**, **strong duality** and **complementary slackness** make the dual and primal optima equal and verifiable.  
- The **inner minimization** becomes separable:  
  - Facility decisions follow sign tests of \(\phi_i(\lambda)\) and \(\psi_j(\mu)\).  
  - Flow variables follow a minimum-cost flow rule.
- The **outer maximization** is performed via subgradient ascent using:
  $$
  (X^* - UY^*, \; Z^* - VY^*)
  $$
  as the direction of improvement.

This yields a verifiable, computationally efficient optimality framework for the DHL network optimization model.



