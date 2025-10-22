## Iteration Version 3

## This Markdown file is the iteration of model, objective function, constraints with detailed mathematics proof and few key notes from the theorem. My formula represents a mathematically rigorous Lagrangian dual of the facility-distribution optimization model

## Author: Mingchen Yuan, my2878@columbia.edu, @yuanmingchen922
## Github Link: "https://github.com/yuanmingchen922/DHL.1"


# My overall workflow is illurstrated as below 
Farkas Lemma⇒Weak Duality⇒Strong Duality⇒KKT Conditions⇒Primal–Dual Convergence

# Objective Function Version 3: A gengeral form 
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cboxed%7B%0A%5Cmax_%7B%5Clambda%2C%5Cmu%20%5Cge%200%7D%20%5C%3B%20%5Cmin_%7BX%2CZ%2CY%7D%20%5C%3B%0A%5CBigg%5B%0A%5Csum_i%20f_m%5Ei%281%2B%5Calpha_i%29Y_i%0A%2B%5Csum_j%20f_d%5Ej%281%2B%5Cbeta_j%29Y_j%0A%2B%5Csum_%7Bi%2Cj%7D%5CBig%28v_m%5Ei%2B%5Cfrac%7B3%7D%7B2000%7Dd%5E%7BMD%7D_%7Bij%7D%281%2B%5Cgamma_%7Bij%7D%29%5CBig%29X_%7Bij%7D%0A%2B%5Csum_%7Bj%2Cc%7D%5Cfrac%7BZ_%7Bjc%7D%7D%7B6%7D%5CBig%28p%5Ej%2B9.75%2B3.5%5Cfrac%7Bd%5E%7BDC%7D_%7Bjc%7D%7D%7B500%7D%5CBig%29%281%2B%5Cdelta_%7Bjc%7D%29%0A%2B%5Csum_%7Bi%2Cj%7D%5Clambda_%7Bij%7D%28X_%7Bij%7D-U_%7Bij%7DY_i%29%0A%2B%5Csum_%7Bj%2Cc%7D%5Cmu_%7Bjc%7D%28Z_%7Bjc%7D-V_%7Bjc%7DY_j%29%0A%5CBigg%5D%0A%7D" />
</p>


A more rigorous layered formatting 
1. Primal Problem: 
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Baligned%7D%0A%5Cmin_%7BX%2CZ%2CY%7D%20%5Cquad%20%26%20f%28X%2CZ%2CY%29%20%5C%5C%0A%5Ctext%7Bs.t.%7D%20%5Cquad%20%0A%26%20g_%7Bij%7D%28X%2CY%29%20%3D%20X_%7Bij%7D%20-%20U_%7Bij%7DY_i%20%5Cle%200%2C%20%5C%5C%0A%26%20h_%7Bjc%7D%28Z%2CY%29%20%3D%20Z_%7Bjc%7D%20-%20V_%7Bjc%7DY_j%20%5Cle%200.%0A%5Cend%7Baligned%7D" />
</p>


2. Lagrange Function: 
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=L%28X%2CZ%2CY%3B%5Clambda%2C%5Cmu%29%0A%3D%20f%28X%2CZ%2CY%29%2B%20%5Csum_%7Bi%2Cj%7D%5Clambda_%7Bij%7D%28X_%7Bij%7D-U_%7Bij%7DY_i%29%2B%20%5Csum_%7Bj%2Cc%7D%5Cmu_%7Bjc%7D%28Z_%7Bjc%7D-V_%7Bjc%7DY_j%29%2C%0A%5Cquad%20%5Clambda%2C%5Cmu%20%5Cge%200." />
</p>


3. Dual Functon: 
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=G%28%5Clambda%2C%5Cmu%29%0A%3D%20%5Cmin_%7BX%2CZ%2CY%7D%20L%28X%2CZ%2CY%3B%5Clambda%2C%5Cmu%29." />
</p>

The dual function represents the minimum value of the Lagrangian for fixed multipliers(\lambda,\mu), and it provides a lower bound on the optimal value of the primal problem by the Weak Duality Theorem

4. Dual Problem: 
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=G%28%5Clambda%2C%5Cmu%29%0A%3D%20%5Cmin_%7BX%2CZ%2CY%7D%20L%28X%2CZ%2CY%3B%5Clambda%2C%5Cmu%29." />
</p>

The dual problem seeks the tightest possible lower bound by optimizing over all nonnegative multipliers.
Under convexity and Slater’s condition, strong duality holds: 
<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cmin_%7BX%2CZ%2CY%7D%20f%28X%2CZ%2CY%29%20%3D%20%5Cmax_%7B%5Clambda%2C%5Cmu%20%5Cge%200%7D%20G%28%5Clambda%2C%5Cmu%29" />
</p>




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
  scenario perturbation or sensitivity coefficients (e.g. ±5% cost variations).  

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

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Csum_j%20Z_%7Bjc%7D%20%3D%20D_c%2C%20%5Cquad%20%5Cforall%20c%20%5Cin%20C" />
</p>


Each customer's demand must be satisfied.

---

#### **(Flow Balance)**

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Csum_c%20Z_%7Bjc%7D%20%3D%20%5Csum_i%20X_%7Bij%7D%2C%20%5Cquad%20%5Cforall%20j%20%5Cin%20I" />
</p>


Each distribution center (DC) must maintain flow balance — total inbound equals total outbound.

---

#### **(Variable Domains)**

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=X%2C%20Z%20%5Cge%200%2C%20%5Cquad%20Y_i%2C%20Y_j%20%5Cin%20%5C%7B0%2C1%5C%7D" />
</p>


All flow variables are non-negative; location variables are binary (or relaxed to [0,1] in LP relaxation).

---

#### **(Linking Constraints to be Relaxed)**

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=X_%7Bij%7D%20%5Cle%20U_%7Bij%7D%20Y_i%2C%20%5Cquad%20Z_%7Bjc%7D%20%5Cle%20V_%7Bjc%7D%20Y_j" />
</p>


These constraints link flow variables to facility-opening decisions.  
They will be **relaxed** in the Lagrangian formulation.

---

### Notes on Big-M Constraints and Lagrangian Relaxation

In **Version 1**, the Big-M linking constraints were written as:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=X_%7Bij%7D%20%5Cle%20M%20Y_i%2C%20%5Cquad%20Z_%7Bjc%7D%20%5Cle%20M%20Y_j" />
</p>


However, in **Version 3**, we introduce **Lagrangian Relaxation**.

The linking constraints are **softened** — they are no longer treated as hard constraints inside the inner minimization problem.  
Instead, they are moved into the objective function with associated **Lagrange multipliers**:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cmax_%7B%5Clambda%2C%5Cmu%20%5Cge%200%7D%0A%5CBigg%5B%0A%5Csum_%7Bi%2Cj%7D%20%5Clambda_%7Bij%7D%28X_%7Bij%7D%20-%20U_%7Bij%7DY_i%29%2B%20%5Csum_%7Bj%2Cc%7D%20%5Cmu_%7Bjc%7D%28Z_%7Bjc%7D%20-%20V_%7Bjc%7DY_j%29%0A%5CBigg%5D" />
</p>


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

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Baligned%7D%0AL%28X%2CZ%2CY%3B%5Clambda%2C%5Cmu%29%0A%26%3D%20%5Csum_i%20f_m%5Ei%281%2B%5Calpha_i%29Y_i%2B%20%5Csum_j%20f_d%5Ej%281%2B%5Cbeta_j%29Y_j%20%5C%5C%0A%26%5Cquad%20%2B%20%5Csum_%7Bi%2Cj%7D%5CBig%28v_m%5Ei%2B%5Cfrac%7B3%7D%7B2000%7Dd_%7Bij%7D%5E%7BMD%7D%281%2B%5Cgamma_%7Bij%7D%29%5CBig%29X_%7Bij%7D%20%5C%5C%0A%26%5Cquad%20%2B%20%5Csum_%7Bj%2Cc%7D%5Cfrac%7BZ_%7Bjc%7D%7D%7B6%7D%5CBig%28p%5Ej%2B9.75%2B3.5%5Cfrac%7Bd_%7Bjc%7D%5E%7BDC%7D%7D%7B500%7D%5CBig%29%281%2B%5Cdelta_%7Bjc%7D%29%20%5C%5C%26%5Cquad%20%2B%20%5Csum_%7Bi%2Cj%7D%5Clambda_%7Bij%7D%28X_%7Bij%7D-U_%7Bij%7DY_i%29%2B%20%5Csum_%7Bj%2Cc%7D%5Cmu_%7Bjc%7D%28Z_%7Bjc%7D-V_%7Bjc%7DY_j%29.%0A%5Cend%7Baligned%7D%0A%5Ctag%7BL%7D" />
</p>


---

#### **Dual Function**

For given multipliers $(\lambda,\mu)$,

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=g%28%5Clambda%2C%5Cmu%29%0A%3D%20%5Cinf_%7B%28X%2CZ%2CY%29%5Ctext%7B%20s.t.%20%7D%28H%29%7D%20L%28X%2CZ%2CY%3B%5Clambda%2C%5Cmu%29.%0A%5Ctag%7Bg%7D" />
</p>


This is called the **dual function** — it provides a lower bound to the primal objective value for any $\lambda,\mu \ge 0$.

---

### **Dual Problem (Outer Maximization)**

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cmax_%7B%5Clambda%20%5Cge%200%2C%20%5C%20%5Cmu%20%5Cge%200%7D%20%5C%3B%20g%28%5Clambda%2C%5Cmu%29.%0A%5Ctag%7BD%7D" />
</p>


This outer maximization searches for the tightest (largest) lower bound among all nonnegative multipliers.

---

### **2) Weak Duality**

Assume $(\bar X,\bar Z,\bar Y)$ a feasible primal solution satisfying both (H) and (LNK).  
Then, by construction:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=L%28%5Cbar%20X%2C%5Cbar%20Z%2C%5Cbar%20Y%3B%5Clambda%2C%5Cmu%29%0A%3D%20%5Ctext%7BCost%7D%28%5Cbar%20X%2C%5Cbar%20Z%2C%5Cbar%20Y%29%2B%20%5Csum_%7Bi%2Cj%7D%5Clambda_%7Bij%7D%28%5Cbar%20X_%7Bij%7D-U_%7Bij%7D%5Cbar%20Y_i%29%2B%20%5Csum_%7Bj%2Cc%7D%5Cmu_%7Bjc%7D%28%5Cbar%20Z_%7Bjc%7D-V_%7Bjc%7D%5Cbar%20Y_j%29." />
</p>


Because $\lambda,\mu \ge 0$ and $\bar X_{ij} \le U_{ij}\bar Y_i, \ \bar Z_{jc} \le V_{jc}\bar Y_j$,

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=L%28%5Cbar%20X%2C%5Cbar%20Z%2C%5Cbar%20Y%3B%5Clambda%2C%5Cmu%29%20%5Cle%20%5Ctext%7BCost%7D%28%5Cbar%20X%2C%5Cbar%20Z%2C%5Cbar%20Y%29." />
</p>


Taking the infimum over all feasible primal points gives:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cboxed%7B%0Ag%28%5Clambda%2C%5Cmu%29%20%5Cle%20p%5E%2A%20%3D%20%5Cmin_%7B%28H%29%2C%28LNK%29%7D%20%5Ctext%7BCost%7D%2C%20%5Cqquad%0A%5Cmax_%7B%5Clambda%2C%5Cmu%20%5Cge%200%7D%20g%28%5Clambda%2C%5Cmu%29%20%5Cle%20p%5E%2A.%0A%7D" />
</p>


This means every multiplier pair $(\lambda,\mu)$ gives a **valid lower bound** to the primal optimum $p^*$.

---

### **3) Strong Duality and Complementary Slackness — LP Relaxation**

If the binary variables $Y_i,Y_j$ are relaxed to continuous variables in $[0,1]$,  
and if the feasible region satisfies **Slater’s condition**, then it's very obvious that the problem is convex and strong duality:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cmin_%7B%28H%29%2C%28LNK%29%7D%20%5Ctext%7BCost%7D%0A%3D%20%5Cmax_%7B%5Clambda%2C%5Cmu%20%5Cge%200%7D%20g%28%5Clambda%2C%5Cmu%29%0A%3D%20L%28X%5E%2A%2CZ%5E%2A%2CY%5E%2A%3B%5Clambda%5E%2A%2C%5Cmu%5E%2A%29%2C" />
</p>


and **complementary slackness** holds:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Clambda_%7Bij%7D%5E%2A%28X_%7Bij%7D%5E%2A%20-%20U_%7Bij%7DY_i%5E%2A%29%20%3D%200%2C%20%5Cquad%0A%5Cmu_%7Bjc%7D%5E%2A%28Z_%7Bjc%7D%5E%2A%20-%20V_%7Bjc%7DY_j%5E%2A%29%20%3D%200.%0A%5Ctag%7BCS%7D" />
</p>


Thus, the saddle point $(X^*,Z^*,Y^*;\lambda^*,\mu^*)$ satisfies both **primal and dual optimality** simultaneously.

---

### **4) Structure of the Inner Minimization**

Expanding $L$ and grouping terms gives:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Baligned%7D%0AL%20%26%3D%20%0A%5Csum_i%20%0A%5Cunderbrace%7B%5CBig%28%0Af_m%5Ei%281%2B%5Calpha_i%29-%20%5Csum_j%20%5Clambda_%7Bij%7D%20U_%7Bij%7D%0A%5CBig%29%7D_%7B%5Cdisplaystyle%20%3D%3A%5Cphi_i%28%5Clambda%29%7D%0AY_i%2B%20%0A%5Csum_j%20%0A%5Cunderbrace%7B%5CBig%28%0Af_d%5Ej%281%2B%5Cbeta_j%29-%20%5Csum_c%20%5Cmu_%7Bjc%7D%20V_%7Bjc%7D%0A%5CBig%29%7D_%7B%5Cdisplaystyle%20%3D%3A%5Cpsi_j%28%5Cmu%29%7D%0AY_j%20%5C%5C%5B8pt%5D%0A%26%5Cquad%20%2B%0A%5Csum_%7Bi%2Cj%7D%20%0A%5Cunderbrace%7B%5CBig%28%0Av_m%5Ei%20%2B%20%5Cfrac%7B3%7D%7B2000%7D%20d_%7Bij%7D%5E%7BMD%7D%281%2B%5Cgamma_%7Bij%7D%29%2B%20%5Clambda_%7Bij%7D%0A%5CBig%29%7D_%7B%5Cdisplaystyle%20%3D%3A%5Ctilde%7Bc%7D_%7Bij%7D%5EX%28%5Clambda%29%7D%0AX_%7Bij%7D%20%5C%5C%5B8pt%5D%0A%26%5Cquad%20%2B%0A%5Csum_%7Bj%2Cc%7D%0A%5Cunderbrace%7B%5Cfrac%7B1%7D%7B6%7D%0A%5CBig%28p%5Ej%20%2B%209.75%20%2B%203.5%5Cfrac%7Bd_%7Bjc%7D%5E%7BDC%7D%7D%7B500%7D%0A%5CBig%29%0A%281%2B%5Cdelta_%7Bjc%7D%29%2B%20%5Cmu_%7Bjc%7D%7D_%7B%5Cdisplaystyle%20%3D%3A%5Ctilde%7Bc%7D_%7Bjc%7D%5EZ%28%5Cmu%29%7DZ_%7Bjc%7D.%0A%5Cend%7Baligned%7D" />
</p>



Given \((\lambda, \mu)\), the inner problem becomes:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cmin%20%5Csum_i%20%5Cphi_i%28%5Clambda%29Y_i%20%2B%20%5Csum_j%20%5Cpsi_j%28%5Cmu%29Y_j%20%2B%20%5Csum_%7Bi%2Cj%7D%5Ctilde%7Bc%7D%5EX_%7Bij%7D%28%5Clambda%29X_%7Bij%7D%20%2B%20%5Csum_%7Bj%2Cc%7D%5Ctilde%7Bc%7D%5EZ_%7Bjc%7D%28%5Cmu%29Z_%7Bjc%7D%0A%5Cquad%20%5Ctext%7Bs.t.%20%7D%20%28H%29" />
</p>



This decomposition makes the inner problem separable and computationally easier to solve.

---

#### (A) Optimal Structure for Y

Since in \((H)\), \(Y\) is independent from \(X, Z\) (as (LNK) is relaxed),  
we have:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=Y_i%5E%2A%28%5Clambda%29%20%3D%20%5Carg%20%5Cmin_%7BY_i%20%5Cin%20%5C%7B0%2C1%5C%7D%7D%20%5Cphi_i%28%5Clambda%29Y_i%20%3D%0A%5Cbegin%7Bcases%7D%0A1%2C%20%26%20%5Cphi_i%28%5Clambda%29%20%3C%200%2C%5C%5C%0A0%2C%20%26%20%5Cphi_i%28%5Clambda%29%20%3E%200%2C%5C%5C%0A%5Ctext%7Bany%20of%20%7D%20%5C%7B0%2C1%5C%7D%2C%20%26%20%5Cphi_i%28%5Clambda%29%20%3D%200.%0A%5Cend%7Bcases%7D" />
</p>


Hence,

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=Y_i%5E%2A%28%5Clambda%29%20%3D%201%5C%7B%5Csum_j%20%5Clambda_%7Bij%7DU_%7Bij%7D%20%3E%20f_m%5Ei%281%2B%5Calpha_i%29%5C%7D%0A%5Ctag%7BY-MFG%7D" />
</p>


Similarly,

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=Y_j%5E%2A%28%5Cmu%29%20%3D%201%5C%7B%5Csum_c%20%5Cmu_%7Bjc%7DV_%7Bjc%7D%20%3E%20f_d%5Ej%281%2B%5Cbeta_j%29%5C%7D%0A%5Ctag%7BY-DC%7D" />
</p>


---

#### (B) Optimal Structure for X and Z

The variables \((X,Z)\) form a **minimum-cost flow** problem with balanced constraints:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cmin%20%5Csum_%7Bi%2Cj%7D%20%5Ctilde%7Bc%7D%5EX_%7Bij%7D%28%5Clambda%29X_%7Bij%7D%20%2B%20%5Csum_%7Bj%2Cc%7D%5Ctilde%7Bc%7D%5EZ_%7Bjc%7D%28%5Cmu%29Z_%7Bjc%7D%0A%5Cquad%20%5Ctext%7Bs.t.%20Demand%20/%20Balance%20/%20Nonnegativity%7D" />
</p>


The KKT shortest-path conditions indicate that for optimal flows,  
if the reduced cost \(\tilde{c}^X_{ij}(\lambda)\) or \(\tilde{c}^Z_{jc}(\mu)\) is minimal on feasible arcs, then those arcs carry positive flow. 

---

### 5) Dual Ascent (Outer Maximization) — Subgradient Method

Assume \((X^*(\lambda,\mu), Z^*(\lambda,\mu), Y^*(\lambda,\mu))\) be the optimal solution of the inner problem for fixed multipliers \((\lambda, \mu)\).

Then the subgradients of the dual function \(g(\lambda,\mu)\) are:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cpartial_%7B%5Clambda_%7Bij%7D%7D%20g%28%5Clambda%2C%5Cmu%29%20%5Cni%20X_%7Bij%7D%5E%2A%20-%20U_%7Bij%7DY_i%5E%2A%2C%20%5Cqquad%0A%5Cpartial_%7B%5Cmu_%7Bjc%7D%7D%20g%28%5Clambda%2C%5Cmu%29%20%5Cni%20Z_%7Bjc%7D%5E%2A%20-%20V_%7Bjc%7DY_j%5E%2A" />
</p>


The **dual ascent (subgradient update)** is:

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Clambda_%7Bij%7D%5E%7Bk%2B1%7D%20%3D%20%5Cbig%5B%5Clambda_%7Bij%7D%5Ek%20%2B%20%5Ctau_k%20%28X_%7Bij%7D%5Ek%20-%20U_%7Bij%7DY_i%5Ek%29%5Cbig%5D_%2B%2C%20%5Cqquad%0A%5Cmu_%7Bjc%7D%5E%7Bk%2B1%7D%20%3D%20%5Cbig%5B%5Cmu_%7Bjc%7D%5Ek%20%2B%20%5Ctau_k%20%28Z_%7Bjc%7D%5Ek%20-%20V_%7Bjc%7DY_j%5Ek%29%5Cbig%5D_%2B" />
</p>


where \([\cdot]_+ = \max\{\,\cdot,0\,\}\), and \(\tau_k > 0\) is the step size, diminishing or Polyak step length.
When the LP relaxation satisfies strong duality, the above method guarantees monotonic convergence of \(g(\lambda^k,\mu^k)\) toward the optimal dual value.

---

### 6) KKT Conditions

Define the primal problem (P):

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cmin%20%5Ctext%7BCost%7D%28X%2CZ%2CY%29%20%5Cquad%20%5Ctext%7Bs.t.%20%7D%20%28H%29%2C%20%28LNK%29." />
</p>


The Lagrangian and its dual follow as before.  
Under strong duality, there exists \((X^*,Z^*,Y^*;\lambda^*,\mu^*)\) is the optimal for all solutions iff the following are satisfied :

- **Primal feasibility:** satisfies (H), (LNK)  
- **Dual feasibility:** \(\lambda^*,\mu^* \ge 0\)  
- **Complementary slackness:**

  <p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Clambda_%7Bij%7D%5E%2A%28X_%7Bij%7D%5E%2A%20-%20U_%7Bij%7DY_i%5E%2A%29%20%3D%200%2C%20%5Cquad%0A%20%20%5Cmu_%7Bjc%7D%5E%2A%28Z_%7Bjc%7D%5E%2A%20-%20V_%7Bjc%7DY_j%5E%2A%29%20%3D%200" />
</p>


- **Stationarity:**  
  Gradients of the Lagrangian w.r.t. all continuous variables vanish. Equivalently, reduced costs are non-negative in the minimum-cost flow.

Together, these KKT conditions are **necessary and sufficient** for optimality in the LP-relaxed problem.

Proof completed

---

### 7) Practical Optimality Test Formulas

#### **(a) Facility Opening Test**

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Csum_j%20%5Clambda_%7Bij%7DU_%7Bij%7D%0A%5Cbegin%7Bcases%7D%3E%20f_m%5Ei%281%2B%5Calpha_i%29%20%26%5CRightarrow%20Y_i%5E%2A%20%3D%201%2C%5C%5C%3C%20f_m%5Ei%281%2B%5Calpha_i%29%20%26%5CRightarrow%20Y_i%5E%2A%20%3D%200%2C%0A%5Cend%7Bcases%7D%0A%5Cqquad%0A%5Csum_c%20%5Cmu_%7Bjc%7DV_%7Bjc%7D%0A%5Cbegin%7Bcases%7D%3E%20f_d%5Ej%281%2B%5Cbeta_j%29%20%26%5CRightarrow%20Y_j%5E%2A%20%3D%201%2C%5C%5C%3C%20f_d%5Ej%281%2B%5Cbeta_j%29%20%26%5CRightarrow%20Y_j%5E%2A%20%3D%200.%0A%5Cend%7Bcases%7D" />
</p>


---

#### **(b) Effective Arc Costs**

<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Ctilde%7Bc%7D%5EX_%7Bij%7D%28%5Clambda%29%20%3D%20v_m%5Ei%20%2B%20%5Cfrac%7B3%7D%7B2000%7Dd%5E%7BMD%7D_%7Bij%7D%281%2B%5Cgamma_%7Bij%7D%29%20%2B%20%5Clambda_%7Bij%7D%2C%0A%5Cqquad%0A%5Ctilde%7Bc%7D%5EZ_%7Bjc%7D%28%5Cmu%29%20%3D%20%5Cfrac%7B1%7D%7B6%7D%5CBig%28p%5Ej%20%2B%209.75%20%2B%203.5%5Cfrac%7Bd%5E%7BDC%7D_%7Bjc%7D%7D%7B500%7D%5CBig%29%281%2B%5Cdelta_%7Bjc%7D%29%20%2B%20%5Cmu_%7Bjc%7D." />
</p>


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


<p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%5Cmax_%7B%5Clambda%2C%5Cmu%20%5Cge%200%7D%20%5C%3B%20%5Cmin_%7B%28X%2CZ%2CY%29%20%5Ctext%7B%20s.t.%20%7D%20%28H%29%7D%20L%28X%2CZ%2CY%3B%5Clambda%2C%5Cmu%29" />
</p>


Performing Lagrangian relaxation on the linking constraints (LNK)  
transforms the problem into a **dual maximization** framework.

- **Weak Duality** ensures that any multiplier pair \((\lambda,\mu)\) provides a lower bound on the original problem.  
- Under **LP relaxation**, **strong duality** and **complementary slackness** make the dual and primal optima equal and verifiable.  
- The **inner minimization** becomes separable:  
  - Facility decisions follow sign tests of \(\phi_i(\lambda)\) and \(\psi_j(\mu)\).  
  - Flow variables follow a minimum-cost flow rule.
- The **outer maximization** is performed via subgradient ascent using:
  <p align="center">
  <img src="https://render.githubusercontent.com/render/math?math=%28X%5E%2A%20-%20UY%5E%2A%2C%20%5C%3B%20Z%5E%2A%20-%20VY%5E%2A%29" />
</p>

  as the direction of improvement.

This yields a verifiable, computationally efficient optimality framework for the DHL network optimization model.



