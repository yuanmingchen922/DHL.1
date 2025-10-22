## Iteration Version 3

## This Markdown file is the iteration of the model, objective function, constraints, with detailed mathematical proof anda  few key notes from the theorem. My formula represents a mathematically rigorous Lagrangian dual of the facility-distribution optimization model

## Author: Mingchen Yuan, my2878@columbia.edu
## Github Link: "https://github.com/yuanmingchen922/DHL.1"


# My overall workflow is illustrated below 
Farkas Lemma⇒Weak Duality⇒Strong Duality⇒KKT Conditions⇒Primal–Dual Convergence

# Objective Function Version 3: A general form 
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
\[
\begin{aligned}
\min_{X,Z,Y} \quad & f(X,Z,Y) \\
\text{s.t.} \quad 
& g_{ij}(X,Y) = X_{ij} - U_{ij}Y_i \le 0, \\
& h_{jc}(Z,Y) = Z_{jc} - V_{jc}Y_j \le 0.
\end{aligned}
\]

3. Lagrange Function: L(X,Z,Y;\lambda,\mu)
&= f(X,Z,Y)
+ \sum_{i,j} \lambda_{ij}(X_{ij} - U_{ij}Y_i)
+ \sum_{j,c} \mu_{jc}(Z_{jc} - V_{jc}Y_j), \\
&\text{where } \lambda_{ij}, \mu_{jc} \ge 0

3. Dual Function: 
G(\lambda,\mu)= \min_{X,Z,Y} \; L(X,Z,Y;\lambda,\mu). 
The dual function represents the minimum value of the Lagrangian for fixed multipliers(\lambda,\mu), and it provides a lower bound on the optimal value of the primal problem by the Weak Duality Theorem

4. Dual Problem: 
\max_{\lambda,\mu \ge 0} \; G(\lambda,\mu)
The dual problem seeks the tightest possible lower bound by optimizing over all nonnegative multipliers.
Under convexity and Slater’s condition, strong duality holds: 
\min_{X,Z,Y} f(X,Z,Y)= \max_{\lambda,\mu \ge 0} G(\lambda,\mu)


# Notations and Variables
i,j∈I: candidate manufacturing and distribution locations

c∈C: customers

Decision Variables: Xij​≥0(flow of bottles from MFG i to DC j), Zjc​≥0(flow from DC j to customer c), Yi​,Yj​∈[0,1] (binary open/close decisions)

Upper bounds: Uij​>0, Vjc​=Dc​>0 
used to replace the Big-M

Parameters: fmi​,fdj​,vmi​,dijMD​,djcDC​,pj include cost and transportation

Scenario coefficients: αi​,βj​,γij​,δjc

Multiplier: λij​≥0 ​,μjc​≥0 as a outer layer dual parameters

Continuous variables must be non-negative, location variables must be binary


# Constraints
(Demand Satisfaction ) ∑​Zjc​=Dc​, ∀c∈C,
(Flow Balance) c∑​Zjc​=i∑​Xij​, ∀j∈I, 
(Variable Domains) X,Z≥0,Yi​,Yj​∈{0,1}.
(Linking Constraint needs to be relaxed) Xij​≤Uij​Yi​,Zjc​≤Vjc​Yj

Notes regarding Big-M constraints (linking Variable)
Big-M linking constraints in Version 1 stated the following: 
X_{ij}\le MY_i, Z_{jc}\le MY_j. In this version (Ver_3), I introduced Lagrangian Relaxation. The constraints are softened. It's no longer considered a hard constraint in the inner problem. I multiply λij and μjc as stated below: 
\max\sum_{i,j}\lambda_{ij}(X_{ij}-U_{ij}Y_i)+\sum_{j,c}\mu_{jc}(Z_{jc}-V_{jc}Y_j)
Then the max function penalized any violation of these constraints, making the inner problem easier to solve. Approaches through using fewer constraints and separable structure

# The following parts show mathematical proof for several parts that need to be satisfied in order to find the optimum for all feasible solutions

All variables and constraints are stated earlier. Denote hard constraints 1-3 as H, and take the linking constraint in Lagrange to be relaxed, denote LK

Proof: The following consists of several parts 
Firstly, define the Lagrangian Function and the dual function
\section*{1) Lagrangian and Dual Function}

\subsection*{Lagrangian Function}
Introduce nonnegative multipliers \(\lambda_{ij}, \mu_{jc} \ge 0\) for constraints (LNK).  
Then the Lagrangian is defined as:
\[
\begin{aligned}
L(X,Z,Y;\lambda,\mu)
&= \sum_i f_m^i(1+\alpha_i)Y_i
+ \sum_j f_d^j(1+\beta_j)Y_j \\
&\quad + \sum_{i,j}\Big(v_m^i+\frac{3}{2000}d_{ij}^{MD}(1+\gamma_{ij})\Big)X_{ij} \\
&\quad + \sum_{j,c}\frac{Z_{jc}}{6}\Big(p^j+9.75+3.5\frac{d_{jc}^{DC}}{500}\Big)(1+\delta_{jc}) \\
&\quad + \sum_{i,j}\lambda_{ij}(X_{ij}-U_{ij}Y_i)
+ \sum_{j,c}\mu_{jc}(Z_{jc}-V_{jc}Y_j).
\end{aligned}
\tag{L}
\]

\subsection*{Dual Function}
For given multipliers \(\lambda,\mu\),
\[
g(\lambda,\mu)
= \inf_{(X,Z,Y)\text{ s.t. }(H)} L(X,Z,Y;\lambda,\mu).
\tag{g}
\]
This is called the dual function — it gives a lower bound to the primal objective value for any \(\lambda,\mu\ge0.\)

\subsection*{Dual Problem (Outer Maximization)}
\[
\max_{\lambda\ge0,\ \mu\ge0} \; g(\lambda,\mu).
\tag{D}
\]
This outer maximization searches for the tightest (largest) lower bound among all nonnegative multipliers.

\section*{2) Weak Duality}
Let \((\bar X,\bar Z,\bar Y)\) be any feasible primal solution, satisfying both (H) and (LNK).  
Then by construction:
\[
L(\bar X,\bar Z,\bar Y;\lambda,\mu)
= \text{Cost}(\bar X,\bar Z,\bar Y)
+ \sum_{i,j}\lambda_{ij}(\bar X_{ij}-U_{ij}\bar Y_i)
+ \sum_{j,c}\mu_{jc}(\bar Z_{jc}-V_{jc}\bar Y_j).
\]
Because \(\lambda,\mu\ge0\) and \(\bar X_{ij}\le U_{ij}\bar Y_i, \bar Z_{jc}\le V_{jc}\bar Y_j,\)
\[
L(\bar X,\bar Z,\bar Y;\lambda,\mu)\le \text{Cost}(\bar X,\bar Z,\bar Y).
\]
Taking the infimum over all feasible primal points gives:
\[
\boxed{
g(\lambda,\mu)\le p^*=\min_{(H),(LNK)} \text{Cost},\qquad
\max_{\lambda,\mu\ge0} g(\lambda,\mu)\le p^*.
}
\tag{Weak Duality}
\]

\section*{3) Strong Duality and Complementary Slackness (LP Relaxation)}
If the binary variables \(Y_i, Y_j\) are relaxed to continuous variables in \([0,1]\), and if the feasible region satisfies Slater’s condition, then:
\[
\min_{(H),(LNK)} \text{Cost}
=\max_{\lambda,\mu\ge0} g(\lambda,\mu)
=L(X^*,Z^*,Y^*;\lambda^*,\mu^*),
\]
and complementary slackness holds:
\[
\lambda_{ij}^*(X_{ij}^*-U_{ij}Y_i^*)=0,\quad
\mu_{jc}^*(Z_{jc}^*-V_{jc}Y_j^*)=0.
\tag{CS}
\]
Thus the saddle point \((X^*,Z^*,Y^*;\lambda^*,\mu^*)\) satisfies both primal and dual optimality simultaneously.

\section*{4) Structure of the Inner Minimization}
Expand \(L\) and group terms:
\[
\begin{aligned}
L
&=\sum_i \Big(f_m^i(1+\alpha_i)-\sum_j\lambda_{ij}U_{ij}\Big)Y_i
+\sum_j \Big(f_d^j(1+\beta_j)-\sum_c\mu_{jc}V_{jc}\Big)Y_j\\
&\quad+\sum_{i,j}\Big(v_m^i+\frac{3}{2000}d_{ij}^{MD}(1+\gamma_{ij})+\lambda_{ij}\Big)X_{ij}\\
&\quad+\sum_{j,c}\Big[\frac{1}{6}\Big(p^j+9.75+3.5\frac{d_{jc}^{DC}}{500}\Big)(1+\delta_{jc})+\mu_{jc}\Big]Z_{jc}.
\end{aligned}
\tag{Sep}
\]




