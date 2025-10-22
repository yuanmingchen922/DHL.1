## Iteration Version 3

## This Markdown file is the iteration of model, objective function, constraints with detailed mathematics proof and few key notes from the theorem. My formula represents a mathematically rigorous Lagrangian dual of the facility-distribution optimization model

## Author: Mingchen Yuan, my2878@columbia.edu, @yuanmingchen922
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


# Please go to version_3.md for more detailed math proof. A reminder needs to be stated here is that the function in the pdf did not redener well. If cased any confusion, I sincerely apologize. However, I can make sure that the function here is the correct one.
