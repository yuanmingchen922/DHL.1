## Iteration Version 3
  
## This Markdown file is the iteration of the model, objective function, constraints, with detailed mathematical proof anda  few key notes from the theorem. My formula represents a mathematically rigorous Lagrangian dual of the facility-distribution optimization model
  
## Author: Mingchen Yuan, my2878@columbia.edu
## Github Link: "https://github.com/yuanmingchen922/DHL.1"
  
  
# My overall workflow is illustrated below 
Farkas Lemma⇒Weak Duality⇒Strong Duality⇒KKT Conditions⇒Primal–Dual Convergence
  
# Objective Function Version 3: A general form 
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\boxed{\max_{\lambda,\mu%20\ge%200}%20\;%20\min_{X,Z,Y}%20\;\Bigg[\sum_i%20f_m^i(1+\alpha_i)Y_i+\sum_j%20f_d^j(1+\beta_j)Y_j+\sum_{i,j}\Big(v_m^i+\frac{3}{2000}d^{MD}_{ij}(1+\gamma_{ij})\Big)X_{ij}+\sum_{j,c}\frac{Z_{jc}}{6}\Big(p^j+9.75+3.5\frac{d^{DC}_{jc}}{500}\Big)(1+\delta_{jc})+\sum_{i,j}\lambda_{ij}(X_{ij}-U_{ij}Y_i)+\sum_{j,c}\mu_{jc}(Z_{jc}-V_{jc}Y_j)\Bigg]}"/></p>  
  
  
A more rigorous layered formatting 
1. Primal Problem:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\\min_{X,Z,Y}%20\quad%20&amp;%20f(X,Z,Y)%20\\\text{s.t.}%20\quad%20&amp;%20g_{ij}(X,Y)%20=%20X_{ij}%20-%20U_{ij}Y_i%20\le%200,%20\\&amp;%20h_{jc}(Z,Y)%20=%20Z_{jc}%20-%20V_{jc}Y_j%20\le%200.\<img src="https://latex.codecogs.com/gif.latex?$3.%20Lagrange%20Function:%20L(X,Z,Y;\lambda,\mu)&amp;amp;=%20f(X,Z,Y)+%20\sum_{i,j}%20\lambda_{ij}(X_{ij}%20-%20U_{ij}Y_i)+%20\sum_{j,c}%20\mu_{jc}(Z_{jc}%20-%20V_{jc}Y_j),%20\\&amp;amp;\text{where%20}%20\lambda_{ij},%20\mu_{jc}%20\ge%2003.%20Dual%20Function:%20G(\lambda,\mu)=%20\min_{X,Z,Y}%20\;%20L(X,Z,Y;\lambda,\mu).%20The%20dual%20function%20represents%20the%20minimum%20value%20of%20the%20Lagrangian%20for%20fixed%20multipliers(\lambda,\mu),%20and%20it%20provides%20a%20lower%20bound%20on%20the%20optimal%20value%20of%20the%20primal%20problem%20by%20the%20Weak%20Duality%20Theorem4.%20Dual%20Problem:%20\max_{\lambda,\mu%20\ge%200}%20\;%20G(\lambda,\mu)The%20dual%20problem%20seeks%20the%20tightest%20possible%20lower%20bound%20by%20optimizing%20over%20all%20nonnegative%20multipliers.Under%20convexity%20and%20Slater’s%20condition,%20strong%20duality%20holds:%20\min_{X,Z,Y}%20f(X,Z,Y)=%20\max_{\lambda,\mu%20\ge%200}%20G(\lambda,\mu)#%20Notations%20and%20Variablesi,j∈I:%20candidate%20manufacturing%20and%20distribution%20locationsc∈C:%20customersDecision%20Variables:%20Xij​≥0(flow of bottles from MFG i to DC j),%20Zjc​≥0(flow from DC j to customer c),%20Yi​,Yj​∈[0,1]%20(binary open/close decisions)Upper%20bounds:%20Uij​&amp;gt;0,%20Vjc​=Dc​&amp;gt;0%20used%20to%20replace%20the%20Big-MParameters:%20fmi​,fdj​,vmi​,dijMD​,djcDC​,pj%20include%20cost%20and%20transportationScenario%20coefficients:%20αi​,βj​,γij​,δjcMultiplier:%20λij​≥0%20​,μjc​≥0%20as%20a%20outer%20layer%20dual%20parametersContinuous%20variables%20must%20be%20non-negative,%20location%20variables%20must%20be%20binary#%20Constraints(Demand%20Satisfaction%20)%20∑​Zjc​=Dc​,%20∀c∈C,(Flow%20Balance)%20c∑​Zjc​=i∑​Xij​,%20∀j∈I,%20(Variable%20Domains)%20X,Z≥0,Yi​,Yj​∈{0,1}.(Linking%20Constraint%20needs%20to%20be%20relaxed)%20Xij​≤Uij​Yi​,Zjc​≤Vjc​YjNotes%20regarding%20Big-M%20constraints%20(linking%20Variable)Big-M%20linking%20constraints%20in%20Version%201%20stated%20the%20following:%20X_{ij}\le%20MY_i,%20Z_{jc}\le%20MY_j.%20In%20this%20version%20(Ver_3),%20I%20introduced%20Lagrangian%20Relaxation.%20The%20constraints%20are%20softened.%20It&amp;#39;s%20no%20longer%20considered%20a%20hard%20constraint%20in%20the%20inner%20problem.%20I%20multiply%20λij%20and%20μjc%20as%20stated%20below:%20\max\sum_{i,j}\lambda_{ij}(X_{ij}-U_{ij}Y_i)+\sum_{j,c}\mu_{jc}(Z_{jc}-V_{jc}Y_j)Then%20the%20max%20function%20penalized%20any%20violation%20of%20these%20constraints,%20making%20the%20inner%20problem%20easier%20to%20solve.%20Approaches%20through%20using%20fewer%20constraints%20and%20separable%20structure#%20The%20following%20parts%20show%20mathematical%20proof%20for%20several%20parts%20that%20need%20to%20be%20satisfied%20in%20order%20to%20find%20the%20optimum%20for%20all%20feasible%20solutionsAll%20variables%20and%20constraints%20are%20stated%20earlier.%20Denote%20hard%20constraints%201-3%20as%20H,%20and%20take%20the%20linking%20constraint%20in%20Lagrange%20to%20be%20relaxed,%20denote%20LKProof:%20The%20following%20consists%20of%20several%20parts%20Firstly,%20define%20the%20Lagrangian%20Function%20and%20the%20dual%20function\section*{1)%20Lagrangian%20and%20Dual%20Function}\subsection*{Lagrangian%20Function}Introduce%20nonnegative%20multipliers%20\(\lambda_{ij},%20\mu_{jc}%20\ge%200"/>%20for%20constraints%20(LNK).%20%20Then%20the%20Lagrangian%20is%20defined%20as:\[\begin{aligned}L(X,Z,Y;\lambda,\mu)&amp;=%20\sum_i%20f_m^i(1+\alpha_i)Y_i+%20\sum_j%20f_d^j(1+\beta_j)Y_j%20\\&amp;\quad%20+%20\sum_{i,j}\Big(v_m^i+\frac{3}{2000}d_{ij}^{MD}(1+\gamma_{ij})\Big)X_{ij}%20\\&amp;\quad%20+%20\sum_{j,c}\frac{Z_{jc}}{6}\Big(p^j+9.75+3.5\frac{d_{jc}^{DC}}{500}\Big)(1+\delta_{jc})%20\\&amp;\quad%20+%20\sum_{i,j}\lambda_{ij}(X_{ij}-U_{ij}Y_i)+%20\sum_{j,c}\mu_{jc}(Z_{jc}-V_{jc}Y_j).\end{aligned}\tag{L}"/></p>  
  
  
\subsection*{Dual Function}
For given multipliers <img src="https://latex.codecogs.com/gif.latex?\lambda,\mu"/>,
<p align="center"><img src="https://latex.codecogs.com/gif.latex?g(\lambda,\mu)=%20\inf_{(X,Z,Y)\text{%20s.t.%20}(H)}%20L(X,Z,Y;\lambda,\mu).\tag{g}"/></p>  
  
This is called the dual function — it gives a lower bound to the primal objective value for any <img src="https://latex.codecogs.com/gif.latex?\lambda,\mu\ge0."/>
  
\subsection*{Dual Problem (Outer Maximization)}
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\max_{\lambda\ge0,\%20\mu\ge0}%20\;%20g(\lambda,\mu).\tag{D}"/></p>  
  
This outer maximization searches for the tightest (largest) lower bound among all nonnegative multipliers.
  
\section*{2) Weak Duality}
Let <img src="https://latex.codecogs.com/gif.latex?(\bar%20X,\bar%20Z,\bar%20Y)"/> be any feasible primal solution, satisfying both (H) and (LNK).  
Then by construction:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?L(\bar%20X,\bar%20Z,\bar%20Y;\lambda,\mu)=%20\text{Cost}(\bar%20X,\bar%20Z,\bar%20Y)+%20\sum_{i,j}\lambda_{ij}(\bar%20X_{ij}-U_{ij}\bar%20Y_i)+%20\sum_{j,c}\mu_{jc}(\bar%20Z_{jc}-V_{jc}\bar%20Y_j)."/></p>  
  
Because <img src="https://latex.codecogs.com/gif.latex?\lambda,\mu\ge0"/> and <img src="https://latex.codecogs.com/gif.latex?\bar%20X_{ij}\le%20U_{ij}\bar%20Y_i,%20\bar%20Z_{jc}\le%20V_{jc}\bar%20Y_j,"/>
<p align="center"><img src="https://latex.codecogs.com/gif.latex?L(\bar%20X,\bar%20Z,\bar%20Y;\lambda,\mu)\le%20\text{Cost}(\bar%20X,\bar%20Z,\bar%20Y)."/></p>  
  
Taking the infimum over all feasible primal points gives:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\boxed{g(\lambda,\mu)\le%20p^*=\min_{(H),(LNK)}%20\text{Cost},\qquad\max_{\lambda,\mu\ge0}%20g(\lambda,\mu)\le%20p^*.}\tag{Weak%20Duality}"/></p>  
  
  
\section*{3) Strong Duality and Complementary Slackness (LP Relaxation)}
If the binary variables <img src="https://latex.codecogs.com/gif.latex?Y_i,%20Y_j"/> are relaxed to continuous variables in <img src="https://latex.codecogs.com/gif.latex?[0,1]"/>, and if the feasible region satisfies Slater’s condition, then:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\min_{(H),(LNK)}%20\text{Cost}=\max_{\lambda,\mu\ge0}%20g(\lambda,\mu)=L(X^*,Z^*,Y^*;\lambda^*,\mu^*),"/></p>  
  
and complementary slackness holds:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\lambda_{ij}^*(X_{ij}^*-U_{ij}Y_i^*)=0,\quad\mu_{jc}^*(Z_{jc}^*-V_{jc}Y_j^*)=0.\tag{CS}"/></p>  
  
Thus the saddle point <img src="https://latex.codecogs.com/gif.latex?(X^*,Z^*,Y^*;\lambda^*,\mu^*)"/> satisfies both primal and dual optimality simultaneously.
  
\section*{4) Structure of the Inner Minimization}
Expand <img src="https://latex.codecogs.com/gif.latex?L"/> and group terms:
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}L&amp;=\sum_i%20\Big(f_m^i(1+\alpha_i)-\sum_j\lambda_{ij}U_{ij}\Big)Y_i+\sum_j%20\Big(f_d^j(1+\beta_j)-\sum_c\mu_{jc}V_{jc}\Big)Y_j\\&amp;\quad+\sum_{i,j}\Big(v_m^i+\frac{3}{2000}d_{ij}^{MD}(1+\gamma_{ij})+\lambda_{ij}\Big)X_{ij}\\&amp;\quad+\sum_{j,c}\Big[\frac{1}{6}\Big(p^j+9.75+3.5\frac{d_{jc}^{DC}}{500}\Big)(1+\delta_{jc})+\mu_{jc}\Big]Z_{jc}.\end{aligned}\tag{Sep}"/></p>  
  
  
\section*{4. Correct Layered Structure (Rigorous Formulation)}
  
\subsection*{1) Primal Problem}
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\begin{aligned}\min_{X,Z,Y}%20\quad%20&amp;%20f(X,Z,Y)%20\\\text{s.t.}%20\quad%20&amp;%20g_{ij}(X,Y)%20=%20X_{ij}%20-%20U_{ij}Y_i%20\le%200,%20\\&amp;%20h_{jc}(Z,Y)%20=%20Z_{jc}%20-%20V_{jc}Y_j%20\le%200.\end{aligned}"/></p>  
  
  
\subsection*{2) Lagrangian Function}
<p align="center"><img src="https://latex.codecogs.com/gif.latex?L(X,Z,Y;\lambda,\mu)=%20f(X,Z,Y)+%20\sum_{i,j}\lambda_{ij}(X_{ij}-U_{ij}Y_i)+%20\sum_{j,c}\mu_{jc}(Z_{jc}-V_{jc}Y_j),\quad%20\lambda,\mu%20\ge%200."/></p>  
  
  
\subsection*{3) Dual Function}
<p align="center"><img src="https://latex.codecogs.com/gif.latex?G(\lambda,\mu)=%20\min_{X,Z,Y}%20L(X,Z,Y;\lambda,\mu)."/></p>  
  
  
\subsection*{4) Dual Problem}
<p align="center"><img src="https://latex.codecogs.com/gif.latex?\max_{\lambda,\mu%20\ge%200}%20G(\lambda,\mu)."/></p>  
  
  
  
  
  
  