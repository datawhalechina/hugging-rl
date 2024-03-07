# GAIL：生成式对抗模仿学习

模仿学习主要有两种形式，分别是行为克隆和逆强化学习。其中，行为克隆把学习一个策略视作关于状态-动作的监督学习问题；逆强化学习是先找到专家策略奖励最大的奖励函数，然后基于奖励函数学习出策略。行为克隆算法虽然简单，但是该类算法很容易受到分布偏移造成的复合误差影响。逆强化学习与之相反，不仅需要计算奖励函数，还需要在内循环中进行强化学习，所以计算成本很高。在这样背景下，文献[1]作者们期望设计一个算法能够明确告诉智能体如何学习出专家策略，而不需要先寻找奖励函数。

为了设计一个绕过IRL(Inverse Reinforcement Learning)中间步骤的算法，作者们先基于最大化因果熵IRL[2],[3]学习出成本函数，再基于RL学习出策略。接下来，提出了一个无模型模仿学习框架，该框架可以绕过寻找奖励函数的步骤。



## 逆强化学习

给定专家策略$\pi_{E}$，最大化因果熵IRL的目标函数为
$$
\begin{aligned}
\underset{c\in\mathcal{C}}{maximize}(\underset{\pi\in\Pi}{min}-H(\pi)+\mathbb{E}_{\pi}[c(s,a)])-\mathbb{E}_{\pi_{E}}[c(s,a)]
\end{aligned}\tag{1}
$$
式(1)中$H(\pi)\triangleq\mathbb{E}_{\pi}[-log\pi(a\vert s)]$。

根据式(1)，可知，最大化因果熵IRL先寻找成本函数$c\in\mathcal{C}$，该成本函数使专家策略成本最低，其它策略成本最高。然后，再通过强化学习找到专家策略
$$
\begin{aligned}
RL(c)=\underset{\pi\in\Pi}{argmin}-H(\pi)+\mathbb{E}_{\pi}(s,a)
\end{aligned}\tag{2}
$$


## 寻找最优策略

作者们通过检测成本函数候选集$\mathcal{C}$的容量，研究最优IRL的表达力。由于在大容量成本侯选集中研究，很容易产生过拟合，所以施加了一个凸的正则化器s$\psi:\mathbb{R}^{\mathcal{S}\times\mathcal{A}}\to\bar{\mathbb{R}}$，可见式(3)
$$
\begin{aligned}
IRL_{\psi}(\pi_E)=\underset{c\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}}{argmax}-\psi(c)+(\min_{\pi\in\Pi}-H(\pi)+\mathbb{E}_{\pi}[c(s,a)])-\mathbb{E}_{\pi_E}[c(s,a)]
\end{aligned}\tag{3}
$$
为了研究$RL(\tilde{c})$产生的策略，先定义策略$\pi\in\Pi$的占用度量$\rho_{\pi}:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$为$\rho_{\pi}(s,a)=\pi(s,a)\sum_{t=0}^{\infty}\gamma^tP(s_t=s\vert\pi)$。占用度量可被解释为状态-动作对的分布。同时，为了方便表达，占用度量可表达为$\mathbb{E}_{\pi}[c(s,a)]=\sum_{s,a}\rho_{\pi}(s,a)c(s,a)$

根据文献[4]，可知，有效占用度量的集合可被表示为仿射约束的可行集：若$p_0(s)$为初始状态的分布，$P({s}'\vert s,a)$为环境的动力学模型，那么$\mathcal{D}=\{\rho:rho\ge0\quad and\quad \sum_{a}\rho(s,a)=\rho_0(s)+\gamma\sum_{{s}',a}P(s\vert{s}',a)\rho({s}',a)\quad \forall s\in\mathcal{S}\}$。同时，策略$\Pi$与$\mathcal{D}$属于一对一的对应关系。根据以上数学关系，可得：



**命题1**：*若$\rho\in\mathcal{D}$，那么$\rho$式策略$\pi_{\rho}(a\vert s)\triangleq\frac{\rho(s,a)}{\sum_{{a}'}\rho(s,{a}')}$，且$\pi_{\rho}$为占用度量$\rho$的唯一策略。*



为了进一步研究，需要的工具：对于函数$f:\mathbb{R}^{\mathcal{S}\times\mathcal{A}}\to\bar{\mathbb{R}}$，对应的凸共轭$f^{\*}=sup_{y\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}}x^Ty-f(y)$



**命题2**： $RL\circ IRL_{\psi}(\pi_{E})=\underset{\pi\in\Pi}{argmin}-H(\pi)+\psi^*(\rho_{\pi}-\rho_{\pi_E})$。



命题2的证明来源于这样的观察：最优成本函数和策略来源于一个特定函数的鞍点。IRL寻找到鞍点的一个坐标，然后通过RL寻找另一个坐标。

根据命题2可知，$\psi$正则化的IRL隐式的寻找一个策略，该策略的占用度量与专家的占用度量相近，其距离被凸函数$\psi^{*}$所测量。从而，可知，不同的$\psi$会产生不同的模型学习算法。若$\psi$为常数函数，那么可得



**推论1**：若$\psi$是一个常数函数，$\tilde{c}\in IRL_{\psi}(\pi_E)$，且$\tilde{\pi}\in RL(\tilde{c})$，那么$\rho_{\tilde{\pi}}=\rho_{\pi_E}$。



为了证明推论1，需要一个引理来说明占用度量的因果熵



**引理1** ：若$\bar{H}(\rho)=-\sum_{s,a}\rho(s,a)\frac{log\rho(s,a)}{\sum_{{a}'}\rho(s,{a}')}$，那么$\bar{H}$为严格凹的。同时对于$\forall \pi\in\Pi,\rho\in\mathcal{D}$，可得$H(\pi)=\bar{H}(\rho_{\pi})$且$\bar{H}(\rho)=H(\rho_{\pi_{\rho}})$。



推论1的证明：

定义$\bar{L}(\rho,c)=-\bar{H}(\rho)+\sum_{s,a}c(s,a)(\rho(s,a)-\rho_E(s,a))$。对于常量函数$\psi$，根据引理2可得
$$
\begin{aligned}
\tilde{c}\in IRL_{\psi}(\pi_E) &= \underset{c\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}}{argmax}\quad\underset{\pi\in\Pi}{min}-H(\pi)+\mathbb{E}_{\pi}[c(s,a)]-\mathbb{E}_{\pi_E}[c(s,a)+const] \\
&= \underset{c\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}}{argmax}\quad\underset{\rho\in\mathcal{D}}{min}-\bar{H}(\rho)+\sum_{s,a}\rho(s,a)c(s,a)-\sum_{s,a}\rho_{E}(s,a)c(s,a)=\underset{c\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}}{argmax}\quad\underset{\rho\in\mathcal{D}}{min}\bar{L}(\rho,c)
\end{aligned}\tag{4}
$$
那么，该优化问题的对偶问题为
$$
\begin{aligned}
\underset{\rho\in\mathcal{D}}{minimize}-\bar{H}(\rho) \\
subject\quad to\quad \rho(s,a)=\rho_E(s,a)\quad\forall s\in\mathcal{S},a\in\mathcal{A}
\end{aligned}\tag{5}
$$
由此可知，对于拉格朗日函数$\bar{L}$，成本函数$c(s,a)$为等式约束的对偶变量。因此，对于式(5)$\tilde{c}$为对偶最优。由于$\mathcal{D}$为凸集且$-\bar{H}$为凸函数，那么强对偶存在。此外，根据引理1保证$-\bar{H}$严格凸，那么原问题最优可根据$\tilde{\rho}=\underset{\rho\in\mathcal{D}}{argmin}\bar{L}(\rho,\tilde{c})=\underset{\rho\in\mathcal{D}}{argmin}-\bar{H}(\rho)+\sum_{s,a}\tilde{c}(s,a)\rho(s,a)=\rho_E$求解对偶唯一最优得到。上式中第一个等式表明$\tilde{\rho}$为$\bar{L}(\cdot,\tilde{c})$的唯一最小，第三个等式是根据原问题(5)的约束得到。由此，推论1得证。

由此，可得如下结论：

- IRL是一个占用度量匹配问题的对偶问题。
- IRL得到求解的最优问题为原问题的最优。



## 生成式对抗模仿学习

根据推论1可知，若$\psi$为常数，那么产生的最优问题仅在专家所有状态和动作上匹配占用度量。然而，这种算法实际上不实用。实际中，只能专家轨迹只能提供有限的样本集，所以在大的环境中专家的占用度量值为0，从而使学习到的策略由于缺乏数据不会访问未见过的状态-动作对。此外，基于函数近似器的优化问题，产生与$\mathcal{S}\times\mathcal{A}$对一样多的约束，从而求解难度越大。

那么根据命题2，对优化问题(5)进行松弛
$$
\begin{aligned}
\underset{\pi}{minimize}\quad d_{\psi}(\rho_{\pi},\rho_E)-H(\pi)
\end{aligned}\tag{6}
$$
为了设计出能够绕过IRL中间步骤且适用于复杂环境的模仿学习，文献[1]利用式(7)作为成本函数$c(s,a)$的正则化函数。
$$
\psi_{GA}(c)=\begin{cases}
\mathbb{E}_{\pi_E}[g(c(s,a))] \quad if\quad c\lt 0\\\\
+\infty\quad otherwise
\end{cases}\tag{7}
$$

$$
g(x)=\begin{cases}-x-log(1-e^x)\quad if \quad x\lt 0\\\\ +\infty\quad otherwise\end{cases}\tag{8}
$$

其对偶表示形式为式(9)
$$
\begin{aligned}
\psi_{GA}^*(\rho_{\pi}-\rho_{\pi_E})=\underset{D\in(0,1)^{\mathcal{S}\times\mathcal{A}}}{max}\mathbb{E}_{\pi}[log(D(s,a))]+\mathbb{E}_{\pi_E}[log(1-D(s,a))]
\end{aligned}\tag{9}
$$
式(9)实际上是二分类问题的损失函数。其中，$D:\mathcal{S}\times\mathcal{A}\to(0,1)$为区分专家策略下的$(s,a)$与智能体策略下的$(s,a)$的分类器。[GAN](https://www.robotech.ink/index.php/Foundation/203.html "GAN")论文中表明最优损失函数为Jensen-Shannon Divergence：$D_{JS}(\rho_{\pi},\rho_{\pi_E})\triangleq D_{KL}(\rho_{\pi}\Vert\frac{\rho_{\pi}+\rho_{E}}{2})+D_{KL}(\rho_E\Vert\frac{\rho_{\pi}+\rho_E}{2})$。若把熵$H$视为策略的正则化器，那么新的模仿学习算法为
$$
\begin{aligned}
\underset{\pi}{minimize}\psi^{*}_{GA}(\rho_{\pi}-\rho_{\pi_E})-\lambda H(\pi)=D_{JS}(\rho_{\pi},\rho_{\pi_E})-\lambda H(\pi)
\end{aligned}\tag{10}
$$
由此，GAIL算法的目标函数为
$$
\begin{aligned}
\underset{\pi}{minimize}\underset{D\in(0,1)^{\mathcal{S}\times\mathcal{A}}}{max}(\mathbb{E}_{\pi}[log(D(s,a))]+\mathbb{E}_{\pi_E}[log(1-D(s,a))])-\lambda H(\pi)
\end{aligned}\tag{11}
$$
如图1所示，GAIL算法伪代码

<div align="center">
  <img src="https://www.robotech.ink/usr/uploads/2024/02/1923898129.png" width=700/>
</div>
<div align="center">
  图1 GAIL算法伪代码
</div>


## 参考文献

[1] Ho J, Ermon S. Generative adversarial imitation learning[J]. Advances in neural information processing systems, 2016, 29.

[2] Ziebart B D, Maas A L, Bagnell J A, et al. Maximum entropy inverse reinforcement learning[C]//Aaai. 2008, 8: 1433-1438.

[3] Ziebart B D, Bagnell J A, Dey A K. Modeling interaction via the principle of maximum causal entropy[J]. 2010.

[4] Puterman M L. Markov decision processes: discrete stochastic dynamic programming[M]. John Wiley & Sons, 2014.

[5] Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial nets[J]. Advances in neural information processing systems, 2014, 27.