# 对抗式模仿学习

行为克隆算法的核心思想是智能体策略与专家策略匹配，而对抗式模仿学习与之不同，它的核心思想是状态-动作的匹配。专家策略是不可知的，但专家策略下的状态-动作分布是服从一个特定分布的，因为专家策略不会使其自身陷入到坏的状态。若直接模仿专家策略，那么面对专家未访问过的状态，智能体会采取均匀策略，这一定与专家策略不一致；若直接模仿专家策略下的状态-动作分布，那么智能体也有更大概率不会使自身陷入到“坏状态”。由此可知，对抗模仿学习算法能够缓解复合误差问题。

然而，这并不意味着对抗式模仿学习没有缺陷，正是因为其核心思想是匹配状态-动作，所以智能体学习出的策略会倾向于使其产生的状态-动作分布能够与专家匹配，而不一定是专家动作。

根据上面描述，可确定对抗式模仿学习的目标函数为式4.1
$$
\begin{equation}
\min_{\pi\in\Pi}\sum_{h=1}^H\psi(P^\pi_h,P^{\pi^E}_h)\tag{4.1}
\end{equation}
$$
其中，专家策略$\pi^E$下的状态-动作分布可直接利用极大似然估计求得。



## 极小极大优化建模

根据式(4.1)，可知，对抗式模仿学习的目标是最小化智能体策略分布与专家策略分布的距离。其中，$\psi$为距离度量函数。根据文献[1]，可知，两个分布之间距离的常见度量方式$f-divergence$，如式(4.2)。
$$
\begin{equation}
D_f(P\Vert Q)=\sum_x p(x)f(\frac{p(x)}{q(x)})\tag{4.2}
\end{equation}
$$
根据函数$f-Divergence$的定义，函数$f$为凸函数$f:[0,+\infty)\to(-\infty,+\infty)$，且$f(1)=0$。常见的$KL-divergence,JS-divergence$，以及$TV$-距离均为$f-divergence$。

若把$TV-距离$作为度量函数，那么目标函数(4.1)变为(4.3)
$$
\begin{equation}
\min_{\pi\in\Pi}\sum_{h=1}^HD_{TV}(P_h^{\pi},\hat{P}\_h^{\pi^E})=\frac{1}{2}\min\_{\pi\in\Pi}\sum_{h=1}^H\sum_{(s,a)\in \mathcal{s}\times\mathcal{A}}\vert P_h^{\pi}(s,a)-\hat{P}\_h^{\pi^E}(s,a)\vert\tag{4.3}
\end{equation}
$$
若直接优化式(4.3)，那么考虑利用$TV$-距离的对偶函数，可得式(4.4)极小极大目标。
$$
\begin{equation}
\min_{\pi\in\Pi}\sum_{h=1}^H D_{TV}(P_h^\pi,\hat{P}\_h^{\pi^E})=\frac{1}{2}\min\_{\pi\in\Pi}\max_{w\in\mathcal{W}}\sum_{(s,a,h)\in\mathcal{S}\times\mathcal{S}\times[\mathcal{H}]}w_{h(s,a)}(\hat{P}\_h^{\pi^E}(s,a)-P_h^{\pi}(s,a))\tag{4.4}
\end{equation}
$$
式(4.4)中利用了$l1$范数的对偶范数，即$l_{\infty}$范数。根据文献[2]，可知$\mathcal{W}=\{w\in \mathbb{R}^{\vert\mathcal{S}\vert\times\vert\mathcal{A}\vert\times H}:\Vert w\Vert_{\infty}\le 1\}$。

那么，若把$w$看作奖励函数时，那么式(4.4)可变为式(4.5)
$$
\begin{equation}
\min_{\pi\in\Pi}\max_{w\in\mathcal{W}}V_{w}(\pi^{E})-V_{w}(\pi)\tag{4.5}
\end{equation}
$$
式(4.5)所表达的优化为在给定策略$\pi$，求专家策略下价值函数$V_{w}(\pi^E)$最大化的奖励函数$w$；给定奖励函数$w$下，求最大化价值函数$V_w(\pi)$的策略$\pi$。



## 学习范式

### 生成式对抗模仿学习

根据文献[3]，可知，行为克隆算法样本效率较低；逆强化学习虽然样本效率高，但是需要较多的计算资源，扩展性较差，且不能够直接学习策略。正是以上原因，生成式对抗模仿学习由最大因果熵逆强化学习进化而来，其样本效率较高，且能够直接学习策略，绕过了IRL(Inverse Reinforcement Learning)的中间奖励函数学习阶段。

为了描述生成式对抗模仿学习，先介绍一下最大化因果熵逆强化学习算法的目标函数，如式(4.6)。
$$
\begin{equation}
maximize_{c\in\mathcal{C}}(\min_{\pi\in\Pi}-H(\pi)+\mathbb{E}\_\pi[c(s,a)])-\mathbb{E}\_{\pi_E}[c(s,a)]\tag{4.6}
\end{equation}
$$
式(4.6)中$c(s,a)$为成本函数或奖励函数的负值。该式表明逆强化学习分为两步：

1. 在给定成本函数下，最小化智能体策略下成本函数与因果熵的负值之后，寻找专家策略。
2. 在给定策略下，最小化专家策略下的成本，寻找最优成本函数。

根据文献[4]和[5]，可知，熵$H(\pi)$为了解决选择策略分布的不确定性，不仅约束了策略分布匹配特性期望，而且不会服从任何其它特殊分布。



## 参考文献

[1] [F-divergence](https://en.wikipedia.org/wiki/F-divergence)

[2] [Dual norm](https://en.wikipedia.org/w/index.php?title=Dual_norm&oldid=1029266114)

[3] Ho J, Ermon S. Generative adversarial imitation learning[J]. Advances in neural information processing systems, 2016, 29.

[4] Ziebart B D , Maas A L , Bagnell J A ,et al.Maximum Entropy Inverse Reinforcement Learning[C]//Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence, AAAI 2008, Chicago, Illinois, USA, July 13-17, 2008.2008.

[5] Bloem M, Bambos N. Infinite time horizon maximum causal entropy inverse reinforcement learning[C]//53rd IEEE conference on decision and control. IEEE, 2014: 4911-4916.
