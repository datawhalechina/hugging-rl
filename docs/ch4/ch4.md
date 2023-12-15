# 生成式对抗模仿学习

逆强化学习与学徒式学习的建模方式均为极小化极大化，它们之间主要差别是学习方式的不同。

逆强化学习算法有两个步骤，分别是学习出专家策略优于其它策略的奖励函数、基于强化学习获得智能体策略，这两步不断交替迭代，最终智能体策略收敛到专家策略，可见式(4.1)。
$$
\begin{equation}
IRL_{\psi}(\pi_E)=arg\max_{c\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}}-\psi(c)+(\min_{\pi\in\Pi}-H(\pi)+\mathbb{E}\_{\pi}[c(s,a)])-\mathbb{E}\_{\pi_E}[c(s,a)]\tag{5.1}
\end{equation}
$$
式(5.1)中，$\psi(c)$为成本函数$c(s,a)$的正则化项，其在成本函数空间$\mathcal{C}$内为常数项。

定义策略$\pi$的占用度量$\rho_{\pi}:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$为$\rho_{\pi}(s,a)=\pi(s,a)\sum_{t=0}^{\infty}\gamma^tP(s_t=s\vert\pi)$。

那么，式5.1可被占用度量$\rho$表示为
$$
\begin{aligned}
IRL_{\psi}(\pi_E)=arg\max_{c\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}}\min\_{\pi\in\Pi}-H(\pi)+\mathbb{E}\_{\pi}[c(s,a)]-\mathbb{E}\_{\pi\_{E}}[c(s,a)] \\
IRL_{\psi}(\pi_E)=arg\max\_{c\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}}\min\_{\rho\in\mathcal{D}}-\bar{H}(\rho)+\sum\_{s,a}\rho(s,a)c(s,a)-\sum\_{s,a}\rho\_{E}(s,a)c(s,a)
\end{aligned}
$$


学徒式学习基于极小极大优化建模的方式，在给定智能体策略下，最大化专家策略的奖励与智能体策略奖励的差；在给定奖励函数的情况下，最小化智能体策略奖励与专家策略奖励的差，可见式(4.2)。
$$
\begin{equation}
\min_{\pi} -H(\pi)+\max_{c\in\mathcal{C}}\mathbb{E}\_{\pi}[c(s,a)]-\mathbb{E}\_{\pi_E}[c(s,a)]\tag{5.2}
\end{equation}
$$
根据文献[1]，可知，学徒式学习等价于正则化函数为$\psi=\delta_{\mathcal{C}}$的逆强化学习。其中，成本函数空间$C$被限制为凸集，且为基本函数$f_1,\dots,f_d$线性组合。

逆强化学习虽然能够使智能体收敛到专家策略，但是不能扩展到复杂环境；学徒式学习虽然能够扩展到复杂环境，但是奖励函数需要精心的设计，且要求专家的奖励函数存在于假设空间内。

为了设计出能够绕过IRL中间步骤且适用于复杂环境的模仿学习，

，从而得到GAIL的目标函数式(5.3)。
$$
\begin{equation}
\mathbb{E}[log(D(s,a))]+\mathbb{E}\_{\pi_E}[log(1-D(s,a))]-\lambda H(\pi)\tag{5.3}
\end{equation}
$$
式(5.3)中$D(s,a)$为

GAIL利用Jensen-Shannon divergence度量两个分布之间差异



## 统计极限





## 参考文献

[1] Ho J, Ermon S. Generative adversarial imitation learning[J]. Advances in neural information processing systems, 2016, 29.
