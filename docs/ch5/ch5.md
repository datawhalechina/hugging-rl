# 生成式对抗模仿学习

逆强化学习与学徒式学习的建模方式均为极小化极大化，它们之间主要差别是学习方式的不同。

逆强化学习算法有两个步骤，分别是学习出专家策略优于其它策略的奖励函数、基于强化学习获得智能体策略，这两步不断交替迭代，最终智能体策略收敛到专家策略，可见式(5.1)。
$$
\begin{equation}
IRL_{\psi}(\pi_E)=argmax_{c\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}}-\psi(c)+(\min_{\pi\in\Pi}-H(\pi)+\mathbb{E}\_{\pi}[c(s,a)])-\mathbb{E}\_{\pi_E}[c(s,a)]\tag{5.1}
\end{equation}
$$
学徒式学习基于极小极大优化建模的方式，在给定智能体策略下，最大化专家策略的奖励与智能体策略奖励的差；在给定奖励函数的情况下，最小化智能体策略奖励与专家策略奖励的差，可见式(5.2)。
$$
\begin{equation}
\min_{\pi} -H(\pi)+\max_{c\in\mathcal{C}}\mathbb{E}\_{\pi}[c(s,a)]-\mathbb{E}\_{\pi_E}[c(s,a)]\tag{5.2}
\end{equation}
$$
根据文献[1]，可知，逆强化学习虽然能够使智能体收敛到专家策略，但是不能扩展到复杂环境；学徒式学习虽然能够扩展到复杂环境，但是奖励函数需要精心的设计，且要求专家的奖励函数存在于假设空间内。

为了设计出能够绕过IRL中间步骤，且适用于大环境的模仿学习，GAIL利用Jensen-Shannon divergence作为度量两个分布之间差异的函数，基于极小化极大化建模。





## 统计极限





## 参考文献

[1] Ho J, Ermon S. Generative adversarial imitation learning[J]. Advances in neural information processing systems, 2016, 29.
