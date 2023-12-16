# 生成式对抗模仿学习

逆强化学习与学徒式学习的建模方式均为极小化极大化，它们之间主要差别是学习方式的不同。

逆强化学习算法有两个步骤，分别是学习出专家策略优于其它策略的奖励函数、基于强化学习获得智能体策略，这两步不断交替迭代，最终智能体策略收敛到专家策略，可见式(4.1)。
$$
\begin{equation}
IRL_{\psi}(\pi_E)=\underset{c\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}}{argmax}-\psi(c)+(\min_{\pi\in\Pi}-H(\pi)+\mathbb{E}\_{\pi}[c(s,a)])-\mathbb{E}\_{\pi_E}[c(s,a)]\tag{4.1}
\end{equation}
$$
式(4.1)中，$\psi(c)$为成本函数$c(s,a)$的正则化项，其在成本函数空间$\mathcal{C}$内为常数项。

定义策略$\pi$的占用度量$\rho_{\pi}:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$为$\rho_{\pi}(s,a)=\pi(s,a)\sum_{t=0}^{\infty}\gamma^tP(s_t=s\vert\pi)$。

那么，式4.1可被占用度量$\rho$表示为
$$
\begin{aligned}
IRL_{\psi}(\pi_E)&=\underset{c\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}}{argmax}\min\_{\pi\in\Pi}-H(\pi)+\mathbb{E}\_{\pi}[c(s,a)]-\mathbb{E}\_{\pi\_{E}}[c(s,a)] \\\\
&=\underset{c\in\mathbb{R}^{\mathcal{S}\times\mathcal{A}}}{argmax}\min\_{\rho\in\mathcal{D}}-\bar{H}(\rho)+\sum\_{s,a}\rho(s,a)c(s,a)-\sum\_{s,a}\rho\_{E}(s,a)c(s,a)
\end{aligned}\tag{4.2}
$$
式(4.2)的对偶表达形式为
$$
\underset{{\rho\in\mathcal{D}}}{minimize} -\bar{H}(\rho)\\\\
subject\quad to\quad\rho(s,a)=\rho_{E}(s,a),\forall s\in\mathcal{S},a\in\mathcal{A} \tag{4.3}
$$
式(4.2)中成本函数$c(s,a)$可视为式(4.3)中等式约束的惩罚项。

根据文献[1]中理论分析，可知，逆强化学习虽然能够使智能体收敛到专家策略，但是不能扩展到复杂的环境，这是因为环境越复杂，其约束越多，其求解难度越大。同时，环境越复杂，状态与动作的组合数也就越多，那么有限数据集中状态-动作的占用度量也就越低，智能体也就越不会访问未学习过的状态。

学徒式学习的学习方式是：在给定智能体策略下，最大化专家策略的奖励与智能体策略奖励的差；在给定奖励函数的情况下，最小化智能体策略奖励与专家策略奖励的差，可见式(4.4)。
$$
\begin{equation}
\min_{\pi} -H(\pi)+\max_{c\in\mathcal{C}}\mathbb{E}\_{\pi}[c(s,a)]-\mathbb{E}\_{\pi_E}[c(s,a)]\tag{4.4}
\end{equation}
$$
根据文献[1]，可知，学徒式学习等价于正则化函数为$\psi=\delta_{\mathcal{C}}$的逆强化学习。其中，成本函数空间$C$被限制为凸集，且为基本函数$f_1,\dots,f_d$的线性组合，可见式(4.5)。
$$
\mathcal{C}_{linear}=\{\sum_i w_if_i : \Vert w\Vert_2\le 1\quad and\quad \mathcal{C}_{convex}=\{\sum_i w_if_i:\sum_iw_i=1,w_i\gt0 \forall i\}\tag{4.5}
$$
学徒式学习虽然能够扩展到复杂环境，但是其成本函数被限制在线性空间内，需要精心设计基本函数才能精确地收敛到专家策略。

为了设计出能够绕过IRL中间步骤且适用于复杂环境的模仿学习，文献[1]利用式(4.6)作为成本函数$c(s,a)$的正则化函数。
$$
\psi_{GA}(c)=\begin{cases}
\mathbb{E}_{\pi_E}[g(c(s,a))] \quad if\quad c\lt 0\\\\
+\infty\quad otherwise
\end{cases}\tag{4.6}
$$
$$
g(x)=\begin{cases}-x-log(1-e^x)\quad if \quad x\lt 0\\\\ +\infty\quad otherwise\end{cases}\tag{4.7}
$$

其对偶表示形式为式(4.8)
$$
\begin{equation}
\psi\_{GA}^*(\rho_{\pi}-\rho\_{\pi_E})=\underset{D\in(0,1)^{\mathcal{S}\times\mathcal{A}}}{max}\mathbb{E}\_{\pi}[log(D(s,a))]+\mathbb{E}\_{\pi\_E}[log(1-D(s,a))]\tag{4.8}
\end{equation}
$$
式(4.8)实际上是二分类问题的损失函数。其中，$D(s,a)$表示的是区分专家策略产生$(s,a)$与智能体产生(s,a)分类器。根据式(4.9)，可知，若成本函数对专家策略的值越高，那么成本函数的惩罚越大，反亦成立。

由此，GAIL算法的目标函数为
$$
\begin{equation}
\underset{\pi}{minimize}\underset{D\in(0,1)^{\mathcal{S}\times\mathcal{A}}}{max}(\mathbb{E}\_{\pi}[log(D(s,a))]+\mathbb{E}\_{\pi\_E}[log(1-D(s,a))])-\lambda H(\pi)\tag{4.9}
\end{equation}
$$



## 统计极限





## 参考文献

[1] Ho J, Ermon S. Generative adversarial imitation learning[J]. Advances in neural information processing systems, 2016, 29.
