# 基于策略约束的方法

策略约束的离线强化学习算法可被分为两类，分别是显式策略约束和隐式策略约束。显式策略约束方法直接约束策略$\pi$与行为策略$\pi_{\beta}$的距离，即限制策略的分布偏移程度，可见式(2.1)。
$$
J(\theta)=\mathbb{E}\_{s\sim d^{\pi\_{\theta}}(.),a\sim\pi\_{\theta}(.\vert s)}[Q^{\pi}(s,a)] \\
s.t.\quad D(\pi\_{\theta(.\vert s)},\hat{\pi}\_{\beta}(.\vert s))\le\epsilon\tag{2.1}
$$
式(2.1)中$D(\pi\_{\theta}(.\vert s),\hat{\pi}\_{\beta}(.\vert s))$为度量两个策略概率分布之间的距离，常见的度量是$f-divergence$。行为策略$\hat{\pi}\_{\beta}(.\vert s)$的估计方法可为行为克隆或非参数估计法。

显式策略约束方法需要估计行为策略$\pi\_{\beta}$，若其估计不准确，那么该方法的性能会大大下降。隐式策略约束方法对策略进行约束的同时，也避免了估计行为策略约束的需要。该方法把带约束的优化问题式(2.1)，转化为Lagrangian函数为式(2.2)。
$$
\begin{equation}
\mathcal{L}(\pi,\lambda)=\mathbb{E}\_{s\sim d^{\pi\_{\beta}}(.)}[\mathbb{E}\_{a\sim\pi(.\vert s)}[\hat{A}^{\pi}(s,a)]+\lambda(\epsilon-D\_{KL}(\pi(.\vert s)\Vert\pi\_{\beta}(.\vert s)))]\tag{2.2}
\end{equation}
$$
利用KKT条件，求解式(2.2)。根据$\frac{\partial\mathcal{L}}{\partial\pi}=0$，可得$\pi^{*}(a\vert s)\propto\pi\_{\beta}(a\vert s)exp(\lambda^{-1}\hat{A}^{\pi\_k}(s,a))$

由于策略$\pi_{\theta}$的估计是参数化函数近似器的方式，那么需要把非参数解$\pi^*$投射到策略空间，也可以直接最小化$\pi\_{\theta}$与$\pi^*$之间KL-Divergence的方式进行参数$\theta$估计。那么，策略提升的目标函数为式(2.3)
$$
\begin{equation}
J(\theta)=\mathbb{E}\_{s,a\sim\mathcal{D}}[log\pi\_{\theta}(a\vert s)exp(\frac{1}{\lambda}\hat{A}^{\pi}(s,a))]\tag{2.3}
\end{equation}
$$


## Batch-Constrained deep Q-learning

BCQ算法是离线强化学习的开篇之作。



## 参考文献

[1] Prudencio R F, Maximo M R O A, Colombini E L. A survey on offline reinforcement learning: Taxonomy, review, and open problems[J]. IEEE Transactions on Neural Networks and Learning Systems, 2023.

[2] Fujimoto S, Meger D, Precup D. Off-policy deep reinforcement learning without exploration[C]//International conference on machine learning. PMLR, 2019: 2052-2062.