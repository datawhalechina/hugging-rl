# 基于正则化的方法

正则化是一个很好调节模型的工具。对于策略的正则化，策略梯度的目标函数可为式(4.1)。
$$
\begin{equation}
J(\theta)=\mathbb{E}\_{s,a\sim\mathcal{D}}[\mathcal{Q}^{\pi\_{\theta}}(s,a)]+\mathcal{R}(\theta)\tag{3.1}
\end{equation}
$$
对于值函数的正则化，目标函数可为式(4.2)。
$$
\begin{equation}
J(\phi)=\mathbb{E}\_{s,a,{s}'\sim\mathcal{D}}[(r(s,a)+\gamma\mathbb{E}\_{{a}'\sim\pi\_{off}(.\vert s)}[\mathcal{Q}\_{\phi}^{\pi}({s}',{a}')]-\mathcal{Q}^{\pi}\_{\phi}(s,a))^2]+\mathcal{R}(\phi)\tag{3.2}
\end{equation}
$$
式(3.1)和(3.2)中$\mathcal{R}(\theta)$表示的是正则化项。

正则化可以调节模型的形状或参数，但是不能限制策略$\pi_{\theta}$与$\pi_{\beta}$的距离。因此，基于正则化的离线强化学习算法，需要其它方法限制策略，例如：保守型模型、策略约束。



## Conservative Q-Learning

为了解决由智能体学到的策略$\pi$与数据产生的策略之间分布偏移产生的价值高估问题，CQL算法学习出保守型Q函数，使其成为Q函数真实值的下界。在理论上，证明了CQL的确产生了Q函数真实值的下界，且该算法可应用到策略学习步骤中。







## 参考文献

[1] Kumar A, Zhou A, Tucker G, et al. Conservative q-learning for offline reinforcement learning[J]. Advances in Neural Information Processing Systems, 2020, 33: 1179-1191.

[2] Prudencio R F, Maximo M R O A, Colombini E L. A survey on offline reinforcement learning: Taxonomy, review, and open problems[J]. IEEE Transactions on Neural Networks and Learning Systems, 2023.
