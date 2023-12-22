# 基于策略约束的方法与BCQ

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

由于策略$\pi_{\theta}$的估计是参数化函数近似器的方式，那么需要把非参数解$\pi^\*$投射到策略空间，也可以直接最小化$\pi\_{\theta}$与$\pi^\*$之间KL-Divergence的方式进行参数$\theta$估计。那么，策略提升的目标函数为式(2.3)
$$
\begin{equation}
J(\theta)=\mathbb{E}\_{s,a\sim\mathcal{D}}[log\pi\_{\theta}(a\vert s)exp(\frac{1}{\lambda}\hat{A}^{\pi}(s,a))]\tag{2.3}
\end{equation}
$$


## Batch-Constrained deep Q-learning

BCQ算法是离线强化学习的开篇之作。在文献[2]中，作者首先分析了推断错误产生的三个原因，分别是数据不足、模型偏差、训练中的不匹配。其中，数据不足是指若数据$({s}',\pi({s}'))$不足，那么$Q_{\theta}({s}',\pi({s}'))$估计也不准确；模型偏差是指贝尔曼运算$\mathcal{\tau}^{\pi}$的动态转换估计的偏差，其转换形式可见式(2.4)
$$
\begin{equation}
\tau^{\pi}Q(s,a)\approx\mathbb{E}_{{s}'\sim\mathcal{B}}[r+\gamma Q({s}',\pi({s}'))]\tag{2.4}
\end{equation}
$$
式(2.4)中期望是关于数据集$\mathcal{B}$中转换函数的期望。

训练中的不匹配是指即使数据足够，那么若数据集$\mathcal{B}$中的数据分布与策略$\pi$对应的数据分布不一致，价值函数的估计也是不足的。

接下来，作者利用gym中Hopper-v1环境和DDPG算法做了三个实验。第一个实验**Final Buffer**是DDPG智能体以一定探索型噪音的方式在Hopper-v1环境中交互训练100万个时间步，存储所有的转换经验。接下来再利用刚收集的数据离线训练另一个DDPG智能体。第二个实验**Concurrent**是两个智能体同时学习，即一个DDPG与环境交互产生数据，两个智能体同时运用这份数据学习。第三个实验**Imitation**是一个训练好的DDPG智能体作为专家与环境交互100万步，收集这部分数据，利用模仿学习根据这份数据学习。

<div align="center">
  <img src="./img/BCQ_exp.png" width=600 height=500/>
</div>
<div align="center">
  图2.1 三种实验结果(实线表示的是带有探索噪音下每个回合平均结果，点划线表示的是估计的真实值,直线表示的是无探索噪音下每个回合平均结果)
</div>

实验结果表明离线强化学习的策略明显比在线强化学习的策略效果差，即使是专家策略下模仿学习效果也很差。根据并发学习环境下的结果，可知，若初始策略下状态分布存在差异，那么也足够导致离线强化学习的推断错误。

### BCQ算法

BCQ算法背后的思想是：为了避免推断错误，策略$\pi$下的状态-动作访问分布应该与该batch数据集中状态-动作访问分布相似。满足该思想的策略被称为batch-constrained。为了得到batch-constrained策略，策略训练的目标应该是：

1. 最小化策略$\pi$与行为策略$\pi_{\beta}$之间的距离。
2. 策略$\pi$下的状态分布与数据集中状态分布应相似。
3. 最大化价值函数。

在以上三个目标中，目标(1)的重要性高于其它两个目标。这是因为若目标(1)不能被满足，那么目标(2)和(3)也不能被满足。

文献[2]分析了batch-constrained策略在有限回合制马尔科决策过程中的理论特性，结论如下：

1. 若策略$\pi$下数据分布完全包含于行为策略$\pi_{\beta}$下数据分布，那么batch-constrained策略能够消除推断错误。
2. 与标准Q-learning相同条件下，Q-learning的batch-constrained变体能够拟合到最优策略。
3. 在确定型MDP下，BCQ被保证匹配或超越行为策略$\pi_{\beta}$。

如图2.2所示，BCQ算法的伪代码。

<div align="center">
  <img src="./img/BCQ.png" width=600 height=500/>
</div>
<div align="center">
  图2.2 BCQ算法的伪代码
</div>

为了满足目标(1)，把数据集$\mathcal{B}$中状态-动作分布建模为$P^G_{\mathcal{B}}(a\vert s)$。由于高纬空间中$P^G_{\mathcal{B}}(a\vert s)$难以估计，所以训练VAE生成模型$G_w(s)$，用于生成动作。



## 参考文献

[1] Prudencio R F, Maximo M R O A, Colombini E L. A survey on offline reinforcement learning: Taxonomy, review, and open problems[J]. IEEE Transactions on Neural Networks and Learning Systems, 2023.

[2] Fujimoto S, Meger D, Precup D. Off-policy deep reinforcement learning without exploration[C]//International conference on machine learning. PMLR, 2019: 2052-2062.