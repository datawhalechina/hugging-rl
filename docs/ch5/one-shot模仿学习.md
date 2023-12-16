# one-shot模仿学习

对于机器人学习新任务，更多的是希望它能够根据少量的演示就能完成任务。然而，模仿学习往往需要大量的数据和精细的特征工程。文献[1]中结合元学习与模仿学习形成了one-shot模仿学习，该算法把同一任务的一种演示和另一种演示的状态作为输入，预测该状态下动作的方式训练神经网络，从而使模型只需根据新任务的一段演示就能完成任务的通用能力。



## 问题设定

one-shot模仿学习问题由$\langle \mathbb{T},\mathbb{D},\pi,\mathcal{R} \rangle$构成，其中

- $\mathbb{T}$是任何的分布，单个任务由$t$表示。
- $\mathbb{D}$是演示的分布，$\mathbb{D}(t)$指的是任务$t$演示的分布。
- $d\sim\mathbb{D}(t)$是一个演示的观测和动作序列。
- $\pi_{\theta}(a\vert o,d)$表示的是策略。
- $\mathcal{R}_t(d)$是任务标量值的评估函数。

该学习问题的目标是对于任务$t\in\mathbb{T}$和演示$d\in\mathbb{D}(t)$最大化策略的期望表现。



## 架构





## 参考文献

[1] Duan Y, Andrychowicz M, Stadie B, et al. One-shot imitation learning[J]. Advances in neural information processing systems, 2017, 30.
