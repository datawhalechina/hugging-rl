# 离线强化学习简介

AlphaGo的诞生，见证了强化学习的威力。然而，在工业界，强化学习落地极其困难。这是因为其需要与环境交互试错的方式学习，大部分真实场景并不像游戏领域那样，与环境交互几乎无成本，例如：若机器人与环境交互过程中摔倒了，可能造成传感器的损坏，就会产生高额的成本。在强化学习中，智能体往往需要与环境大量的交互与试错，这种成本是很高的。

与强化学习不同的是，离线强化学习利用离线静态数据学习，其不需要与环境交互，可见图1.1所示。若利用离线强化学习对智能体策略模型初始化，再用强化学习的方式进行微调，那么就会让强化学习解决更多具有挑战性的问题。

<div align=center><img width="800" src="./imgs/offline_rl.png" /></div>

<div align='middle'>图1.1 不同的强化学习范式</div>

图1.1中(a)为强化学习范式，也可以称为在线强化学习范式；(b)为off-policy强化学习范式；(c)为离线强化学习范式。



## 模型评估方法





## 参考文献

[1] Prudencio R F, Maximo M R O A, Colombini E L. A survey on offline reinforcement learning: Taxonomy, review, and open problems[J]. IEEE Transactions on Neural Networks and Learning Systems, 2023.
