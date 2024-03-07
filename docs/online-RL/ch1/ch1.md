# MCAC：稀疏奖励环境下Monte-Carlo增强的Actor-Critic算法

现实世界中，强化学习算法面对的往往是稀疏奖励环境。在稀疏奖励环境中，对探索产生了很大的挑战。这是因为稀疏奖励函数使智能体无法有意义的区分不同策略之间的区别。对稀疏奖励函数问题，处理该问题常见的方法是利用专家演示数据提供状态空间中高奖励区域的初始信号。然而，之前基于演示的方式往往使算法本身变得很复杂，且增加了实施以及调节超参数的难度。文献[1]作者们另辟蹊径，提出了MCAC算法了，既没有过多的增加模型复杂性，也没有增加额外的超参数。

MCAC算法背后的核心思想是：鼓励智能体在次优轨迹的邻域内保持初始乐观，且随着算法学习不断减少乐观以鼓励探索新的行为。具体实现方式为：MCAC 中引入了两个修改，分别是基于专家演示数据初始化replay buffer和选择TD-learning中价值估计的目标值与Monte-Carlo价值估计目标值之间的最大值作为Q值的目标值。之所以利用Monte-Carlo价值估计，这是因为MC方法更能够高效的捕获长期奖励信息，使奖励信息得到更快的传递。

## Monte-Carlo增强的Actor-Critic

### MCAC算法

如图1所示，MCAC算法伪代码。

<div align="center">
  <img src="https://www.robotech.ink/usr/uploads/2024/02/2087342808.png" width=600 />
</div>
<div align="center">
  图1 MCAC算法
</div>

图1中式(5.2)对应于本文的式(2)，式(5.4)对应于本文的式(4)。

TD-Learning由于基于时序差分的方式，很难传播奖励信息，所以在学习早期的价值估计很低。另一方面，Monte-Carlo方式的目标值虽然很容易捕获长期奖励，但是对表现稍差的轨迹会严重的低估$Q$值，即MC为无偏估计而方差大。与MC不同，TD-Learning属于有偏估计但方差低。在MCAC算法，选择MC目标值与TD-Learning目标值之间的最大值能够降低MC对高价值区域的邻域的过度低估问题。



### MCAC算法的实际实现

MCAC可被看成actor-critic算法的包装。MCAC算法首先基于次优演示数据$\mathcal{D}_{offline}$初始化Replay Buffer。然后，在每个episode，收集一个完整轨迹$\tau^i$，轨迹中的第$j$个转换为$(s_j^i,a_j^i,s_{j+1}^i,r_j^i)$。接下来，基于任何actor-critic方法学习$Q$函数的近似$Q_{\theta}(s_t,a_t)$。对于给定的转换$\tau_j^{i}=(s_j^i,a_j^i,s_{j+1}^i,r_j^i)\in\tau^i\subsetneq\mathcal{R}$，其损失函数为
$$
\begin{aligned}
J(\theta)=l(Q_{\theta}(s_j^i,a_j^i),Q^{target}(s_j^i,a_j^i))
\end{aligned}\tag{1}
$$
为了实现MCAC，文献[1]作者们首先校准了MC目标值。MC目标值校准的前提假设是：最后观测到的奖励值会一直重复，具体可见式(2)
$$
\begin{aligned}
Q^{target}_{MC-\infty}(s_j^i,a_j^i)=\gamma^{t-j+1}\frac{r^i_T}{1-\gamma}+\sum_{k=j}^T\gamma^{k-j}r(s_k^i,a_k^i)
\end{aligned}\tag{2}
$$
MCAC中$Q$函数的目标值只是原始目标值与MC目标值的最大化，可见式(3)
$$
\begin{aligned}
Q^{target}_{MCAC}(s_j^i,a_j^i)=max[Q^{target}(s_j^i,a_j^i),Q^{target}_{MC-\infty}(s_j^i,a_j^i)]
\end{aligned}\tag{3}
$$
最终，得到$Q$函数的损失函数为
$$
\begin{aligned}
J(\theta)=l(Q_{\theta}(s_j^i,a_j^i),Q^{target}_{MCAC}(s_j^i,a_j^i))
\end{aligned}\tag{4}
$$


### MCAC算法效果

为了更好理解MCAC影响Q值估计的方法，基于SAC算法在Pointmass Navigation环境中训练50000步之后的Replay Buffer中$Q$值估计进行了可视化，可见图2所示。

<div align="center">
  <img src="https://www.robotech.ink/usr/uploads/2024/02/4272164604.png" width=600 />
</div>
<div align="center">
  图2 MCAC Replay-Buffer可视化
</div>

图2中Bellman Q Estimate是基于Bellman方程运算的Q值估计，也就是TD-Learning；GQE为文献[2]中GAE的方式计算Q值估计方法；MCMC的Q值估计就是文献[1]的估计方法。

根据图2中上面一行，可知，无MCAC的SAC智能体无法学习到有用的$Q$函数，因此无法学习出完成任务的策略。同时，也可以看到GAE比Bellman的方式有效，但无MCAC有效。根据图2中下面一行，可知，若智能体利用了MCAC，那么可学习到有用的Q函数，从而可靠的完成任务，其Replay Buffer中Bellman估计、GAE估计以及MCAC估计相似。

<div align="center">
  <img src="https://www.robotech.ink/usr/uploads/2024/02/3706850504.png" width=600 />
</div>
<div align="center">
  图3 MCAC与标准RL算法结果
</div>

根据图3，可知，与文献[5]的SAC、文献[6]的TD3、以及文献[2]的GAE相比，MCAC增强的版本比原始版本算法性能优越，且样本效率高。

<div align="center">
  <img src="https://www.robotech.ink/usr/uploads/2024/02/956296637.png" width=600 />
</div>
<div align="center">
  图4 MCAC与基于专家演示的RL算法结果
</div>

根据图4，可知，与文献[3]的OEDF、文献[4]的AWAC、以及文献[7]的CQL这类基于专家演示数据初始化策略的算法相比，MCAC增强的版本比原始版本算法性能优越，样本效率高，且训练更稳定。即使部分环境没有提升效果，但是也没有降低算法性能，即无负面影响。

## 参考文献

[1] Wilcox A, Balakrishna A, Dedieu J, et al. Monte carlo augmented actor-critic for sparse reward deep reinforcement learning from suboptimal demonstrations[J]. Advances in Neural Information Processing Systems, 2022, 35: 2254-2267.

[2] Schulman J, Moritz P, Levine S, et al. High-dimensional continuous control using generalized advantage estimation[J]. arXiv preprint arXiv:1506.02438, 2015.

[3] Nair A, McGrew B, Andrychowicz M, et al. Overcoming exploration in reinforcement learning with demonstrations[C]//2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018: 6292-6299.

[4] Nair A, Gupta A, Dalal M, et al. Awac: Accelerating online reinforcement learning with offline datasets[J]. arXiv preprint arXiv:2006.09359, 2020.

[5] Haarnoja T, Zhou A, Abbeel P, et al. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor[C]//International conference on machine learning. PMLR, 2018: 1861-1870.

[6] Fujimoto S, Hoof H, Meger D. Addressing function approximation error in actor-critic methods[C]//International conference on machine learning. PMLR, 2018: 1587-1596.

[7] Kumar A, Zhou A, Tucker G, et al. Conservative q-learning for offline reinforcement learning[J]. Advances in Neural Information Processing Systems, 2020, 33: 1179-1191.