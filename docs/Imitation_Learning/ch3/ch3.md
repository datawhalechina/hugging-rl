# 逆强化学习

逆强化学习(Inverse Reinforcement Learning)是模仿学习另一个重要的学习范式。首先，该学习范式根据专家数据学习出奖励函数；然后，智能体与环境交互，以强化学习的方式最大化累积收益，奖励函数为其提供强化信号。目前，逆强化学习已经被应用在机器人操纵、自动驾驶、游戏、以及自然语言处理领域。



## 面对的挑战及其处理方法

**挑战一**：首先，由于智能体需要与环境交互，因此逆强化学习是计算成本高且资源敏感的学习范式。同时，对于高风险领域，这是不安全的，例如：自动驾驶、飞机控制。其次，逆强化学习是一个奖励函数估计与策略学习不断迭代的学习范式，即样本效率较低。文献[2]利用人类引导的方式减少交互次数，从而提高样本效率。

**挑战二**：逆强化学习中策略与奖励函数之间的关系是模糊不清楚。确切来说，一个策略可以对大量的奖励函数是最优的。

### 模糊性挑战的应对方法

为了应对逆强化学习中策略与奖励函数之间关系模糊不清楚，研究人员在奖励函数中引入了额外结构。根据文献[3]，可知，引入额外结构的方法可以分为三类，分别是最大化边际方法(maximum-margin methods)、最大化策略熵、以及贝叶斯算法。接下来，对以下三种方法分别进行介绍。

#### 最大化边际方法

该类算法的核心思想是：相较于解释其它策略，奖励函数应能够更彻底的解释最优策略。文献[4]，对于给定最优策略估计奖励函数，同时最大化边际；文献[5]寻找一个有权重的特征与奖励之间的线性映射，以便于估计策略与演示策略的接近程度。这种基于特征的奖励函数产生了各种利用特征期望进行边际优化的方法。文献[6]提出了两种功能性方法：最大化边际和投影，即在不获取专家策略的情况下，最大化特征期望损失边际。



#### 最大化策略熵方法

文献[7]中，MaxEntIRL是第一个基于最大熵的逆强化学习算法。同时，文献[7]中，表明，最大化熵的范式能够应对专家策略的次优性和随机性。随后，文献[8]和[9]，利用利用路径积分法把MaxEntIRL扩展到连续状态-动作空间领域。虽然线性模型得到的奖励函数能够适用于大多数领域，但是不能应对复杂的环境。文献[10]，利用深度学习形成了最大化熵的深度逆强化学习，使算法能够应对复杂的环境。然而，最大熵深度逆强化学习依赖大量的专家样本，即样本效率不高。文献[11]，提出了GCL算法以应对Deep IRL样本效率不高的问题。



#### 贝叶斯算法

简单来说，这类算法利用专家策略形成的数据更新奖励函数的先验。文献[12]中，BIRL是最早的贝叶斯逆强化学习，它利用玻尔兹曼分布建模奖励函数的似然；利用Beta分布建模奖励函数的先验。同时，为了能够在奖励函数的连续空间计算后验，利用MCMC得到后验均值的估计。由于产生后验样本需要解马尔科夫决策过程，因此以上基于贝叶斯的算法很难扩展到连续动作空间。为了克服这种限制，文献[13]利用专家演示数据的偏好标签提出了一个似然方程，用于从后验分布生成样本，该算法被命名为Bayesian REX(Bayesian Reward Extrapolation)；文献[14]利用变分推断近似后验。



## 参考文献

[1] Zare M, Kebria P M, Khosravi A, et al. A survey of imitation learning: Algorithms, recent developments, and challenges[J]. arXiv preprint arXiv:2309.02473, 2023.

[2] Hadfield-Menell D, Russell S J, Abbeel P, et al. Cooperative inverse reinforcement learning[J]. Advances in neural information processing systems, 2016, 29.

[3] Jarrett D, Hüyük A, Van Der Schaar M. Inverse decision modeling: Learning interpretable representations of behavior[C]//International Conference on Machine Learning. PMLR, 2021: 4755-4771.

[4] Ng A Y, Russell S. Algorithms for inverse reinforcement learning[C]//Icml. 2000, 1: 2.

[5] Ratliff N D, Bagnell J A, Zinkevich M A. Maximum margin planning[C]//Proceedings of the 23rd international conference on Machine learning. 2006: 729-736.

[6] Abbeel P, Ng A Y. Apprenticeship learning via inverse reinforcement learning[C]//Proceedings of the twenty-first international conference on Machine learning. 2004: 1.

[7] Ziebart B D, Maas A L, Bagnell J A, et al. Maximum entropy inverse reinforcement learning[C]//Aaai. 2008, 8: 1433-1438.

[8] Aghasadeghi N, Bretl T. Maximum entropy inverse reinforcement learning in continuous state spaces with path integrals[C]//2011 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2011: 1561-1566.

[9] Kalakrishnan M, Pastor P, Righetti L, et al. Learning objective functions for manipulation[C]//2013 IEEE International Conference on Robotics and Automation. IEEE, 2013: 1331-1336.

[10] Wulfmeier M, Ondruska P, Posner I. Maximum entropy deep inverse reinforcement learning[J]. arXiv preprint arXiv:1507.04888, 2015. 

[11] Finn C, Levine S, Abbeel P. Guided cost learning: Deep inverse optimal control via policy optimization[C]//International conference on machine learning. PMLR, 2016: 49-58.

[12] Ramachandran D, Amir E. Bayesian Inverse Reinforcement Learning[C]//IJCAI. 2007, 7: 2586-2591.

[13] Brown D, Coleman R, Srinivasan R, et al. Safe imitation learning via fast bayesian reward inference from preferences[C]//International Conference on Machine Learning. PMLR, 2020: 1165-1177.

[14] Chan A J, van der Schaar M. Scalable bayesian inverse reinforcement learning[J]. arXiv preprint arXiv:2102.06483, 2021.
