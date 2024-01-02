# 行为克隆

行为克隆把学习专家策略的问题当作监督学习任务，直接学习从环境状态到专家动作的映射。该学习范式的优势是：它不需要环境状态转换的知识，仅根据被提供的专家演示数据学习专家策略。同时，这种监督学习范式是很高效的。因此，最直接的建模方式是极大似然估计。如式(2.1)所示，行为克隆的数学表达式。
$$
\begin{equation} 
\mathbb{P}(tr)=\rho(s_1)\prod_{h=1}^HP_h(s_{h+1}\vert s_h,a_h)\pi_h(a_h\vert s_h)\tag{2.1}
\end{equation}
$$
式(2.1)经过化简，可得式(2.2)。
$$
\begin{equation}
log(\mathbb{P}(tr))=\sum_{h=1}^Hlog\pi_h(a_h\vert s_h)+constant\tag{2.2}
\end{equation}
$$
其中，式(2)中constant指与策略$\pi$无关的项，只与初始状态分布、转移函数有关。

 那么，对于数据集$\mathcal{D}$中$N$条专家轨迹，则有目标函数式(2.3)
$$
\max_\pi\sum_{h=1}^{H}\sum_{s_h,a_h\in \mathcal{D}}log\pi_h(a_h\vert s_h) \\\\
\operatorname{ s.t. } 
\sum_a\pi_h(a|s)=1,\forall s\in S,h\in[H].\tag{2.3}
$$
在表格模型中，目标函数(2.3)的最优解如式(2.4) 
$$
\begin{equation}
\pi^{BC}\_h(a\vert s)=\begin{cases}
\frac{\\#tr\_h(.,.)=(s,a)}{\sum\_{{a}'}\\#tr\_h(.,.)=(s,{a}')} & if \sum\_{{a}'}\\#tr\_h(.,.)=(s,{a}')>0 \\\\
\frac{1}{\vert\mathcal{A}\vert} & otherwise
\end{cases}\tag{2.4}
\end{equation}
$$
根据式(2.4)，可知，若状态$s$存在于轨迹的第$h$时刻中，那么策略为所有轨迹$h$时刻状态-动作$(s,a)$出现个数与状态$s$出现个数的比值；若状态$s$不存在于任何轨迹的$h$时刻，那么时刻$h$该状态下所有动作的概率为$\frac{1}{\vert\mathcal{A}\vert}$，$\vert\mathcal{A}\vert$表示所有动作个数，即是均匀策略。对于专家数据集来说，若状态$s$不出现或极少出现，代表着专家知道在该类状态$s$并不会导致累积收益最大化，反而对收益产生影响。因此，行为克隆算法对于专家数据集中不包含的状态$s$采取均匀策略，导致策略$\pi$在未访问状态$s$上存在模仿间隔(imitation gap)，也被称为**协变量偏移**。有的文献也把协变量偏移问题称为**复合误差**。

协变量偏移问题是行为克隆一个很重要的劣势。更准确的说，该问题产生的原因是：在模型训练期间，学习器根据专家策略产生的状态学习；然而，在自己策略产生的状态进行测试。因此，测试期间观测到的状态分布与训练期间观测的状态分布不一致导致协变量偏移问题发生。同时，解决协变量偏移问题在安全性要求很高的领域是重要的，例如：若自动驾驶汽车遇到训练时未访问过的状态，那么从错误中恢复的能力对于避免事故很重要。

## 协变量偏移问题的处理

为了提供行为克隆算法的鲁棒性和处理协变量偏移问题，三个广泛研究的领域被提出(如图2.1所示)。

<div align="center">
  <img src="./img/cvs.png", height=300>
</div>
<div align="center">
  图2.1 协变量偏移问题处理方法的分类
</div>

接下来，主要介绍一下交互式模仿学习和约束性模仿学习，而逆强化学习会在第三章详细介绍。

### 交互式模仿学习

交互式模仿学习在一些文献中也被称为主动学习。



### 约束性模仿学习





## 参考文献

[1]Zare M, Kebria P M, Khosravi A, et al. A survey of imitation learning: Algorithms, recent developments, and challenges[J]. arXiv preprint arXiv:2309.02473, 2023.

