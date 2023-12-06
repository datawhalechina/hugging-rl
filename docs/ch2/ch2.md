# 行为克隆

## 算法的学习方法

行为克隆的核心思想是在拥有专家数据的情况下，通过离线学习的方式获得专家策略。如式(2.1)所示，行为克隆的数学表达式。
$$
\begin{equation} 
\mathbb{P}(tr)=\rho(s_1)\prod_{h=1}^HP_h(s_{h+1}\vert s_h,a_h)\pi_h(a_h\vert s_h)\tag{2.1}
\end{equation}
$$
根据式(2.1)，可知，学习出策略$\pi$就是根据专家样本数据进行参数估计，那么很容易想到最大似然估计。于是，式(2.1)经过化简，可得式(2.2)。
$$
\begin{equation}
log(\mathbb{P}(tr))=\sum_{h=1}^Hlog\pi_h(a_h\vert s_h)+constant\tag{2.2}
\end{equation}
$$
其中，式(2)中constant指与策略$\pi$无关的项，只与初始状态分布、转移函数有关。

 那么，对于数据集$\mathcal{D}$中$N$条专家轨迹，则有目标函数式(2.3)
$$
\max_\pi\sum_{h=1}^{H}\sum_{s_h,a_h\in \mathcal{D}}log\pi_h(a_h\vert s_h) \\
\operatorname{ s.t. } 
\sum_a\pi_h(a|s)=1,\forall s\in S,h\in[H].\tag{2.3}
$$
为了便于理解行为克隆，在表格模型中进行分析，其目标函数(2.3)的最优解如式(2.4) 
$$
\begin{equation}
\pi^{BC}\_h(a\vert s)=\begin{cases}
\frac{\\#tr\_h(.,.)=(s,a)}{\sum\_{{a}'}\\#tr\_h(.,.)=(s,{a}')} & if \sum\_{{a}'}\\#tr\_h(.,.)=(s,{a}')>0 \\\\
\frac{1}{\vert\mathcal{A}\vert} & otherwise
\end{cases}\tag{2.4}
\end{equation}
$$
根据式(2.4)，可知，若状态$s$存在于轨迹的第$h$时刻中，那么策略为所有轨迹$h$时刻$(s,a)$出现个数与$s$出现个数的比值；若状态$s$不存在于任何轨迹的$h$时刻，那么时刻$h$时该状态下所有动作的概率为$\frac{1}{\vert\mathcal{A}\vert}$，$\vert\mathcal{A}\vert$表示所有动作个数，即是均匀策略。

行为克隆算法对于数据集中不包含的状态$s$采取均匀策略，导致策略$\pi$在未访问状态$s$上存在模仿间隔(imitation gap)。那么，误差的上下界是刻画模仿间隔的极限。

此外，直接刻画$s\notin \mathcal{D}$样本对BC算法影响的概念是缺失质量(missing mass)，其数学表达形式如式(2.5)所示。
$$
\begin{equation}
\mathbb{E}[\sum_{h=1}^{H}\sum_{s\epsilon\mathcal{S}}d^{\pi}_{h}(s)\mathbb{I}(s\notin\mathcal{D})]\le\frac{4}{9}\frac{\vert\mathcal{S}\vert H}{N}\tag{2.5}
\end{equation}
$$
根据式(2.5)，可知，若数据集趋向于无穷大，那么缺失质量趋向于0，即算法越接近专家策略；若待学习问题的状态空间或决策长度(Herizon)越大，那么该问题的学习难度越大，需要的样本越多。



## 误差的上下界

在计算学习理论中，与数据分布无关的VC维，以及与数据分布有关的Rademacher复杂度，都是用于表示一个学习任务的难易程度，也可以理解算法对于特定任务的学习能力。与之对应的，刻画BC算法学习能力的是专家策略的累积奖励与BC算法策略累积奖励之间误差的上下界。

### 上界 

**理论二：行为克隆算法的样本复杂度**

*假设智能体有$N$条长度为$H$的专家轨迹，且这些轨迹由确定性专家策略$\pi^E$生成。那么，式(2.3)中行为克隆算法，对于任何表格模型和有限MDP，均有*
$$
\begin{equation}
V(\pi^E)-\mathbb{E}[V(\pi^{BC})]\le\frac{\vert\mathcal{S}\vert H^2}{N} \tag{2.6}
\end{equation}
$$
*式(2.6)中，行为克隆算法的价值函数取期望考虑的是数据收集的随机性。若专家策略为随机策略，那么式(2.6)会多出一个$log{N}$项，相较于$N$，可忽略。*

根据样本复杂度，可得出结论：(1)随着数据集的增多，行为克隆算法学习出的策略越来越接近专家策略；(2)行为克隆算法的最差效果是有界的；(3)与监督学习相比， 行为克隆算法的样本复杂度多出一个$H^2$，这也体现了模仿学习是个序列决策任务。 



### 下界





### 复合误差





 

### 案例







## 参考文献

[1] Rajaraman N , Yang L F , Jiao J ,et al.Toward the Fundamental Limits of Imitation Learning[J].  2020.DOI:10.48550/arXiv.2009.05990.

[2] [Minimax - Wikipedia](https://en.wikipedia.org/wiki/Minimax)

[3] [Minimax estimator - Wikipedia](https://en.wikipedia.org/wiki/Minimax_estimator)
