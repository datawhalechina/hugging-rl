# 行为克隆

行为克隆的核心思想是在拥有专家数据的情况下，通过离线学习的方式获得专家策略。如式(1)所示，行为克隆的数学表达式。
$$
\begin{equation} 
\mathbb{P}(tr)=\rho(s_1)\prod_{h=1}^HP_h(s_{h+1}\vert s_h,a_h)\pi_h(a_h\vert s_h)\tag{1}
\end{equation}
$$
根据式(1)，可知，学习出策略$\pi$就是根据专家样本数据进行参数估计，那么很容易想到最大似然估计。于是，式(1)经过化简，可得式(2)。
$$
\begin{equation}
log(\mathbb{P}(tr))=\sum_{h=1}^Hlog\pi_h(a_h\vert s_h)+constant\tag{2}
\end{equation}
$$
其中，式(2)中constant指与策略$\pi$无关的项，只与初始状态分布、转移函数有关。

 那么，对于数据集$D$中$N$条专家轨迹，则有目标函数式(3)
$$
\max_\pi\sum_{h=1}^{H}\sum_{s_h,a_h\epsilon D}log\pi_h(a_h\vert s_h) \\
\operatorname{ s.t. } 
\sum_a\pi_h(a|s)=1,\forall s\epsilon S,h\epsilon[H].\tag{3}
$$
为了便于理解行为克隆，在表格模型中进行分析，其目标函数(3)的最优解为式(4) 
$$
\begin{equation}
\pi^{BC}\_h(a\vert s)=\begin{cases}
\frac{\#tr\_h(.,.)=(s,a)}{\sum\_{{a}'}\#tr\_h(.,.)=(s,{a}')} & if \sum\_{{a}'}\#tr\_h(.,.)=(s,{a}')>0 \\
\frac{1}{\vert\mathcal{A}\vert} & otherwise
\end{cases}\tag{4}
\end{equation}
$$
根据式(4)，可知，若状态$s$存在与轨迹的第$h$时刻中，那么策略为所有轨迹$h$时刻$(s,a)$个数与$s$个数的比值；若状态$s$不存在于任何轨迹的$h$时刻，那么该状态下所有动作的概率为$\frac{1}{\vert\mathcal{A}\vert}$，$\vert\mathcal{A}\vert$表示所有动作个数，即均匀策略。

