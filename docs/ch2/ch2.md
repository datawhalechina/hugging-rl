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
\max_\pi\sum_{h=1}^{H}\sum_{s_h,a_h\in \mathcal{D}}log\pi_h(a_h\vert s_h) \\\\
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



## 有限长度回合制MDP下的统计极限

*引理2.1 在无交互场景下，专家策略$\pi^{E}$为确定型时，对于行为克隆算法的任何策略$\pi^{BC}$与专家策略不一致的概率有界于$min\{1,\frac{\vert\mathcal{S}\vert}{N}\}$，即式(2.6)*
$$
\begin{equation}
\mathbb{E}[\frac{1}{H}\sum_{t=1}^{H}\mathbb{E}\_{s_t\sim f^t_{\pi^{E}}}[\mathbb{E}\_{a\sim\pi_t^{BC}(.\vert s_t)}[\mathbb{I}(a\neq\pi_t^{E}(s_t))]]]\le min\\{1,\frac{\vert\mathcal{S}\vert}{N}\\}\tag{2.6}
\end{equation}
$$
策略$\hat{\pi}$与专家策略不一致概率，主要是在专家策略下不会产生的状态，即BC算法不会学习到的策略，因此不一致概率有界于$\frac{\vert\mathcal{S}\vert}{N}$。

### 上界 

**定理2.2：** *假设智能体有$N$条长度为$H$的专家轨迹，且这些轨迹由确定性专家策略$\pi^E$生成。那么，行为克隆算法产生策略$\pi^{BC}$*

- *专家策略与BC策略之间误差有界于*

$$
\begin{equation}
J(\pi^E)-\mathbb{E}[J(\pi^{BC})]\le min\\{H,\frac{\vert\mathcal{S}\vert H^2}{N}\\} \tag{2.7}
\end{equation}
$$
- *对于任何$\delta\in(0,min\{1,\frac{H}{10}\})$，以$1-\delta$的概率误差有界于*

$$
\begin{equation}
J(\pi^E)-J(\pi^{BC})\le\frac{\vert\mathcal{S}\vert H^2}{N}+\frac{\sqrt{\vert\mathcal{S}\vert}H^2log(\frac{H}{\delta})}{N}\tag{2.8}
\end{equation}
$$

式(2.7)与(2.8)是建立在引理2.1之上，即误差主要来源于策略$\pi^{BC}$未访问的状态，也是专家不会访问的状态。根据误差上界，可得结论：(1)随着数据集的增多，行为克隆算法学习出的策略越来越接近专家策略；(2)行为克隆算法的最差效果是有界的。



### 下界

**定理2.3：** *在无交互场景下，对于智能体的策略$\pi^{BC}$，存在一个马尔科夫决策过程和确定型专家策略$\pi^E$，策略$\pi^{BC}$的下界为*
$$
\begin{equation}
J(\pi^E)-E[J(\pi^{BC})] \ge min\\{H, \frac{\vert\mathcal{S}\vert H^2}{N}\\}\tag{2.9}
\end{equation}
$$


### 极小极大最优

根据定理2.2和2.3可知，有限长度回合制马尔科夫决策过程下的行为克隆算法上下界均为$\hat{O}(\frac{\vert\mathcal{S}\vert H^2}{\epsilon})$，表明该算法为极小极大最优。

 

## 无限长度折扣MDP下的统计极限

文献[2~3]对无限长度的折扣马尔科夫决策过程下行为克隆算法的上下界进行了推导与证明。

### 上界

**定理2.4:**  *若专家策略$\pi_{E}$与模仿策略$\pi^{BC}$之间满足式(2.10)*
$$
\begin{equation}
\mathbb{E}\_{s\sim d_{\pi_{E}}}[D_{KL}(\pi_E(.\vert s),\pi_{BC}(.\vert s))] \le \epsilon\tag{2.10}
\end{equation}
$$
*那么，BC算法误差上界为*
$$
\begin{equation}
V_{\pi_{E}}-V_{\pi_{BC}}\le\frac{2\sqrt{2}R_{max}}{(1-\gamma^{2})}\sqrt{\epsilon}\tag{2.11}
\end{equation}
$$


### 下界

文献[3]对无交互场景下模仿学习的下界给出了一个命题。

**命题2.1:** 给定专家数据$D=\{(s_{\pi_E}^{i},a_{\pi_E}^i)\}\_{i=1}^m$，对于任何算法$Alg: D\to \pi$，存在一个MDP $\mathcal{M}$和专家策略$\pi_E$，有
$$
\begin{equation}
V^{\mathcal{M}}\_{\pi_E}-V^{\mathcal{M}}\_{\pi}\ge(\frac{1}{1-\gamma},\frac{\vert\mathcal{S}\vert}{(1-\gamma^2)m})
\end{equation}
$$
根据定理2.4和命题2.1可知，无限长度折扣马尔科夫决策过程下的行为克隆算法上下界均为$\hat{O}(\frac{\vert\mathcal{S}\vert}{(1-\gamma^2)\epsilon})$，即表明行为克隆算法在环境转移概率未知的设定下，是极小极大最优算法。

无论是有限长度回合制马尔科夫决策过程，还是无限长度的折扣马尔科夫决策过程，误差的上下界是根据特例推导而来，使用的特例是Reset Cliff。



## 参考文献

[1] Rajaraman N , Yang L F , Jiao J ,et al.Toward the Fundamental Limits of Imitation Learning[J].  2020.DOI:10.48550/arXiv.2009.05990.

[2] Xu T , Li Z , Yu Y .Error Bounds of Imitating Policies and Environments[J].  2020.DOI:10.48550/arXiv.2010.11876.

[3] Xu T, Li Z, Yu Y. Error bounds of imitating policies and environments for reinforcement learning[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, 44(10): 6968-6980.

[4] [Minimax - Wikipedia](https://en.wikipedia.org/wiki/Minimax)

[5] [Minimax estimator - Wikipedia](https://en.wikipedia.org/wiki/Minimax_estimator)
