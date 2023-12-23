# 基于不确定估计的方法与REM

基于不确定性估计的离线强化学习方法允许策略在保守型离线策略与离线策略之间转换。也可以这样理解，若函数近似的不确定性被评估，那么低不确定性区域策略的约束可被松弛。其中，不确定性的估计可以是策略、价值函数、或模型。

以估计$Q$函数不确定性为例，若$\mathcal{P}_{\mathcal{D}}(Q^{\pi})$表示$Q$函数关于数据集$\mathcal{D}$的分布，那么策略梯度的目标函数可写为
$$
\begin{equation}
J(\theta)=\mathbb{E}_{s,a\sim\mathcal{D}}[\mathbb{E}_{Q^{\pi}\sim\mathcal{P}_{\mathcal{D}(.)}}[Q^{\pi}(s,a)-\alpha U_{\mathcal{P}_{\mathcal{D}}}(\mathcal{P}_{\mathcal{D}}(.))]]\tag{4.1}
\end{equation}
$$
式(4.1)中$U_{\mathcal{P}_{\mathcal{D}}}(.)$为Q函数分布$\mathcal{P}_{\mathcal{D}}(.)$的度量，即增加了一个不确定性的惩罚项。





## Random Ensemble Mixture





## 参考文献

[1] Levine S, Kumar A, Tucker G, et al. Offline reinforcement learning: Tutorial, review, and perspectives on open problems[J]. arXiv preprint arXiv:2005.01643, 2020.

[2] Prudencio R F, Maximo M R O A, Colombini E L. A survey on offline reinforcement learning: Taxonomy, review, and open problems[J]. IEEE Transactions on Neural Networks and Learning Systems, 2023.

[3] Agarwal R, Schuurmans D, Norouzi M. An optimistic perspective on offline reinforcement learning[C]//International Conference on Machine Learning. PMLR, 2020: 104-114.