# 扩散模型基础

扩散模型是一类概率生成模型，它通过注入噪声逐步破坏数据，然后学习其逆过程，以生成样本。目前，扩散模型的研究主要有三种方式：去噪扩散概率模型[2,3] (Denoising Diffusion Probabilistic Models, 简称DDPMs)、基于分数的生成模型[4,5] (Score-Based Generative Models,简称SGMs)、随机微分方程[6,7,8] (Stochastic Differential Equations,简称Score SDEs)。

## DDPMs

扩散概率模型是一个被参数化的马尔科夫链，在有限时间内通过变分推断训练模型生成样本。它的学习范式可分为两个过程，一个是不断的向样本数据中加入高斯噪音直至破坏样本的**扩散过程**；另一个是逆扩散过程，也称**逆过程**，该过程需要学习马尔科夫链中状态之间的转换函数。如图1.1所示，扩散模型的有向图模型。

<div align="center">
  <img src="./img/directed_graph.png" />
</div>
<div align="center">
  图1.1 扩散模型的有向图模型
</div>

根据图1.1，可知，扩散图模型是隐变量模型，其数学表达可见式(1.1)。其中，$\mathbf{x}_1,\dots,\mathbf{x}_T$是与$\mathbf{x}_0\sim q(\mathbf{x}_0)$维度相同的隐变量。
$$
\begin{equation}
p_{\theta}(\mathbf{x}_0):=\int p_{\theta}(\mathbf{x}_{0:T})d\mathbf{x}_{1:T} \tag{1.1}
\end{equation}
$$
其中，联合分布$p_{\theta}(\mathbf{x}_{0:T})$为逆过程。逆过程中马尔科夫链之间的转换函数$p_{\theta}(\mathbf{x}_t)$为高斯分布。若起始点$T$的分布函数为$p(\mathbf{x}_T)=\mathcal{N}(\mathbf{x}_T;0,\mathbf{I})$，那么
$$
\begin{equation}
p_{\theta}(\mathbf{x}_{0:T}):=p(\mathbf{x}_T)\prod_{t=1}^Tp_{\theta}(\mathbf{x}_{t-1}\vert\mathbf{x}_t),\qquad p_{\theta}(\mathbf{x}_{t-1}\vert\mathbf{x}_t):=\mathcal{N}(\mathbf{x}_{t-1};\mu_{\theta}(\mathbf{x}_{t-1},t),\Sigma_{\theta}(\mathbf{x}_t,t))\tag{1.2}
\end{equation}
$$
与其它隐变量不同，扩散模型的近似后验$q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)$，被称为前向过程或扩散过程，是一个固定的马尔科夫链。扩散过程是一个向数据中按照方差为$\beta_1,\dots,\beta_{T}$顺序不断增加高斯噪音的过程。
$$
\begin{equation}
q(\mathbf{x}_{1:T}\vert\mathbf{x}_0):=\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1}),\qquad q(\mathbf{x}_t\vert\mathbf{x}_{t-1}):=\mathcal{N}(\mathbf{x}_t;\sqrt{1-\beta_t}\mathbf{x}_{t-1},\beta_t\mathbf{I})\tag{1.3}
\end{equation}
$$
模型训练是优化变分界[10]的过程，可见式(1.4)
$$
\begin{equation}
\mathbb{E}[-log{p_{\theta}(\mathbf{x}_0)}]\le\mathbb{E}_q[-log\frac{p_{\theta}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\vert \mathbf{x}_0)}]=\mathbb{E}_q[-log{p(\mathbf{x}_T)}-\sum_{t\ge1}log\frac{p_{\theta(\mathbf{x}_{t-1}\vert x_t)}}{q(\mathbf{x}_t\vert x_{t-1})}=:L\tag{1.4}
\end{equation}
$$
前向过程中方差$\beta_t$可以是被学习出来的，也可以是一个常数。若$\alpha_t:=1-\beta_t,\quad\bar{\alpha}_t:=\prod_{s=1}^t\alpha_s$，那么前向过程中任意时间步$t$的$\mathbf{x}_t$均可在$x_0$条件下表示
$$
\begin{equation}
q(\mathbf{x}_t\vert\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_t;\sqrt{\bar{\alpha}_t}\mathbf{x}_0,(1-\bar{\alpha}_t)\mathbf{I})\tag{1.5}
\end{equation}
$$
根据文献[2]，可知，若要进一步提升，就是降低损失函数的方差，那么损失函数可被写为
$$
\begin{equation}
\mathbb{E}_q[\underbrace{D_{KL}(q(\mathbf{x}_T\vert\mathbf{x}_0)\Vert p(\mathbf{x}_T))}_{L_T}+\sum_{t\gt1}\underbrace{D_{KL}(q(\mathbf{x}_{t-1}\vert\mathbf{x}_t,\mathbf{x}_0)\Vert p_{\theta}(\mathbf{x}_{t-1}\vert\mathbf{x}_t))}_{L_{t-1}}-\underbrace{log{p_{\theta}(\mathbf{x}_0\vert\mathbf{x}_1)}}_{L_0}]\tag{1.6}
\end{equation}
$$
式(1.6)直接利用KL-Divergence比较$p_{\theta}(\mathbf{x}_{t-1}\vert\mathbf{x}_t)$与前向过程的后验比较。在条件$\mathbf{x}_0$下前向过程后验是可计算的
$$
q(\mathbf{x}_{t-1}\vert\mathbf{x}_t,\mathbf{x}_0)=\mathcal{N}(\mathbf{x}_{t-1};\tilde{\mu}_t(\mathbf{x}_t,\mathbf{x}_0),\tilde{\beta}_t\mathbf{I}),\\\\
where\quad\tilde{\mu}_t(\mathbf{x}_t,\mathbf{x}_0):=\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0+\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t\quad and\quad\tilde{\beta}_t:=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t\tag{1.7}
$$




## SGMs







## Score SDEs







## 参考文献

[1] Yang L, Zhang Z, Song Y, et al. Diffusion models: A comprehensive survey of methods and applications[J]. ACM Computing Surveys, 2023, 56(4): 1-39.

[2] Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in neural information processing systems, 2020, 33: 6840-6851.

[3] Sohl-Dickstein J, Weiss E, Maheswaranathan N, et al. Deep unsupervised learning using nonequilibrium thermodynamics[C]//International conference on machine learning. PMLR, 2015: 2256-2265.

[4] Song Y, Ermon S. Generative modeling by estimating gradients of the data distribution[J]. Advances in neural information processing systems, 2019, 32.

[5] Song Y, Ermon S. Improved techniques for training score-based generative models[J]. Advances in neural information processing systems, 2020, 33: 12438-12448.

[6] Karras T, Aittala M, Aila T, et al. Elucidating the design space of diffusion-based generative models[J]. Advances in Neural Information Processing Systems, 2022, 35: 26565-26577.

[7] Song Y, Durkan C, Murray I, et al. Maximum likelihood training of score-based diffusion models[J]. Advances in Neural Information Processing Systems, 2021, 34: 1415-1428.

[8] Song Y, Ermon S. Generative modeling by estimating gradients of the data distribution[J]. Advances in neural information processing systems, 2019, 32.

[9] [Rao–Blackwell theorem - Wikipedia](https://en.wikipedia.org/wiki/Rao–Blackwell_theorem)

[10] [Evidence lower bound - Wikipedia](https://en.wikipedia.org/wiki/Evidence_lower_bound#Maximizing_the_ELBO)

