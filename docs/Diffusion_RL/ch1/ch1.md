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

根据图1.1，可知，扩散图模型是隐变量模型，其数学表达可见式(1.1)。其中，$\mathbf{x}\_1,\dots,\mathbf{x}\_T$是与$\mathbf{x}\_0\sim q(\mathbf{x}\_0)$维度相同的隐变量。
$$
\begin{equation}
p\_{\theta}(\mathbf{x}\_0):=\int p\_{\theta}(\mathbf{x}\_{0:T})d\mathbf{x}\_{1:T} \tag{1.1}
\end{equation}
$$
其中，联合分布$p\_{\theta}(\mathbf{x}\_{0:T})$为逆过程。逆过程中马尔科夫链之间的转换函数$p\_{\theta}(\mathbf{x}\_t)$为高斯分布。若起始点$T$的分布函数为$p(\mathbf{x}\_T)=\mathcal{N}(\mathbf{x}\_T;0,\mathbf{I})$，那么
$$
\begin{equation}
p\_{\theta}(\mathbf{x}\_{0:T}):=p(\mathbf{x}\_T)\prod\_{t=1}^Tp\_{\theta}(\mathbf{x}\_{t-1}\vert\mathbf{x}\_t),\qquad p\_{\theta}(\mathbf{x}\_{t-1}\vert\mathbf{x}\_t):=\mathcal{N}(\mathbf{x}\_{t-1};\mu\_{\theta}(\mathbf{x}\_{t-1},t),\Sigma\_{\theta}(\mathbf{x}\_t,t))\tag{1.2}
\end{equation}
$$
与其它隐变量不同，扩散模型的近似后验$q(\mathbf{x}\_{1:T}\vert\mathbf{x}\_0)$，被称为前向过程或扩散过程，是一个固定的马尔科夫链。扩散过程是一个向数据中按照方差为$\beta\_1,\dots,\beta\_{T}$顺序不断增加高斯噪音的过程。
$$
\begin{equation}
q(\mathbf{x}\_{1:T}\vert\mathbf{x}\_0):=\prod\_{t=1}^T q(\mathbf{x}\_t\vert\mathbf{x}\_{t-1}),\qquad q(\mathbf{x}\_t\vert\mathbf{x}\_{t-1}):=\mathcal{N}(\mathbf{x}\_t;\sqrt{1-\beta\_t}\mathbf{x}\_{t-1},\beta\_t\mathbf{I})\tag{1.3}
\end{equation}
$$
模型训练是优化变分界[10]的过程，可见式(1.4)
$$
\begin{equation}
\mathbb{E}[-log{p\_{\theta}(\mathbf{x}\_0)}]\le\mathbb{E}\_q[-log\frac{p\_{\theta}(\mathbf{x}\_{0:T})}{q(\mathbf{x}\_{1:T}\vert \mathbf{x}\_0)}]=\mathbb{E}\_q[-log{p(\mathbf{x}\_T)}-\sum\_{t\ge1}log\frac{p\_{\theta(\mathbf{x}\_{t-1}\vert x\_t)}}{q(\mathbf{x}\_t\vert x\_{t-1})}=:L\tag{1.4}
\end{equation}
$$
前向过程中方差$\beta\_t$可以是被学习出来的，也可以是一个常数。若$\alpha\_t:=1-\beta\_t,\quad\bar{\alpha}\_t:=\prod\_{s=1}^t\alpha\_s$，那么前向过程中任意时间步$t$的$\mathbf{x}\_t$均可在$x\_0$条件下表示
$$
\begin{equation}
q(\mathbf{x}\_t\vert\mathbf{x}\_0)=\mathcal{N}(\mathbf{x}\_t;\sqrt{\bar{\alpha}\_t}\mathbf{x}\_0,(1-\bar{\alpha}\_t)\mathbf{I})\tag{1.5}
\end{equation}
$$
根据文献[2]，可知，若要进一步提升，就是降低损失函数的方差，那么损失函数可被写为
$$
\begin{equation}
\mathbb{E}\_q[\underbrace{D\_{KL}(q(\mathbf{x}\_T\vert\mathbf{x}\_0)\Vert p(\mathbf{x}\_T))}\_{L_T}+\sum\_{t\gt1}\underbrace{D\_{KL}(q(\mathbf{x}\_{t-1}\vert\mathbf{x}\_t,\mathbf{x}\_0)\Vert p\_{\theta}(\mathbf{x}\_{t-1}\vert\mathbf{x}\_t))}\_{L\_{t-1}}-\underbrace{log{p\_{\theta}(\mathbf{x}\_0\vert\mathbf{x}\_1)}}\_{L\_0}]\tag{1.6}
\end{equation}
$$
式(1.6)直接利用KL-Divergence比较$p_{\theta}(\mathbf{x}\_{t-1}\vert\mathbf{x}\_t)$与前向过程的后验比较。在条件$\mathbf{x}_0$下前向过程后验是可计算的
$$
q(\mathbf{x}\_{t-1}\vert\mathbf{x}\_t,\mathbf{x}\_0)=\mathcal{N}(\mathbf{x}\_{t-1};\tilde{\mathbf{\mu}}\_t(\mathbf{x}\_t,\mathbf{x}\_0),\tilde{\beta}\_t\mathbf{I}),\\\\
where\quad\tilde{\mu}\_t(\mathbf{x}\_t,\mathbf{x}\_0):=\frac{\sqrt{\bar{\alpha}\_{t-1}}\beta\_t}{1-\bar{\alpha}\_t}\mathbf{x}\_0+\frac{\sqrt{\alpha\_t}(1-\bar{\alpha}\_{t-1})}{1-\bar{\alpha}\_t}\mathbf{x}\_t\quad and\quad\tilde{\beta}\_t:=\frac{1-\bar{\alpha}\_{t-1}}{1-\bar{\alpha}\_t}\beta\_t\tag{1.7}
$$
根据式(1.7)，可知，式(1.6)中所有KL-Divergence均是高斯分布之间的比较。接下来，理解式(1.6)中三个部分及其计算方法。

### 前向过程与$L\_T$

在文献[2]中并未对方差$\beta_t$参数化，而是把它当作常数对待。因此，$q(\mathbf{x}\_T\vert\mathbf{x}\_0)$与$p(\mathbf{x}\_T)$中无参数需要学习，即$L_T$为常数项。



### 逆过程与$L\_{1:T-1}$

逆过程中转换函数$p\_{\theta}(\mathbf{x}\_{t-1}\vert\mathbf{x}\_t)=\mathcal{N}(\mathbf{x}\_{t-1};\mu\_{\theta}(\mathbf{x}\_t,t),\Sigma\_{\theta}(\mathbf{x}\_t,t))$

正态分布中方差$\Sigma\_{\theta}(\mathbf{x}\_t,t)=\sigma\_t^2\mathbf{I}$被设定为常数。若$\mathbf{x}\_0\sim\mathcal{N}(0,1)$，那么$\sigma\_t^2=\beta\_t$为最优；若$\mathbf{x}\_0$为确定型，那么$\sigma\_t^2=\tilde{\beta}\_t=\frac{1-\bar{\alpha}\_{t-1}}{1-\bar{\alpha}\_t}\beta\_t$为最优。然而，以上两种方差设定方式有相似的结果。

正态分布中均值$\mu\_{\theta}(\mathbf{x}\_t,t)$是参数化项，那么最直接的方式是预测前向过程后验的均值，即
$$
\begin{equation}
L\_{t-1}=\mathbb{E}\_q[\frac{1}{2\sigma\_t^2}\Vert\tilde{\mu}\_t(\mathbf{x}\_t,\mathbf{x}\_0)-\mu\_{\theta}(\mathbf{x}\_t,t)\Vert^2]+C\tag{1.8}
\end{equation}
$$
接下来，介绍均值的第二种参数化方法，也文献[2]重要工作

若$\mathbf{x}\_t(\mathbf{x}\_0,\mathbf{\epsilon})=\sqrt{\bar{\alpha}\_t}\mathbf{x}\_0+\sqrt{1-\bar{\alpha}\_t}\epsilon$且$\epsilon\sim\mathcal{N}(0,\mathbf{I})$，那么
$$
\begin{aligned}
L_{t-1}-C &= \mathbb{E}\_{\mathbf{x}\_{0},\epsilon}[\frac{1}{2\sigma^2\_t}\Vert\tilde{\mu}\_t(\mathbf{x}\_t(\mathbf{x}\_0,\epsilon),\frac{1}{\sqrt{\bar{\alpha}}\_t}(\mathbf{x}\_t(\mathbf{x}\_0,\epsilon)-\sqrt{1-\bar{\alpha}\_t}\epsilon))-\mu\_{\theta}(\mathbf{x}\_t(\mathbf{x}\_0,\epsilon),t)\Vert^2] \\\\
&=\mathbb{E}\_{\mathbf{x}\_0,\epsilon}[\frac{1}{2\sigma\_t^2}\Vert\frac{1}{\sqrt{\alpha\_t}}(\mathbf{x}\_t(\mathbf{x}\_0,\epsilon)-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}}\epsilon)-\mu\_{\theta}(\mathbf{x}\_t(\mathbf{x}\_0,\epsilon),t)\Vert^2]
\end{aligned}\tag{1.9}
$$
可知，在给定$\mathbf{x}\_t$下，$\mu\_{\theta}$预测$\frac{1}{\sqrt{\alpha\_t}}(\mathbf{x}\_t-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}}\_t}\epsilon)$，那么
$$
\begin{equation}
\mu\_{\theta}(\mathbf{x}\_t,t)=\tilde{\mu}\_t(\mathbf{x}\_t,\frac{1}{\sqrt{\bar{\alpha}\_T}}(\mathbf{x}\_t-\sqrt{1-\bar{\alpha}\_t}\epsilon\_{\theta}(\mathbf{x}\_t)))=\frac{1}{\sqrt{\alpha\_t}}(\mathbf{x}\_t-\frac{\beta\_t}{\sqrt{1-\bar{\alpha}}\_t}\epsilon\_{\theta}(\mathbf{x}\_t,t))\tag{1.10}
\end{equation}
$$
式(1.10)中$\epsilon\_{\theta}$是一个函数近似器。可以理解为不是直接参数化均值，而是参数化噪声。那么式(1.9)简化为
$$
\begin{equation}
\mathbb{E}\_{\mathbf{x}\_0,\epsilon}[\frac{\beta\_t^2}{2\sigma\_t^2\alpha\_t(1-\bar{\alpha}\_t)}\Vert\epsilon-\epsilon\_{\theta}(\sqrt{\bar{\alpha}\_t}\mathbf{x}\_0+\sqrt{1-\bar{\alpha}\_t}\epsilon,t)\Vert^2]\tag{1.11}
\end{equation}
$$
式(1.11)类似于每个时刻$t$的去噪匹配，即优化目标函数与去噪匹配相类似。



### 逆过程解码与$L\_0$

假设图片数据由整数集$\{0,1,\ldots,255\}$构成，那么为了保证与模型输入$\mathbf{x}\_T\sim\mathcal{N}(0,\mathbf{I})$一致，把线性缩放到到$[-1,1]$。为了使$L_0$项的输出为独立离散的，那么
$$
p\_{\theta}(\mathbf{x}\_0\vert\mathbf{x}\_1)=\prod\_{i=1}^{D}\int\_{\delta-(x\_0^i)}^{\delta+(x\_0^i)}\mathcal{N}(x;\mu\_{\theta}^i(\mathbf{x}\_1,1),\sigma\_1^2)dx \\\\
\delta+(x)=\begin{cases}
\infty & if\quad x=1 \\\\
x+\frac{1}{255} & if \quad x \lt 1
\end{cases}
\qquad
\delta-(x)=\begin{cases}
-\infty & if\quad x=-1 \\\\
x-\frac{1}{255} & if \quad x \gt -1
\end{cases}\tag{1.12}
$$

<div align="center">
  <img src="./img/ddpm.png" />
</div>
<div align="center">
  图1.2 DDPM的伪代码
</div>

根据图1.2，可知，只需训练一个与时间$t$依赖的噪声近似模型$\epsilon\_{\theta}$，即可得到扩散模型；采样过程与Langevin dynamics[11]类似，$\epsilon\_{\theta}$用于学习数据密度梯度。



## SGMs

根据文献[4]，可知，与基于似然方式的生成模型和生成式对抗模型不同，基于分数的生成模型不需要对抗训练，也不需要在训练时采样。首先，它需要近似高斯噪声扰动后数据梯度的分布函数；然后，利用Annealed Langevin Dynamics生成样本。接下来，利用数学语言描述基于分数的生成模型。

假设从未知数据分布$p\_{data}(\mathbf{x})$采样得到独立同分布的数据集$\{\mathbf{x}\_i\in\mathbb{R}^D\}\_{i=1}^N$。同时，定义

- 概率密度$p(\mathbf{x})$的分数为$\nabla\_{\mathbf{x}}log{p(\mathbf{x})}$
- 分数网络$\mathbf{s}\_{\theta}:\mathbb{R}^D\to\mathbb{R}^D$是一个参数化的神经网络，用于近似$p\_{data}(\mathbf{x})$分数

生成模型的目标是学习一个生成服从分布$p\_{data}(\mathbf{x})$的新样本，那么目标函数为
$$
\begin{equation}
\frac{1}{2}\mathbb{E}\_{p\_{data}}[\Vert\mathbf{s}\_{\theta}(\mathbf{x}）-\nabla\_{\mathbf{x}}log{p\_{data}}(\mathbf{x})\Vert\_2^2]=\mathbb{E}\_{p\_{data}(\mathbf{x})}[tr(\nabla\_{\mathbf{x}}\mathbf{s}\_{\mathbf{\theta}}(\mathbf{x}))+\frac{1}{2}\Vert\mathbf{s}\_{\theta}(\mathbf{x})\Vert\_2^2]+CONSTANT\tag{1.13}
\end{equation}
$$
式(1.13)中$\nabla\_{\mathbf{x}}\mathbf{s}\_{\theta}(\mathbf{x})$为$\mathbf{s}_{\mathbf{\theta}}(\mathbf{x})$的雅可比矩阵。然而，求解该目标函数会遇到以下挑战：

- 由于雅可比矩阵迹计算复杂，分数匹配无法扩展到深度神经网络和高维数据。

- 由于现实任务的数据往往处于低维流形，数据分布的梯度是未定义的。
- 在数据分布密度低的区域，往往因数据不足造成估计不准确。
- 若数据分布的两种模式被低维数据分布密度区域分开，Langevin Dynamics无法在有限时间内恢复两种模式的相对权重。

为了解决雅可比矩阵计算复杂的问题，利用高斯分布梯度计算方便，对数据扰动。同时，数据扰动也能解决数据分布处于低维流形的问题，也减少了数据分布密度低的区域。为了解决Langevin Dynamics无法区分数据模式权重的问题，利用annealed Langevin Dynamics进行数据生成。

若$\sigma_i$为高斯分布的方差，且集合$\{\sigma\_i\}\_{i=1}^L$中元素满足$\frac{\sigma\_1}{\sigma\_2}=\cdots=\frac{\sigma\_{L-1}}{\sigma\_L}\gt1$，那么扰动后的数据分布为$q\_{\sigma}(\mathbf{x})=\int p\_{data}(\mathbf{t})\mathcal{N}(\mathbf{x}\vert\mathbf{t},\sigma^2I)$。同时，为了应对以上挑战，$\sigma\_1$应足够的大，$\sigma\_L$应足够大。那么，分数网络应该估计数据扰动后分布的梯度，即$\forall\sigma\in\{\sigma\_i\}_{i=1}^L:\mathbf{s}\_{\theta}(\mathbf{x},\sigma)\approx\nabla\_{\mathbf{x}}log{q\_{\sigma}(\mathbf{x})}$。此时，$\mathbf{s}\_{\theta}(\mathbf{x},\sigma)$称为Noise Conditional Score Network(NCSN)，目标函数为
$$
\begin{equation}
l(\theta;\sigma)=\frac{1}{2}\mathbb{E}\_{p\_{data}(\mathbf{x})}\mathbb{E}\_{\tilde{x}\sim\mathcal{N}(\mathbf{x},\sigma^2 I)}[\Vert\mathbf{s}\_{\theta}(\tilde{\mathbf{x}},\sigma)+\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}\Vert\_2^2]\tag{1.14}
\end{equation}
$$
对于所有的$\sigma\_i$，损失函数为
$$
\begin{equation}
\mathcal{L}(\mathbf{\theta};\{\sigma\_i\}\_{i=1}^L)=\frac{1}{L}\sum\_{i=1}^{L}\lambda(\sigma\_i)l(\mathbf{\theta};\sigma\_i)\tag{1.15}
\end{equation}
$$
式(1.15)中$\lambda(\sigma\_i)=\sigma\_i^2$。以此为目标函数训练神经网络，就可以得到数据分布梯度的估计函数。

根据分数估计函数，生成样本的Annealed Langevin dynamics算法伪代码如图1.3所示

<div align="center">
  <img src="./img/ald.png" height=300/>
</div>
<div align="center">
  图1.3 Annealed Langevin dynamics算法伪代码
</div>






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

[11] [Generative Modeling by Estimating Gradients of the Data Distribution | Yang Song (yang-song.net)](https://yang-song.net/blog/2021/score/)

