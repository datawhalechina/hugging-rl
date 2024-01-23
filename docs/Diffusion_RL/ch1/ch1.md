`# 扩散模型基础

扩散模型是一类概率生成模型，它通过注入噪声逐步破坏数据，然后学习其逆过程，以生成样本。目前，扩散模型主要有三种形式：去噪扩散概率模型[2,3] (Denoising Diffusion Probabilistic Models, 简称DDPMs)、基于分数的生成模型[4,5] (Score-Based Generative Models,简称SGMs)、随机微分方程[6,7,8] (Stochastic Differential Equations,简称Score SDEs)。

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

若$\sigma_i$为高斯分布的方差，且集合$\\{\sigma\_i\\}\_{i=1}^L$中元素满足$\frac{\sigma\_1}{\sigma\_2}=\cdots=\frac{\sigma\_{L-1}}{\sigma\_L}\gt1$，那么扰动后的数据分布为$q\_{\sigma}(\mathbf{x})=\int p\_{data}(\mathbf{t})\mathcal{N}(\mathbf{x}\vert\mathbf{t},\sigma^2I)$。同时，为了应对以上挑战，$\sigma\_1$应足够的大，$\sigma\_L$应足够大。那么，分数网络应该估计数据扰动后分布的梯度，即$\forall\sigma\in\\{\sigma\_i\\}_{i=1}^L:\mathbf{s}\_{\theta}(\mathbf{x},\sigma)\approx\nabla\_{\mathbf{x}}log{q\_{\sigma}(\mathbf{x})}$。此时，$\mathbf{s}\_{\theta}(\mathbf{x},\sigma)$称为Noise Conditional Score Network(NCSN)，对于每个给定的$\sigma$，目标函数为
$$
\begin{equation}
l(\theta;\sigma)=\frac{1}{2}\mathbb{E}\_{p\_{data}(\mathbf{x})}\mathbb{E}\_{\tilde{x}\sim\mathcal{N}(\mathbf{x},\sigma^2 I)}[\Vert\mathbf{s}\_{\theta}(\tilde{\mathbf{x}},\sigma)+\frac{\tilde{\mathbf{x}}-\mathbf{x}}{\sigma^2}\Vert\_2^2]\tag{1.14}
\end{equation}
$$
对于所有的$\sigma\_i$，损失函数为
$$
\begin{equation}
\mathcal{L}(\mathbf{\theta};\\{\sigma\_i\\}\_{i=1}^L)=\frac{1}{L}\sum\_{i=1}^{L}\lambda(\sigma\_i)l(\mathbf{\theta};\sigma\_i)\tag{1.15}
\end{equation}
$$
式(1.15)中$\lambda(\sigma\_i)=\sigma\_i^2$，以便于每项$\lambda(\sigma\_i)l(\mathbf{\theta};\sigma\_i)$为同一个数量级。以此为目标函数训练神经网络，就可以得到数据分布梯度的估计函数。

根据分数估计函数，生成样本的Annealed Langevin dynamics算法伪代码如图1.3所示

<div align="center">
  <img src="./img/ald.png" height=300/>
</div>
<div align="center">
  图1.3 Annealed Langevin dynamics算法伪代码
</div>
文献[4]提出的基于分数的生成模型虽然能够生成高质量的样本。然而，根据文献[5]，可知，该方法不能扩展到高维特征，且在某些场景下不稳定。因此，需要对基于分数的生成模型改进，可改进的方面有

- 高斯噪音的方差$\\{\sigma\_{i}\\}\_{i=1}^L$的设定
- 分数网络$\mathbf{s}\_{\theta}(\mathbf{x},\sigma)$包含参数$\sigma$的方式
- 步长参数$\epsilon$的设定
- 采样步数$T$

### 正态噪音方差的选择

对于初始噪音方差$\sigma\_1$的选择，文献[4]提出应与训练数据中数据对之间欧氏距离一样大；噪音方差$\sigma\_{L}$为0.01，保持不变；对于其它的噪音方差的选择，应符合以$\gamma=\frac{\sigma\_{i-1}}{\sigma}$为比例的等比数列，且满足式(1.16)
$$
\begin{equation}
\Phi(\sqrt{2D}(\gamma-1)+3\gamma)-\Phi(\sqrt{2D}(\gamma-1)-3\gamma)\approx0.5\tag{1.16}
\end{equation}
$$
式(1.16)中$\Phi$为标准正态分布，$D$为数据维度。

### 分数网络包含噪音信息的方式

文献[4]中分数网络是以方差$\sigma$为条件的网络，那么对于无归一化的分数网络，其内存的需求与$L$呈现线性关系，这是不适用的。因此，$\mathbf{s}\_{\theta}(\mathbf{x},\sigma)=\mathbf{s}\_{\theta}(\mathbf{x})/\sigma$是一种简单高效的替代方式，$\mathbf{s}\_{\theta}(\mathbf{x})$为无条件分数网络。

### 退火朗之万动力学参数配置

**命题1**：若令$\gamma=\frac{\sigma\_{i-1}}{\sigma\_{i}}$,$\alpha=\epsilon\cdot\frac{\sigma\_i^2}{\sigma\_L^2}$，利用算法1，生成数据满足$\mathbf{x}\_{T}\sim\mathcal{N}(0,s\_t^2\mathbf{I})$。其中
$$
\begin{equation}
\frac{s_T^2}{\sigma\_i^2}=(1-\frac{\epsilon}{\sigma^2\_L})^{2T}(\gamma^2-\frac{2\epsilon}{\sigma\_L^2-\sigma\_L^2(1-\frac{\epsilon}{\sigma\_L^2})^2})+\frac{2\epsilon}{\sigma\_L^2-\sigma_L^2(1-\frac{\epsilon}{\sigma\_L^2})^2}\tag{1.17}
\end{equation}
$$
根据式(1.17)，可知，$\forall i\in(1,T]$，$\frac{s^2\_T}{\sigma\_i^2}$相等。

若期望$\frac{s^2\_T}{\sigma\_i^2}\approx1$，那么可先根据计算资源选择尽可能大的$T$，然后选择$\epsilon$使其尽可能$\frac{s^2\_T}{\sigma\_i^2}$尽可能接近1。

### 移动平均提升稳定性

虽然基于分数的生成模型，不需要对抗训练，但是仍会出现训练不稳定、颜色偏移的问题。然而，这个问题可通过指数移动平均解决。指数移动平均是指在训练分数网络时，参数更新方式为
$$
{\mathbf{\theta}}'\leftarrow m{\mathbf{\theta}}'+(1-m){\mathbf{\theta}\_i}\tag{1.18}
$$
式(1.18)中$\theta\_i$为第$i$次模型训练之后的模型参数，${\theta}'$为第$i-1$次训练之后的模型参数，$m$为动量参数。



## Score SDEs

DDPM和SPM扩散模型均遵循一个统一的范式：首先利用高斯噪音扰动数据使原始数据分布与标准正态分布一致；然后，从标准正态分布的数据抽样，利用朗之万MCMC抽样法，生成原始数据。与之相比，文献[6]Score SDE(Stochastic Differential Equation)模型利用随机微分方程扩展到无限噪音规模，其扩散过程被建模为$It\hat{o}$随机微分方程的解，可见式(1.19)。
$$
\begin{equation}
d\mathbf{x}=\mathbf{f}(\mathbf{x},t)dt + g(t)d\mathbf{w}\tag{1.19}
\end{equation}
$$
式(1.19)中，$\mathbf{w}$为标准维纳过程，也就是布朗运动；$\mathbf{f}(\cdot,t):\mathbb{R}^d\to\mathbb{R}^d$是一个值为向量的函数，被称为$\mathbf{x}(t)$的漂移系数；$g(\cdot):\mathbb{R}\to\mathbb{R}$是一个标量函数，被称为$\mathbf{x}(t)$的扩散系数。根据文献[12]，可知，只要飘逸系数和扩散系数满足Lipschitz连续，那么随机微分方程的强解一定存在。同时，$\mathbf{x}(t)$的概率密度函数用$p\_t(\mathbf{x})$表示；对于$0\le s\lt t\le T$，$p\_{st}(\mathbf{x}(t)\vert\mathbf{x}(s))$表示从$\mathbf{x}(s)$转换到$\mathbf{x}(t)$。

DDPM和SGM可被视为两个不同SDEs的离散化。其中，SGM扩散过程对应的随机微分方程为$d\mathbf{x}=\sqrt{\frac{d[\sigma^2(t)]}{dt}}d\mathbf{w}$；DDPM扩散过程对应的随机微分方程为$d\mathbf{x}=-\frac{1}{2}\beta(t)\mathbf{x}dt+\sqrt{\beta(t)}d\mathbf{w}$。对于SGM的微分方程，随着$t\to\infty$，其产生一个方差爆炸的过程，因此对应的随机微分方程被称为"VE-SDE"；对于DDPM的随机微分方程，随着$t\to\infty$，其方差为1的过程，因此对应的随机微分方程被称为"VP-SDE"。

根据文献[6]，可知，扩散过程的逆过程也是一个扩散过程，其反向时间随机微分方程可见式(1.20)
$$
\begin{equation}
d\mathbf{x}=[\mathbf{f}(\mathbf{x},t)-g(t)^2\nabla\_{\mathbf{x}}log{p\_t(\mathbf{x})}]dt+g(t)d\mathbf{\bar{w}}\tag{1.20}
\end{equation}
$$
式(1.20)中$\mathbf{\bar{w}}$为时间反向的标准维纳过程，$dt$为无穷小的负的时间步长。

只要式(1.20)中$\nabla\_{\mathbf{x}}log{p\_{t}(\mathbf{x})}$对于所有时间$t$已知，那么就可以根据式(1.20)推导逆扩散过程，从而生成样本。

如图1.4所示，Score SDEs扩散过程和逆扩散过程示意图。

<div align="center">
  <img src="./img/score-sde.png"/>
</div>
<div align="center">
  图1.4 Score SDEs扩散过程和逆扩散过程
</div>

图(1.4)中$\mathbf{x}(0)\sim p\_0$表示样本数据；$\mathbf{x}(T)\sim p\_T$表示先验分布，不包含$p_0$的任何信息，例如：高斯分布。

### 随机微分方程的分数估计

与DDPMs和SDMs一样，训练一个与时间依赖的分数模型$\mathbf{s\_{\theta}}(\mathbf{x},t)$估计分数$\nabla\_\mathbf{x}log{p\_t(\mathbf{x})}$，可见式(1.21)
$$
\begin{equation}
\theta^{\*}=\underset{\theta}{argmin}\mathbb{E}\_{t}\{\lambda(t)\mathbb{E}\_{\mathbf{x}(0)}\mathbb{E}\_{\mathbf{x}(t)\vert\mathbf{x}(0)}[\Vert\mathbf{s}\_{\theta}(\mathbf{x}(t),t)-\nabla\_{\mathbf{x}(t)}log{p\_{0t}(\mathbf{x}(t)\vert\mathbf{x}(0))}\Vert\_2^2]\}\tag{1.21}
\end{equation}
$$
式(1.21)中$\lambda:[0,T]\to\mathbb{R}\_{\gt0}$是一个正权重函数。与DDPMs和SDMs一样，$\lambda\propto\frac{1}{\mathbb{E}[\Vert\nabla\_{\mathbf{x}(t)}log{p\_{0t}(\mathbf{x}(t)\vert\mathbf{x}(0))}\Vert^2\_2]}$

若转换函数$p\_{0t}(\mathbf{x}(t)\vert\mathbf{x}(0))$已知，那么式(1.21)变高效可解。若函数$\mathbf{f}(\cdot,t)$为防射函数，以及转换函数为高斯分布，那么均值和方差可知，从而分数估计网络可训练。

### 样本生成

分数估计网络训练完成后，可利用数值SDE求解器求解反向时间随机微分方程。数值求解器提供了随机微分方程的近似求解器，例如：Euler-Maruyama和随机Runge-Kutta方法。为了提高采样质量，利用数值SDE求解器得到下一时刻$t$的样本估计$\mathbf{x}\_t$，再利用基于分数的MCMC方法纠正$\mathbf{x}\_t$，即数值SDE求解器扮演“预测器”的角色，MCMC方法扮演“纠正器”的角色，该方法被称为Predictor-Corrector采样。

#### 黑盒常微分方程求解器

对于所有的扩散过程，均有一个对应的确定过程，它的ODE(Ordinary Differential Equation)为
$$
\begin{equation}
d\mathbf{x}=[\mathbf{f}(\mathbf{x},t)-\frac{1}{2}g(t)^2\nabla\_{\mathbf{x}}log{p\_{t}(\mathbf{x})}]dt\tag{1.22}
\end{equation}
$$
只要分数$\nabla\_{\mathbf{x}}log{p\_{t}}(\mathbf{x})$已知，那么式(1.22)就可以被确定，该常微分方程被称为概率流ODE。那么，求解该常微分方程就可以进行采样。因为利用神经网络估计分数，所以该方法被称为黑盒常微分方程求解器。



### 条件扩散模型

利用随机微分方程把扩散模型扩展到连续结构，不仅允许从$p_0$采样生成数据样本，而且允许基于条件的生成$p\_0(\mathbf{x}(0)\vert\mathbf{y})$。给定扩散过程随机微分方程(1.19)，可从$p\_t(\mathbf{x}(t)\vert\mathbf{y})$采样，其逆扩散过程随机微分方程为
$$
\begin{equation}
d\mathbf{x}=\{\mathbf{f}(\mathbf{x},t)-g(t)^2[\nabla\_{\mathbf{x}}log{p_t(\mathbf{x})}+\nabla\_{\mathbf{x}}log{p\_{t}(\mathbf{y}\vert\mathbf{x})}]\}dt+g(t)d\mathbf{\bar{w}}\tag{1.23}
\end{equation}
$$
根据式(1.23)，可知，只要给定扩散过程梯度的估计$\nabla\_{\mathbf{x}}log{p\_t(\mathbf{y}\vert\mathbf{x}(t))}$，那么就可以求解逆问题的随机微分方程。可通过训练一个模型估计扩散过程$p\_t(\mathbf{y}\vert\mathbf{x}(t))$，也可以利用启发式方法和领域知识估计其梯度。



## 参考文献

[1] Yang L, Zhang Z, Song Y, et al. Diffusion models: A comprehensive survey of methods and applications[J]. ACM Computing Surveys, 2023, 56(4): 1-39.

[2] Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in neural information processing systems, 2020, 33: 6840-6851.

[3] Sohl-Dickstein J, Weiss E, Maheswaranathan N, et al. Deep unsupervised learning using nonequilibrium thermodynamics[C]//International conference on machine learning. PMLR, 2015: 2256-2265.

[4] Song Y, Ermon S. Generative modeling by estimating gradients of the data distribution[J]. Advances in neural information processing systems, 2019, 32.

[5] Song Y, Ermon S. Improved techniques for training score-based generative models[J]. Advances in neural information processing systems, 2020, 33: 12438-12448.

[6] Song Y, Sohl-Dickstein J, Kingma D P, et al. Score-based generative modeling through stochastic differential equations[J]. arXiv preprint arXiv:2011.13456, 2020.

[7] Song Y, Durkan C, Murray I, et al. Maximum likelihood training of score-based diffusion models[J]. Advances in Neural Information Processing Systems, 2021, 34: 1415-1428.

[8] Karras T, Aittala M, Aila T, et al. Elucidating the design space of diffusion-based generative models[J]. Advances in Neural Information Processing Systems, 2022, 35: 26565-26577.

[9] [Rao–Blackwell theorem - Wikipedia](https://en.wikipedia.org/wiki/Rao–Blackwell_theorem)

[10] [Evidence lower bound - Wikipedia](https://en.wikipedia.org/wiki/Evidence_lower_bound#Maximizing_the_ELBO)

[11] [Generative Modeling by Estimating Gradients of the Data Distribution | Yang Song (yang-song.net)](https://yang-song.net/blog/2021/score/)

[12] [Itô diffusion - Wikipedia](https://en.wikipedia.org/wiki/Itô_diffusion)

`