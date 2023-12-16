# one-shot模仿学习

对于机器人学习新任务，更多的是希望它能够根据少量的演示就能完成任务。然而，模仿学习往往需要大量的数据和精细的特征工程。文献[1]中结合元学习与模仿学习形成了one-shot模仿学习，该算法把同一任务的一种演示和另一种不同初始状态演示的状态作为输入，预测该状态下动作的方式训练神经网络，从而使模型只需根据新任务的一段演示就能完成任务的通用能力。

## 问题设定

*one-shot模仿学习问题由$\langle \mathbb{T},\mathbb{D},\pi,\mathcal{R} \rangle$构成，其中*

- *$\mathbb{T}$是任何的分布，单个任务由$t$表示。*
- *$\mathbb{D}$是演示的分布，$\mathbb{D}(t)$指的是任务$t$演示的分布。*
- *$d\sim\mathbb{D}(t)$是一个演示的观测和动作序列。*
- *$\pi_{\theta}(a\vert o,d)$表示的是策略。*
- *$\mathcal{R}_t(d)$是任务标量值的评估函数。*

*该学习问题的目标是对于任务$t\in\mathbb{T}$和演示$d\in\mathbb{D}(t)$最大化策略的期望表现。*

虽然one-shot模仿学习的设定希望模型拥有跨任务的通用能力，但论文中Particle Reaching和Block Stacking两个任务族的网络结构是不同的。总的来说，由于演示序列的存在，该网络是一个序列网络结构。

## 架构

### Particle Reaching任务的网络架构

- **Plain LSTM**：隐藏层为512单元的LSTM编码演示轨迹，其输出与当前状态concat到一起，再输入全连接神经网络，最终输出动作。
- **LSTM with attention**：LSTM模块根据演示序列输出不同路标的权重向量。然后，根据当前状态得到路标位置的权重组合。最后，concat智能体位置与路标位置权重组合，输入到全连接神经网络生成动作。
- **Final state with attention**：与上面两个不同的是该网络架构只利用演示的最后状态作为输入，得到路标的权重向量。接下来的步骤，与以上两个网络一致。

以上三个网络越来越适用于特定场景，表明了表达力和泛化性之间存在权衡。



### Block Stacking任务的网络架构





## 参考文献

[1] Duan Y, Andrychowicz M, Stadie B, et al. One-shot imitation learning[J]. Advances in neural information processing systems, 2017, 30.
