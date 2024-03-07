# BeT：一次克隆k个模式

行为克隆算法的前提假设是数据来自于解决特定任务单一模式的专家演示。然而，真实世界的预先收集的数据包含行为的多个模式，即使是同一个人对同样的行为也会展示多种模式。另一方面，Transformer模型容量足够大，且拥有建模多种token的能力。因此，[BeT](https://arxiv.org/abs/2206.11251)把Transofmer与Behavior Cloning相结合以能够预测多峰分布的动作。

同时，作者们在5个数据集中进行实验，结果表明：

- 在多模态数据集上，与之前的行为建模算法相比，BeT能够实现较高的性能。
- BeT能够覆盖数据集中的主要模式，而不是一个模式。

如图1所示，BeT的架构。

<div align="center">
  <img src="https://www.robotech.ink/usr/uploads/2024/02/2151712265.png" width=800/>
  <p>图1 BeT架构</p>
</div>


## 算法设计

BeT基于Transformer的Decorder架构进行序列到序列的建模。主要的创新点就是动作离散化，首先把数据集中动作利用K-means算法进行聚类，形成$k$个聚类中心。然后，每个动作与最近聚类中心的距离作为残差项，可见图1.A所示。在模型训练阶段，把最近$h$个观测序列作为输入预测$k$个聚类中心的概率分布，以及$k\times\vert A\vert$残差矩阵($\vert A\vert$为动作的维度)。其中，$k$个聚类中心的概率分布以[Focal Loss](https://arxiv.org/abs/1708.02002 "Focal Loss")作为损失函数(可见式(1)所示)，而残差项以[Masked Multi-Task Loss](https://arxiv.org/abs/1504.08083 "Masked Multi-Task Loss")为损失函数(可见式(2)所示)，可见图1.B。测试阶段，先基于聚类中心概率分布选择聚类中心，再基于聚类中心选择残差向量，聚类中心与残差向量求和为最终动作序列。

$$
\begin{aligned}
\mathcal{L}_{focal}(p_t)=-(1-p_t)^{\gamma}log(p_t)
\end{aligned}\tag{1}
$$

相较于Cross-Entropy，Focal Loss更关注容易分类错误的Hard样本，因此非常适合样本不平衡的问题，两者的比较可见图2所示。
<div align="center">
  <img src="https://www.robotech.ink/usr/uploads/2024/02/3978262805.png" width=500/>
  <p>图2 CE与Focal Loss的比较</p>
</div>

$$
\begin{aligned}
MT-Loss(\mathbf{a},(<\hat{a}_i^{(j)}>)_{j=1}^k)=\sum_{j=1}^k\mathbb{I}[\lfloor\mathbf{a}\rfloor =j]\cdot\Vert <\mathbf{a}>-<\hat{a}^{(j)}> \Vert_2^2
\end{aligned}\tag{2}
$$

式(2)中$\mathbb{I}$为指示函数，$\lfloor\mathbf{a}\rfloor$为动作$\mathbf{a}$对应的聚类中心，$<\mathbf{a}>$为动作的残差项。

<div align="center">
  <img src="https://www.robotech.ink/usr/uploads/2024/02/810268854.png" width=800/>
  <p>图3 厨房环境中模型效果</p>
</div>

如图3所示，BeT与其它算法在厨房环境中的比较，不同的颜色代表不同任务。由此可见，BeT算法能够执行更长期的任务，且能够维护执行任务的多样性。




## 相关思考

BeT一个很明显的缺点就是K-means算法中$k$的选择，不同$k$对模型的效果是一个值得研究的方面。在机器人操纵任务中，主要关注模型的任务的时序连续能力和长期任务的规划能力，只有这样的模型才能完成复杂的任务。根据BeT原理可知，大部分利用Transformer作为backbone的模型，本身的创新点很少，主要就是希望能够把其长序列建模能力、生成能力和高容量应用在各自的领域。