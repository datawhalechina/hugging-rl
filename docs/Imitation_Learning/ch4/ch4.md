# 对抗模仿学习

逆强化学习算法由于计算复杂度高的问题，将其扩到到复杂环境具有很大的挑战。对抗模型学习提供了一个不同的学习范式，该算法由专家策略生成器和策略判别器构成，专家策略生成器用于模仿专家策略，策略判别器用于区分专家策略和非专家策略。该两个智能体不断博弈，从而专家策略生成器生成的策略越接近专家策略。

## 算法发展历程

对抗模仿学习能够应对逆强化学习局限的能够，吸引了大量的注意力。尤其，GAIL[2]算法提出之后，产生了不少的研究。在GAIL算法中，生成器通过最小化专家侧露与生成策略之间的距离模仿专家策略；判别器通过最大化专家策略的奖励来区分专家策略与非专家策略。同时，奖励信号也是告诉该策略被模仿的难以程度。在度量专家策略与非专家策略之间距离的方式上，GAIL[2]利用Shannon-Jensen divergence作为度量两个分布之间距离；文献[3]中AIRL算法利用KL-Divergence度量两个分布之间距离。根据文献[4]中生成对抗神经网络的研究中，发现，Wassertein distance作为度量方法时，模型的训练更稳定。因此，文献[5],[6]把Wassertein distance度量方法用到了对抗模仿学习中。

生成对抗的建模方式，是一种求解min-max优化问题的方式。然而，在训练时，这种方式往往会遇到梯度消失和拟合失败的问题。文献[7]中提出了PWIL算法，通过primal-dual的方式近似Wassertein distance，从而缓解梯度消失的问题。



## 参考文献

[1] Zare M, Kebria P M, Khosravi A, et al. A survey of imitation learning: Algorithms, recent developments, and challenges[J]. arXiv preprint arXiv:2309.02473, 2023.

[2] Ho J, Ermon S. Generative adversarial imitation learning[J]. Advances in neural information processing systems, 2016, 29.

[3] Fu J, Luo K, Levine S. Learning robust rewards with adversarial inverse reinforcement learning[J]. arXiv preprint arXiv:1710.11248, 2017.

[4] Arjovsky M, Chintala S, Bottou L. Wasserstein generative adversarial networks[C]//International conference on machine learning. PMLR, 2017: 214-223.

[5] Kostrikov I, Agrawal K K, Dwibedi D, et al. Discriminator-actor-critic: Addressing sample inefficiency and reward bias in adversarial imitation learning[J]. arXiv preprint arXiv:1809.02925, 2018.

[6] Li Y, Song J, Ermon S. Infogail: Interpretable imitation learning from visual demonstrations[J]. Advances in neural information processing systems, 2017, 30.

[7] Dadashi R, Hussenot L, Geist M, et al. Primal wasserstein imitation learning[J]. arXiv preprint arXiv:2006.04678, 2020.
