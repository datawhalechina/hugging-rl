# 扩散模型基础

扩散模型是一类概率生成模型，它通过注入噪声逐步破坏数据，然后学习其逆过程，以生成样本。目前，扩散模型的研究主要有三种方式：去噪扩散概率模型[2,3] (Denoising Diffusion Probabilistic Models, 简称DDPMs)、基于分数的生成模型[4,5] (Score-Based Generative Models,简称SGMs)、随机微分方程[6,7,8] (Stochastic Differential Equations,简称Score SDEs)。

## DDPMs

扩散概率模型是一个被参数化的马尔科夫链，在有限时间内通过变分推断训练模型生成样本。它的学习范式可分为两个过程，一个是不断的向样本数据中加入高斯噪音直至破坏样本的**扩散过程**；另一个是逆扩散过程，也称**逆过程**，该过程需要学习马尔科夫链中状态之间的转换函数。





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

