# 模仿学习简介

在面对复杂的不确定性环境时，传统的控制技术很难应对，往往需要编写大量的代码。同时，强化学习需要与环境交互不断试错学习，对于试错成本较高的领域，很难适用。然而，若是能够积累专家数据，智能体根据这部分数据学习出专家策略，那么解决环境不确定性和高昂学习成本的问题，这就是模仿学习产生的背景。文献[1]，给出了模型学习的定义。

**定义一**：*智能体利用专家演示数据学习出策略，用于解决给定任务的过程，被称为模仿学习。*

根据文献[3]，可知，模仿学习的学习范式可被分为三类，分别是：行为克隆、逆强化学习、对抗模仿学习、以及Imitation from Observation。以上学习范式，后续章节会进行介绍。

## 马尔科夫决策过程

### 有限长度回合制马尔科夫决策过程



### 无限长度折扣马尔科夫决策过程





## 参考文献

[1] Hussein A, Gaber M M, Elyan E, et al. Imitation learning: A survey of learning methods[J]. ACM Computing Surveys (CSUR), 2017, 50(2): 1-35.

[2] Osa T, Pajarinen J, Neumann G, et al. An algorithmic perspective on imitation learning[J]. Foundations and Trends® in Robotics, 2018, 7(1-2): 1-179.

[3]Zare M, Kebria P M, Khosravi A, et al. A survey of imitation learning: Algorithms, recent developments, and challenges[J]. arXiv preprint arXiv:2309.02473, 2023.