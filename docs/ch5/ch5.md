# 生成式对抗模仿学习

行为克隆算法直接模仿专家的策略，由于专家策略是不可知的，那么智能体对未访问过状态采取均匀策略，从而导致复合误差的产生；逆强化学习算法有两个步骤，分别是学习出专家策略优于其它策略的奖励函数、利用奖励函数基于强化学习获得智能体策略，这两步不断交替迭代，最终智能体策略收敛到专家策略，这一结果在文献[1]中得到证明；



## 参考文献

[1] Ho J, Ermon S. Generative adversarial imitation learning[J]. Advances in neural information processing systems, 2016, 29.
