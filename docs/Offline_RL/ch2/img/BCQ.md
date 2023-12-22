

- 输入：数据集$\mathcal{B}$，模型训练步数$T$，目标网络更新频率$\tau$，min-batch size $N$，策略的最大扰动$\Phi$，数据集中动作种类的数量$n$，最小化权重$\lambda$
- 初始化：用随机参数$\theta_1,\theta_2,\phi,w$初始化Q网络$Q_{\theta_1},Q_{\theta_2}$，策略扰动网络$\xi_{\phi}$，动作的VAE网络$G_w=\{E_{w_{1}},D_{w_{2}}\}$，以及目标网络$Q_{{\theta}'_1},Q_{{\theta}'_2},\xi_{{\phi}'}$
- 每个步数$t$
  - 数据集$\mathcal{B}$中采样大小为$N$的$(s,a,r,{s}')$转换。
  - 计算：$\mu,\sigma=E_{{w}_1}(s,a)$，$\tilde{a}=D_{w_{2}}(s,z)$，$z\sim \mathcal{N}(\mu,\sigma)$
  - 计算：$w\leftarrow argmin_{w}\sum(a-\tilde{a})^2+D_{KL}(\mathcal{N}(\mu,\sigma)\Vert\mathcal{N}(0,1))$
  - 采样$n$个动作：$\{a_i\sim G_w({s}')\}_{i=1}^n$
  - 扰动动作：$\{a_i=a_i+\xi_{\phi}({s}',a_i,\Phi)\}_{i=1}^n$
  - 计算：$y=r+\gamma\underset{a_i}{max}[\lambda\underset{j=1,2}{min}Q_{{\theta}'_j}({s}',a_i)+(1-\lambda)\underset{j=1,2}{max}Q_{{\theta}'_j}({s}',a_i)]$
  - $\theta\leftarrow argmin_{\theta}\sum(y-Q_{\theta}(s,a))^2$
  - $\phi\leftarrow argmax_{\phi}\sum Q_{\theta_1}(s,a+\xi_{\phi}(s,a,\Phi))$,$a\sim G_w(s)$
  - 更新目标网络：${\theta}'_i\leftarrow \tau\theta+(1-\tau){\theta}'_i$,${\phi}'\leftarrow\tau\phi+(1-\tau){\phi}'$

