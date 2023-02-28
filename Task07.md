# Task07 标签传播（消息传递）与节点分类
【注】文章中部分表格、流程和公式来自：https://raw.githubusercontent.com/Relph1119/my-team-learning/master/docs/cs224w_learning46/task07.md

这既不是图嵌入方法，也不是表示学习的方法。
## 一、半监督节点分类
用已知的类别的节点预测未知类别的节点。
- 直推式学习：没有新节点，预测原图中未知标签的节点
- 归纳式学习：预测新加入节点的标签

标签传播方法属于**直推式学习**。

半监督节点分类方法的对比：
| 方法 | 图嵌入 | 表示学习 | 使用属性特征 | 使用标注 | 直推式 | 归纳式 
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 人工特征工程 | 是 | 否 | 否 | 否 | / | / |
| 基于随机游走的方法 | 是 | 是 | 否 | 否 | 是 | 否 |
| 基于矩阵分解的方法 | 是 | 是 | 否 | 否 | 是 | 否 |
| 标签传播 | 否 | 否 | 是/否 | 是 | 是 | 否 |
| 图神经网络 | 是 | 是 | 是 | 是 | 是 | 是 |

- 人工特征工程：节点重要度、集群系数、Graphlet等
- 基于随机游走的方法，构建自监督表示学习任务实现图嵌入，无法泛化到新节点，例如DeepWalk、Node2Vec、LINE、SDNE等
- 标签传播：假设“物以类聚、人以群分”，利用领域节点类别预测当前节点类别，无法泛化到新节点，例如Label Propagation、Interative Classification、Belief Propagation、Correct & Smooth等
- 图神经网络：利用深度学习和神经网络，构建领域节点信息聚合计算图，实现节点嵌入和类别预测，可泛化到新节点，例如GCN、GraphSAGE、GAT、GIN等
- Label Propagation：消息传递机制，利用领域节点类别信息

## 二、标签传播与集体分类
假设相邻节点具有相似类别。具有相似属性特性的节点，更可能相连且具有相同类别，由已知类别的节点猜出未知类别的节点。
### 1.方法分类
- Label Propagation（Relational Classification）
- Iterative Classification
- Correct & Smooth
- Belief Propagation
- Masked Label Prediction

### 2.Label Propagation
- 算法步骤：
    1. 初始化所有节点，对已知标签设置为 $Y_v=\{0,1\}$ ，未知标签设置为 $Y_v = 0.5$ 
    2. 开始迭代，计算该节点周围的所有节点P值的总和的平均值（加权平均）
    3. 
    $$
    P(Y_v = C) = \frac{1}{\displaystyle \sum_{(v, u) \in E} A_{v, u}} \sum_{(v, u) \in E} P(Y_u = c)
    $$

    4. 当节点P值都收敛之后，可以设定阈值，进行类别划分，例如大于0.5设置为类别1，小于0.5设置为类别0

- 举例：
  一个二分类问题，已知标签的节点设为0和1，未知标签的节点都设为0.5。则下一轮中未知标签的节点属于类别0的概率为与其相连的所有节点求和再求平均。然后一轮一轮继续，直至所有节点均收敛。如果不收敛，则达到设定轮次停止。如果是有权图，则做加权平均。

- 算法缺点：
    - 仅使用网络连接信息，没有使用节点属性信息
    - 模型不保证收敛

### 3.Iterative Classification
又称为ICA算法。
- 定义：
    - 节点 $v$ 的属性信息为 $f_v$ ，连接特征为 $z_v$
    - 分类器 $\phi_1(f_v)$ 仅使用节点属性特征 $f_v$  ，称为base classfier
    - 分类器 $\phi_2(f_v, z_v)$ 使用节点属性特征 $f_v$ 和网络连接特征 $z_v$ （即领域节点类别信息），称为relational classifier
    -  $z_v$ 为包含领域节点类别信息的向量

- 算法步骤：
    1. 使用已标注的数据训练两个分类器： $\phi_1(f_v)$ 、 $\phi_2(f_v, z_v)$ 
    2. 迭代直至收敛：用 $\phi_1$ 预测 $Y_v$ ，用 $Y_z$ 计算 $z_v$ ，然后再用 $\phi_2$ 预测所有节点类别，更新领域节点 $z_v$ ，用新的 $\phi_2$ 更新 $Y_v$ 

- 算法缺点：不保证收敛

- 该算法可抽象为马尔科夫假设，即 $P(Y_v) = P(Y_v | N_v)$ ，$N_v$ 为临域节点。

### 4. Correct & Smooth
又称为C&S算法，是一种对预测结果进行后处理的算法，也就是说其与前面什么预测方法无关，用图神经网络或者其他方法都可以。

- 算法步骤：
    1. 在已有类别标注的节点上训练基础分类器
    2. 用这个分类器去预测所有节点的分类结果，包含分类的类别概率（soft label），这里预测出的结果会与实际的label有偏差，这个偏差就是下一步Correct步骤中的偏差
    3. Correct步骤：计算training error，将不确定性进行传播和均摊（error在图中也有homophily）
       - 扩散training errors  $E^{(0)}$ ，这里只计算有标注的节点，没有标注的节点的error都设为0
       - 将邻接矩阵 $A$ 进行归一化，得到 $ \tilde{A} = D^{-\frac{1}{2}} A D^{\frac{1}{2}}$ ， $\tilde{A}$ 特性：
         -  $\tilde{A}$ 的特征值 $\lambda \in [-1, 1]$
         -  幂运算之后，依然保证收敛
         -  如果 $i$ 和 $j$ 节点相连，则 $\displaystyle \tilde{A}_{ij} = \frac{1}{\sqrt{d_i} \sqrt{d_j}}$ 
       - 迭代计算：$E^{(t+1)} \leftarrow (1 - \alpha) \cdot E^{(t)} + \alpha \cdot \tilde{A} E^{(t)}$ ，则对于已经标注的节点，其error是减小的，对于未标注的节点，其error是增大的。 $\alpha$ 是超参数，越大，表明节点更愿意相信传播过来的error，越小，表明其更愿意相信自己上一次的error
       - Correct后的最终预测结果 (记为 $Z(t)$ )为 ： $Z(t)=\text{2步骤中的预测结果} + s \times E^{(t)}$ ，其中 $s$ 是超参数
    4. Smooth步骤：对最终的预测结果进行Label Propagation
        - 通过图得到 $Z^{(0)}$
        - 迭代计算：$Z^{(t+1)} \leftarrow (1 - \alpha) \cdot Z^{(t)} + \alpha \cdot \tilde{A} Z^{(t)}$ ，其中 $Z^{(t)}$ 中已经标注的节点仍然使用非0即1的数值表示，而不是预测的概率值

- 总结：
    1. Correct & Smooth(C&S)方法用图的结构信息进行后处理
    2. Correct步骤对不确定性（error）进行扩散
    3. Smooth步骤对最终的预测结果进行扩散
    4. C&S是一个很好的半监督节点分类方法

### 5.Belief Propagation
- 类似消息传递，基于动态规划，即下一时刻的状态仅取决于上一时刻，当所有节点达到共识时，可得最终结果

- 算法思路：
    1. 定义一个节点序列
    2. 按照边的有向顺序排列
    3. 从节点 $i$ 到节点 $i+1$ 计数（类似报数）

- 定义：
    - Label-label potential matrix $\psi$：当邻居节点 $i$ 为类别 $Y_i$ 时，节点 $j$ 为类别 $Y_j$ 的概率（标量），反映了节点和其邻居节点之间的依赖关系
    - Prior belief $\phi$ ： $\phi(Y_i)$ 表示节点 $i$ 为类别 $Y_i$ 的概率
    -  $m_{i \rightarrow j}(Y_j)$ ：表示节点 $i$ 认为节点 $j$ 是类别 $Y_j$ 的概率
    - $\mathcal{L}$ ：表示节点的所有标签

- 算法步骤：
    1. 初始化所有节点信息都为1
    2. 迭代计算：


    $$
    m_{i \rightarrow j}(Y_j) = \sum_{Y_i \in \mathcal{L}} \psi(Y_i, Y_j) \phi_i(Y_i) \prod_{k \in N_j \backslash j} m_{k \rightarrow i} (Y_i), \ \forall Y_j \in \mathcal{L}
    $$


    3. 收敛之后，可得到结果：


    $$
    b_i(Y_i) = \phi_i(Y_i) \prod_{j \in N_i} m_{j \rightarrow i} (Y_i), \ \forall Y_j \in \mathcal{L}
    $$

- 优点：易于编程实现，可以应用于高阶 $\psi(Y_i, Y_j, Y_k, Y_v \dots)$ 
- 存在问题：如果图中有环，会不收敛

### 6.Masked Label Prediction

- 灵感来自于语言模型BERT

- 算法思路：
    1. 随机将节点的标签设置为0，用 $[X, \tilde{Y}]$ 预测已标记的节点标签
    2. 使用 $\tilde{Y}$ 预测未标注的节点
