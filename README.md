# Bootstrap Deep Metric for Seed Expansion in Attributed Networks (BDM)

## 简介 (Introduction)

本项目是论文 **"Bootstrap Deep Metric for Seed Expansion in Attributed Networks" (BDM)** 的官方实现，该论文发表于 SIGIR 2024。BDM 是一个新颖的图深度度量学习框架，专为解决属性网络中的种子扩展问题而设计。该框架通过学习一种映射，将网络的拓扑结构和节点内容投影到一个表示空间中，使得社区成员在空间中彼此靠近，而非成员则相互远离。基于此表示空间，BDM 能够有效地从一小组种子节点出发，识别并扩展社区成员。

**论文链接:**
[Bootstrap Deep Metric for Seed Expansion in Attributed Networks](https://dl.acm.org/doi/10.1145/3626772.3657687)
*Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24)*


## 引用 (Citation)

如果您在研究中发现 BDM 有用，请考虑引用我们的论文：

```bibtex
@inproceedings{10.1145/3626772.3657687,
  author    = {Liang, Chunquan and Wang, Yifan and Chen, Qiankun and Feng, Xinyuan and Wang, Luyue and Li, Mei and Zhang, Hongming},
  title     = {Bootstrap Deep Metric for Seed Expansion in Attributed Networks},
  booktitle = {Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24)},
  year      = {2024},
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3626772.3657687},
  pages = {1629–1638},
  numpages = {10},
  location = {Washington DC, USA}, 
}
```

## BDM 框架概览 (Overview of BDM Framework)

BDM 框架的核心思想和主要组件如下图所示：

![BDM Framework Overview](https://github.com/user-attachments/assets/7c5783a5-d6b4-4751-9a14-0867d96f3a63)

*图注：BDM 框架概览。上层：使用深度度量进行种子扩展的核心思想。我们将网络的拓扑结构和节点内容映射到一个表示空间，使得社区成员彼此靠近，而非社区成员则相互远离。随后，我们选择靠近近似社区中心的节点作为扩展节点。下层：BDM 通过最小化产生的正样本锚点 $a^P$ 与 $H_{e\theta}$ 中正样本节点表示之间的度量损失，以及自锚点 $H_{e\phi}$ 与 $H_{e\theta}$ 中各节点表示之间的度量损失，来训练其主映射 $f_{\theta}$，其中 $\theta$ 和 $\phi$ 是可学习的参数。辅助映射 $f_{\phi}$ 的参数 $\phi$ 作为 $\theta$ 的慢速移动平均值进行更新。此处，sg 表示停止梯度（stop-gradient）。*

## 环境要求 (Requirements)

- `numpy==1.24.3`
- `scikit-learn==1.3.1`
- `torch==2.0.1`
- `torch_geometric==2.3.0`

## 使用方法 (Usage)

通过以下命令运行 BDM 演示：

```python
python run_bdm.py -d 'Cora' -c 0 -s 1 -u 'False'
```

参数说明：
- `-d`: 使用的数据集名称 (e.g., 'Cora', 'CiteSeer', 'PubMed')。
- `-c`: 用作正样本的标签索引。
- `-s`: 随机种子集的大小。
- `-u`: 是否使用高维节点特征作为种子节点 (True/False)。


