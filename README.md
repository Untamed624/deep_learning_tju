# deep_learning_tju
# 基于全局接触图与局部图嵌入的 Micro-C 新型结构发现方案

## 1. 核心思想

本方案的核心思想是：**不将 Micro-C 接触矩阵简单切分成彼此独立的局部窗口，而是首先基于完整的 Micro-C 接触矩阵构建全局加权图，在全局图上利用图神经网络（GNN）学习染色质接触网络中的节点关系和拓扑结构；随后针对不同局部基因组区域，从全局图的节点表示中提取局部结构信息，并通过图级聚合生成具有全局上下文信息的局部 Graph Embedding。**

最终希望在局部 Graph Embedding 空间中比较不同接触结构之间的相似性，并识别与已有 OPCID、CHIN、CHID 等结构类型不同的潜在新型接触结构。

整体流程可以表示为：

```text
Micro-C Contact Matrix
        ↓
O/E Normalization
        ↓
Global Weighted Contact Graph
        ↓
GNN / Graph Representation Learning
        ↓
Global Node Embeddings
        ↓
Local Region / Local Structure Extraction
        ↓
Local Graph Pooling
        ↓
Context-aware Local Graph Embedding
        ↓
Similarity / Clustering / Novelty Detection
        ↓
Known Structure Recognition + Novel Structure Discovery
```

---

# 2. Micro-C 接触矩阵表示

Micro-C 数据首先表示为接触矩阵：

$$
C\in R^{N\times N}
$$

其中：

* \(N\)：genomic bins 的数量；
* \(C_{ij}\)：第 \(i\) 个 genomic bin 与第 \(j\) 个 genomic bin 的接触信号。

由于原始接触频率受到 genomic distance 的明显影响，因此在构图之前进行 Observed/Expected（O/E）归一化。

$$
O_{ij}=C_{ij}
$$

其中 \(O\) 表示实际观测到的接触强度。

对于 genomic distance：

$$
d=|i-j|
$$

可以根据所有具有相同 genomic distance 的 bin 对计算期望接触强度：

$$
E(d)=mean\{C_{ij}:|i-j|=d\}
$$

因此：

$$
OE_{ij}=\frac{O_{ij}}{E(|i-j|)}
$$

其中：

* \(O/E>1\)：该接触强度高于相同基因组距离下的期望水平；
* \(O/E<1\)：该接触强度低于期望水平；
* \(O/E\approx1\)：接近该距离下的平均接触水平。

O/E 矩阵可以减少 genomic distance 对接触频率的影响，使图中的边权更接近“异常或特异性空间接触”。

---

# 3. 全局加权图构建

与“先切局部窗口再分别构图”的方法不同，本方案首先使用完整的 Micro-C 接触矩阵构建一个全局接触图。

定义：

$$
G=(V,E,X,W)
$$

其中：

### 3.1 节点 \(V\)

每一个 genomic bin 对应一个图节点：

$$
V=\{v_1,v_2,\ldots,v_N\}
$$

例如：

```text
Bin 1 → Node 1
Bin 2 → Node 2
Bin 3 → Node 3
...
Bin N → Node N
```

节点可以附带节点特征 \(x_i\)，例如：

* genomic position；
* GC content；
* 局部接触统计特征；
* 节点度；
* 其他可获得的基因组注释信息。

在基础实验中，可以首先只使用简单节点特征，使模型主要依赖 Micro-C 接触结构本身。

---

### 3.2 边 \(E\)

如果两个 genomic bins 之间存在接触关系，则可以建立一条边：

$$
e_{ij}=(v_i,v_j)
$$

---

### 3.3 边权 \(W\)

边的权重由 O/E 接触强度表示：

$$
w_{ij}=OE_{ij}
$$

因此：

```text
Node i ───────── Node j
             w = O/E
```

边权不是简单的 0/1，而是保留连续的接触强度。

因此该图属于：

> **Weighted Contact Graph**

而不是简单的 binary contact graph。

---

# 4. 为什么不直接设置固定边阈值

传统方法可能采用：

$$
OE_{ij}>T
$$

当 \(OE_{ij}\) 大于某个阈值 \(T\) 时建立边，否则删除。

但是对于新型结构发现任务而言，固定阈值存在一定问题。

首先，很难确定一个具有普适性的阈值。不同区域、不同细胞条件以及不同测序深度可能导致接触强度分布不同。

其次，如果设置过高的阈值，一些弱但具有结构意义的接触可能被删除。

因此，本方案倾向于：

> **保留边权，而不是首先将接触关系二值化。**

在需要控制图规模时，可以采用 Top-k、kNN 或其他自适应稀疏化策略。

例如对于节点 \(i\)，保留其接触强度最高的 \(k\) 个邻居：

$$
N_k(i)=TopK_j(OE_{ij})
$$

但是仍然保留真实边权：

$$
w_{ij}=OE_{ij}
$$

这样：

> \(k\) 主要控制图的规模，而不是人为定义“什么接触才属于结构”。

---

# 5. 在全局图上进行 GNN 表示学习

构建全局接触图之后，不再针对每一个局部区域单独训练一个 GNN，而是在完整图上进行消息传递。

GNN 的基本目标是：

$$
G\rightarrow H
$$

其中：

$$
H=[h_1,h_2,\ldots,h_N]
$$

每一个：

$$
h_i\in R^d
$$

表示节点 \(v_i\) 在整个接触网络中的结构表示。

GNN 可以通过邻居聚合学习：

$$
h_i^{(l+1)}
=
UPDATE
\left(
h_i^{(l)},
AGGREGATE
\left(
\{h_j^{(l)},w_{ij}\}
\right)
\right)
$$

因此节点最终表示不仅包含自身信息，还包含其邻居以及多跳邻域的信息。

例如：

```text
        B
        │
        │
A ───── C ───── D
```

经过 GNN 后：

```text
C 的表示
    ↓
知道 A、B、D 的信息

A 的表示
    ↓
知道 C
    ↓
进一步间接获得 B、D 的结构信息
```

因此，全局 GNN 学习的是：

> **节点在整个染色质接触网络中的拓扑和连接上下文。**

---

# 6. 全局 Node Embedding 与局部 Graph Embedding 的区别

这里需要明确区分两个层次。

## 6.1 Node Embedding

GNN 首先产生每一个 genomic bin 的表示：

$$
v_i\rightarrow h_i
$$

因此：

$$
h_i
$$

属于 **Node Embedding**。

它描述的是：

> 某一个 genomic bin 在全局接触网络中的结构上下文。

---

## 6.2 Local Graph Embedding

研究目标并不是最终比较单个 genomic bin，而是比较一个局部接触结构。

因此需要定义一个局部区域：

$$
R=\{v_{i_1},v_{i_2},\ldots,v_{i_m}\}
$$

提取该区域对应的节点表示：

$$
H_R=
\{h_{i_1},h_{i_2},\ldots,h_{i_m}\}
$$

然后进行 pooling：

$$
z_R=POOL(H_R)
$$

最终：

$$
\boxed{z_R}
$$

就是该局部区域的 **Local Graph Embedding**。

---

# 7. Context-aware Local Graph Embedding

由于局部 embedding 并不是从一个孤立的局部图直接计算得到的，而是：

$$
Global\ Graph
\rightarrow
GNN
\rightarrow
Global\ Node\ Embeddings
\rightarrow
Local\ Pooling
$$

因此得到的局部 embedding 不仅包含局部区域本身的结构，还包含该区域与全局接触网络之间的关系。

因此可以将其称为：

> **Context-aware Local Graph Embedding**

即：

> **具有全局上下文信息的局部图嵌入。**

它描述的不仅是：

> “这个局部区域长什么样？”

还描述：

> “这个局部区域在整个染色质空间接触网络中处于什么样的结构环境？”

---

# 8. 为什么全局图能够帮助局部结构识别

假设两个局部区域在局部范围内具有相似的拓扑：

```text
Region A:

1 ── 2
│    │
4 ── 3
```

```text
Region B:

1 ── 2
│    │
4 ── 3
```

如果分别独立构图，两者可能获得非常接近的 embedding。

但是在完整 Micro-C 图中：

```text
Region A
    │
    ├─────── 远距离区域 X
    │
    └─────── 远距离区域 Y
```

而：

```text
Region B
    │
    └─────── 很少存在远距离连接
```

虽然二者局部拓扑类似，但全局空间环境不同。

全局 GNN 可以通过消息传递将这些上下文信息传播到局部节点表示中，从而使：

$$
H_A\neq H_B
$$

最终：

$$
z_A\neq z_B
$$

因此可以识别：

> **局部形态相似但全局空间环境不同的接触结构。**

---

# 9. 已知结构的 Embedding 表示

对于已有的结构类型，例如：

* CHIN
* OPCID
* CHID

不应该简单地认为每一种结构对应一个固定 embedding。

实际上，每一个具体结构实例都会得到一个 embedding：

$$
CHIN_1\rightarrow z_1
$$

$$
CHIN_2\rightarrow z_2
$$

$$
CHIN_3\rightarrow z_3
$$

因此同一种结构类型最终会形成一个 embedding distribution：

```text
            CHIN

          ● ● ●
        ● ● ● ●
         ● ●
```

OPCID：

```text
           OPCID

          ● ● ●
         ● ● ●
```

CHID：

```text
            CHID

           ● ●
          ● ● ●
```

因此：

$$
P(z|CHIN)
$$

$$
P(z|OPCID)
$$

$$
P(z|CHID)
$$

可以分别描述不同已知结构的 embedding 分布。

---

# 10. 基于 Embedding Similarity 的结构识别

对于一个新的局部区域：

$$
R_x
$$

经过 GNN 和 pooling 后得到：

$$
z_x
$$

可以计算它与已有结构 embedding 的相似性。

例如使用 cosine similarity：

$$
sim(z_x,z_i)
=
\frac{z_x\cdot z_i}
{\|z_x\|\|z_i\|}
$$

也可以使用：

* Euclidean distance；
* Cosine distance；
* Mahalanobis distance；
* k-nearest neighbors；
* Cluster distance；
* Prototype distance。

例如：

```text
Candidate X

与 CHIN     similarity = 0.91
与 OPCID    similarity = 0.42
与 CHID     similarity = 0.38
```

这可以用于判断：

> Candidate X 在结构表示空间中更接近哪一种已知结构。

但是不应仅仅因为它与 CHIN 最接近，就直接认为：

$$
Candidate\ X=CHIN
$$

因为它可能是：

> **CHIN-like but structurally distinct**

即：

> 与 CHIN 相似，但仍然具有区别于典型 CHIN 的结构特征。

---

# 11. 新结构发现

因此，本方案不仅进行已知结构识别，还进行 Novelty Detection。

对于一个新的局部 embedding：

$$
z_x
$$

分别计算：

$$
D(z_x,P_{CHIN})
$$

$$
D(z_x,P_{OPCID})
$$

$$
D(z_x,P_{CHID})
$$

如果：

$$
D(z_x,P_{CHIN})\gg 0
$$

$$
D(z_x,P_{OPCID})\gg 0
$$

$$
D(z_x,P_{CHID})\gg 0
$$

同时大量相似的局部区域都形成一个稳定 cluster，那么可以将其作为：

> **Potential Novel Contact Structure**

而不是简单归入已有结构。

---

# 12. 最终的任务定义

因此整个问题可以定义为：

> **基于全局 Micro-C 接触图的上下文感知局部图表示学习与新型染色质接触结构发现。**

其机器学习任务可以拆分为：

$$
\boxed{
Global\ Graph\ Representation\ Learning
}
$$

*

$$
\boxed{
Local\ Structure\ Representation
}
$$

*

$$
\boxed{
Similarity\ Analysis
}
$$

*

$$
\boxed{
Novelty\ Detection
}
$$

最终目标不是单纯训练一个：

```text
Graph → CHIN / OPCID / CHID
```

的分类器，而是建立一个：

```text
Micro-C Global Graph
        ↓
      GNN
        ↓
Global Node Embeddings
        ↓
Local Region Pooling
        ↓
Local Graph Embeddings
        ↓
Embedding Space
        ↓
┌───────────────┬────────────────┐
│               │                │
Known           Similar          Novel
Structures      Structures       Structures
│               │                │
CHIN            CHIN-like        Candidate X
OPCID           OPCID-like       Candidate Y
CHID            CHID-like        Candidate Z
```

的结构表示空间。

---

# 13. 整体研究框架

最终可以将整个实验概括为：

```text
                    Micro-C
                       │
                       ↓
              Contact Matrix
                       │
                       ↓
                O/E Normalization
                       │
                       ↓
          Global Weighted Contact Graph
                       │
              ┌────────┴────────┐
              │                 │
            Nodes             Edges
         genomic bins       O/E weights
              │                 │
              └────────┬────────┘
                       ↓
              Global GNN Encoder
                       ↓
              Node Embeddings
                       ↓
             Local Region Extraction
                       ↓
                Local Pooling
                       ↓
       Context-aware Local Graph Embedding
                       │
             ┌─────────┴─────────┐
             ↓                   ↓
       Known Structure       Novelty Detection
             ↓                   ↓
    CHIN / OPCID / CHID     Novel Candidates
             │                   │
             └─────────┬─────────┘
                       ↓
               Similarity Analysis
                       ↓
              Clustering / Validation
                       ↓
             Novel Structure Hypothesis
```

## 14. 核心创新点概括

本方案的核心区别可以概括为：

**第一，不对局部区域完全独立建模，而是首先建立全局 Micro-C 加权接触图。**

**第二，不将接触关系简单二值化，而是使用 O/E 保留连续接触强度作为边权。**

**第三，通过全局 GNN 获得包含多跳空间接触信息的节点表示。**

**第四，通过局部节点聚合生成具有全局上下文的 Local Graph Embedding。**

**第五，不仅识别 CHIN、OPCID、CHID 等已知结构，还在 embedding 空间中寻找无法被已有结构分布充分解释的新型结构。**

因此，最终研究目标可以概括为：

$$
\boxed{
\text{Global Contact Graph}
\rightarrow
\text{Context-aware Local Embedding}
\rightarrow
\text{Known Structure Recognition}
+
\text{Novel Structure Discovery}
}
$$

这一框架的关键假设是：

> **不同类型的染色质接触结构在全局接触网络中具有可学习的拓扑和边权模式，并能够在图表示空间中形成具有一定稳定性的结构分布；新的接触结构则可能表现为与已有结构分布具有明显差异、但自身能够形成稳定聚类的局部图模式。**
