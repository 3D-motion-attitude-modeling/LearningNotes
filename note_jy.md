# Day 1（基础语法学习）

## 一、Markdown基础语法

- 分段操作：`Enter`

- 换行但不分段：`Shift`+`Enter`

- 创建不同级别的标题：`#`

- 模块引用：`>`

  >Like this one

- 无序列表：`*` 

- 有序列表：`1.`

- 创建任务列表：`-[]`
  		- [ ] Like this one

- 引用代码块：```

- 键入公式：`$$`

- 创建表格：`|第一列|第二列|`

  | 第一列 | 第二列 |
  | ------ | ------ |
  |        |        |

- 分割线：`***`

- 生成目录：`[toc]`

- 嵌入Link：`[Explanation](Link)`

  [Like this](http://example.net/)

- 插入图片：`![Alt text][path "title"]`

  <img src="pictures\test.jpg" title="like this" style="zoom:50%;" />

- 斜体：`*text*`

- 加粗：`**text**`

- 嵌入式代码块：``

- 下标：`~`

​		H~2~

- 上标：`^`

   H^2^

- 高亮：`==text==`

   ==Like this==

   

## 二、Git bash基础语法

### 盘符命令

- 切换盘符：`cd xxx/xxx`
- 查看当前文件夹路径：`pwd`
- 查看当前文件夹内容：`ls`

### 文件命令

- 将远端内容同步到本地仓库：`git clone <SSH_key>`

- 文件暂存至待提交区：`git add 文件名.文件类型`
  - 提交所有变化：`git add -A`
  - 提交被修改与被删除的文件，不包括新文件： `git add -u`
  - 提交新文件和被修改的文件，不包括删除（常用）： `git add .`
- 将暂存区文件提交：`git commit -m "注释"`
- 查看提交文件与仓库文件的差异：`git diff 文件名.文件类型`

- 将文件同步到远端仓库：`git push`
- 将远端文件更新到工作区文件：`git pull`
- 将远端文件取回到本地仓库，但不更新工作区文件： `git fetch`
  - 查看取回文件与工作区文件的差异：`git log -p FETCH_HEAD`
- 将本地仓库内容更新到工作区：`git merge`

### 分支命令

- 查看本地所有分支：`git branch`

- 查看远端所有分支：`git branch -r`

- 新建分支：`git branch <name>`

- 删除本地分支：`git branch -d <name>`

- 删除后更新到远端：`git push origin(远程主机名，一般为origin):<name>`

- 重命名本地分支：`git branch -m <oldname> <newname>`

- 切换到某个分支：`git checkout <name>`

  - 切换后，可将dev分支合并到master分支，并不影响dev分支的开发：

    `git merge dev`

### 远端，本地仓库，暂存区，工作区的关系图

![](pictures\git.jpg)



# Day 2  $Logistic$ 回归与神经网络学习

## 一、$Logistic$ 回归

在**二元分类**问题中，采用Logistic回归的方式，得到预估函数。其中：

目标函数：$$y \in \{0,1\}$$

估计函数：$$\hat{y} = \sigma(w^Tx + b)$$ , 其中 $$g(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$ 为激活函数，值域为 $$[0, 1]$$ 。该估计函数本质为条件概率 $$P(y=1|x)$$。

损失函数：$$\mathcal{L}(\hat{y}, y) = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$$ ，本质为 $$-\log P(y|x)$$ 。

​					即最小化损失函数，即最大化条件概率 $$P(y|x) = \hat{y}^y(1-\hat{y})^{1-y}$$

成本函数：$$\mathcal{J}(w, b) = \frac{1}{m}\sum_{i = 1}^{m}\mathcal{J}(\hat{y}^{(i)}, y^{(i)})$$ , 本质为对所有样本采用极大似然估计的方法，求取参数 $$w, b$$ 。 

## 二、$Logistic$ 回归中的梯度求取

以下为针对某一训练样本中，某一参数求取梯度。

​		$$\frac{\partial{\mathcal{L}}}{\partial{a}} = -\frac{y}{a}+\frac{1-y}{1-a}$$ , 其中 $$a = \hat{y}$$

​		$\frac{\partial{\mathcal{L}}}{\partial{z}} = a-y$

​		$\frac{\partial{\mathcal{L}}}{\partial{w_{1}}} = x_{1}(a-y)$ , $\frac{\partial{\mathcal{L}}}{\partial{b}} = a-y$

​		$\frac{\partial{\mathcal{J}}}{\partial{w_{1}}} = \frac{1}{m}\sum_{i = 1}^{m}\frac{\partial{\mathcal{L}}}{\partial{w_{1}}}$

## 三、向量化

​		通过引入python中的 `numpy` 库，将参数与样本均进行向量化。使用并行运算，避免了在算法中显式的使用 `for` 循环，明显提高算法运行速度。以下简单介绍 $Logistic$ 回归的向量实现以及 `numpy` 库的常用指令。

### $Logistic$ 向量化	

```python
import numpy as np

Z = np.dot(W.T, X) + b
A = sigma(Z)
dZ = A - Y
dW = 1 / m * np.dot(X, dZ.T)
db = 1 / m * np.sum(dZ)
W = W - alpha * dW
b = b - alpha * db
```

​		其中 Z 为 $n*m$ 维矩阵，$n$ 取决于目标输出个数，在二元分类问题中值为 1 ，$m$ 取决于训练样本个数，A、Y、dZ与Z同维；W为$n_x * 1$ 维列向量；X为 $n_x * m$ 维矩阵。

### numpy库常用指令

- 满足高斯分布的随机变量矩阵：`w = np.random.randn(n, m)`
- 零矩阵：`w = np.zeros((n,1))`
- 重塑矩阵维度：`w.reshape(n, m)`
- 矩阵转置：`w.T`
- 矩阵求列/行和后压缩维度：`cal/row = np.sum(w, axis = 0/1)`
- 矩阵中元素进行对应运算：`np.exp/log/abs/maximum(w)`
- 同维度矩阵对应位置元素相乘：`a * b`

## 四、简单神经网络（单隐层）

​		以下符号与算法基于下图中右侧单隐层神经网络，左侧为单个神经元的运算（图中为 $Logistic$ 回归）。

![](pictures\单隐层神经网络.png)

		### 符号说明与正向传播

$a^{[i]}$ 代表第 $i$ 层神经网络神经元的值，在本例中第 0 层为输入层，第 1 层为隐藏层， 第 2 层位输出层。

$Z^{[i]}, W^{[i]}, b^{[i]}$ 分别代表第 $i$ 层神经网络的中间值与对应参数。

$X$ 为网络输入，$X = a^{[0]}$

$Y$ 为网络输出， $Y = a^{[2]}$

则神经网络正向传播过程为：
$$
\begin{cases}
Z^{[1]} = W^{[1]}A^{[0]} + b^{[1]} \\
A^{[1]} = g^{[1]}(Z^{[1]}) \\
Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]} \\
A^{[2]} = g^{[2]}(Z^{[2]}) \\
\end{cases}
$$
其中 $W^{[1]}$ 为 $n^{[1]} * n^{[0]}$ 维矩阵， $b^{[1]}$ 为 $n^{[1]} * 1$ 维矩阵 ， $g^{[i]}(z)$ 表示第 $i$ 层的激活函数。

### 常用激活函数

- $g(z) = \sigma(z) = \frac{1}{1+e^{-z}}$ , 用于二元分类问题中输出神经元的激活函数。

- $g(z) = \tanh(z) = \frac{e^{z} + e^{-z}}{e^{z} - e^{-z}}$ , 在梯度下降过程中性能优于 $\sigma$ 函数，但依旧会有在 $z$ 值较大时，梯度趋于零的问题。

- ReLU，线性修正单元函数（常用）：
  $$
  \begin{equation}
  g(z) = 
  \begin{cases}
  0 & z < 0 \\
  z & z \ge 0
  \end{cases}
  \end{equation}
  $$

- Leaky ReLU：
  $$
  \begin{equation}
  g(z) = 
  \begin{cases}
  0.01z & z < 0 \\
  z & z \ge 0
  \end{cases}
  \end{equation}
  $$

### 反向传播

以下为向量化后的反向传播计算公式，采用代码的形式记录。

```Python
dZ_2 = A_2 - Y													# n_2 * m
dW_2 = 1 / m * np.dot(dZ_2, A_1.T)								# n_2 * n_1
db_2 = 1 / m * np.sum(dZ_2, axis = 1, keepdims = Ture)			# n_2 * 1
dZ_1 = np.dot(dW_2.T, dZ_2) * (g_1对z_1的导数，取决于隐层激活函数)   # n_1 * m
dW_1 = 1 / m * np.dot(dZ_1, X.T)								# n_1 * n_0
db_1 = 1 / m * np.sum(dZ_1, axis = 1, keepdims = Ture)			# n_1 * 1	
```

