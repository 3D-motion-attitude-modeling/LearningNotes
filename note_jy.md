# Part 1（基础语法学习）

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
- 将文件从暂存区删除：`git checkout 文件名.文件类型`
- 将暂存区文件提交：`git commit -m "注释"`
- 查看提交记录：`git log`
- 查看提交文件与仓库文件的差异：`git diff 文件名.文件类型`

- 将文件同步到远端仓库：`git push`
- 将远端文件更新到工作区文件：`git pull`
- 将远端文件取回到本地仓库，但不更新工作区文件： `git fetch`
  - 查看取回文件与工作区文件的差异：`git log -p FETCH_HEAD`
- 将本地仓库内容更新到工作区：`git merge`

### 分支命令

- 查看本地所有分支：`git branch`

- 查看远端所有分支：`git branch -r`

- 查看本地分支与远程分支的关系：`git branch -vv`

- 关联指定本地分支到指定远程分支： `git push --set-upstream origin <本地分支名>:<远程分支名>` （要求远程分支已经存在）

- 新建分支：`git branch <name>`

- 删除本地分支：`git branch -d <name>`

- 删除后更新到远端：`git push origin(远程主机名，一般为origin):<name>`

- 重命名本地分支：`git branch -m <oldname> <newname>`

- 切换到某个分支：`git checkout <name>`

  - 切换后，可将dev分支合并到master分支（我现在在master分支），并不影响dev分支的开发：

    `git merge dev`

### 远端，本地仓库，暂存区，工作区的关系图

![](pictures\git.jpg)

### 网站资源

[Git联合开发](https://cloud.tencent.com/developer/article/1936062)

[Git版本回退](https://blog.csdn.net/qq_39505245/article/details/119788832)

# Part 2  $Logistic$ 回归与神经网络学习

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

则神经网络**正向传播**过程为：
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

# Part 3 多层神经网络架构

## 一，前向传播公式

​		符号定义与前述相同，大写即代表向量化的表示。
$$
\begin{cases}
Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}\\
A^{[l]} = g^{[l]}(Z^{[l]})
\end{cases}
$$

## 二，反向传播公式

​		$L$ 代表神经网络层数，其中输入层为**第0层**。公式如下（使用python语言描述部分矩阵运算）：
$$
\begin{cases}
\mathrm{d}Z^{[l]} = \mathrm{d}A^{[l]} * g^{[l]^{'}}(Z^{[l]}) \\
\mathrm{d}W^{[l]} = \mathrm{d}Z^{[l]}A^{[l-1]^{T}}/m         \\
\mathrm{d}b^{[l]} = \mathrm{np.sum}(\mathrm{d}Z^{[l]}, \mathrm{axis = 1}, \mathrm{keepdims = ture})/m \\
\mathrm{d}A^{[l-1]} = W^{[l]^{T}}\mathrm{d}Z^{[l]}
\end{cases}
$$
​		其中 $\mathrm{d}A^{[L]}$ 由损失函数 $\mathcal{L}(\hat{y}, y)$ 给出。

## 三，神经网络架构图

​		以二分类问题的深度学习网络为例，如下图所示，前 $L-1$ 层激活函数为 $ReLU$ ，第 $L$ 层激活函数为 $Sigmoid$ 。

![](pictures\final outline.png)

# Part4 $Pandas$数据导入

## 一，$Pandas$ 基础

​       常用的 $Pandas$ 数据结构之一是 $DataFrame$ 。DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。

​		$DataFrame$ 的构造方法：`pandas.DataFrame(data, index, columns, dtype, copy)`

​		参数说明：

- `data` : 一组数据，可以是dict，lists等等
- `index` : 行标签，默认为 0，1，2 ···
- `columns` : 列标签，默认为 0，1，2 ···
- `dtype` : 数据类型，如float
- `copy` : 从输入处拷贝数据，默认为false		

​		使用 `loc` 指令返回指定行数据：

- `df.loc[0]` : 返回第一行
- `df.loc[[0,1]]` : 返回第一与第二行

​		使用 `read_csv` 读取CSV文件数据，`head(n)` 返回前n行数据，`tail(n)` 返回后n行数据， `info()` 返回基本信息。 `read_csv` 包含参数 `na_values` ，用来指定空数据的类型，例如 `na_values = ["na", "--"]` 。

## 二，$Pandas$ 数据清洗

1. **删除包含空字段的行**

​	`DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)`

​		参数说明：

- `axis`：默认为 **0**，表示逢空值剔除整行，如果设置参数 **axis＝1** 表示逢空值去掉整列。
- `how`：默认为 **'any'** 如果一行（或一列）里任何一个数据有出现 NA 就去掉整行，如果设置 **how='all'** 一行（或列）都是 NA 才去掉这整行。
- `thresh`：设置需要多少非空值的数据才可以保留下来的。
- `subset`：设置想要检查的列。如果是多个列，可以使用列名的 list 作为参数。
- `inplace`：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据。

​	2.**替换空行数据**

​	`df["ST_NUM"].fillna(df["ST_NUM"].mean(), inplace = True)`		

​		上述代码将 `ST_NUM` 列的空数据用该列的平均值替代，且会修改源数据。此外还可使用 `median()` ， `mode()`，`std()` 计算响应的统计参数。

​	3.**清楚重复数据**

​	`drop_duplicates()` 剔除重复数据。 

# Part5  机器学习基础

## 一、方差与偏差

1. 训练集的高误差——高偏差（欠拟合），可以通过增加网络规模、增加训练时间等方法改进。
2. 测试集的高误差——高方差（过拟合），可以通过增加数据量，对数据正则化等方法改进。

## 二、L2正则化

​			多层深度学习网络，通过在成本函数中加入正则化项来实现正则化，以下使用 $Frobenius$ $Norm$ 实现正则化。
$$
\mathcal{J}(W,b) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{J}(\hat{y}^{(i)},y^{(i)}) + \frac{\lambda}{2m}\sum_{l=1}^{L}||w^{[l]}||^{2}
$$
其中 $\frac{\lambda}{2m}\sum_{l=1}^{L}||w^{[l]}||^{2}$ 为 $Frobenius$ $Norm$ ，是每一项的平方和。

​		通过正则化，可以在每次迭代过程中减小 $w$ 的值，因为更新过程变为  $w^{[l]} = (1-\frac{\lambda}{m})*w^{[l]}-\alpha*\mathrm{d}w^{[l]}$ 。

​		而减小 $w$ 的值便意味着神经元的影响变小，从而使得网络在实际意义上更稀疏，一定程度上避免了过拟合的出现。

## 三、Dropout及其他正则化方法

​		*inverted dropout* 方法指的是在前向传播的过程中，对每一层神经网络的神经元随机失活。例如对某一层而言，设定一个 *keep-prop* （由专门的布尔矩阵负责，按该概率生成布尔型数据）假设为80%，20个中有4个神经元失活，该神经元对应的 *a* 值置为零，同时在求取下一层的 *z* 值时，为了保持期望不变，对 $W*A$ 的部分除以 *keep-prop* 。

​		完成正向传递后，因为该次传递中有部分 *a* 值置为了零，由公式:
$$
{d}W^{[l]} = \mathrm{d}Z^{[l]}A^{[l-1]^{T}}/m
$$
​        可知，对应的 *w* 的梯度将为0，即该次迭代不更新。

​		除此之外，还有通过数据集变形来短时间扩充数据集、提前停止网络迭代避免 *W* 参数过大等方法。

## 四、归一化数据

​		为了保证梯度下降算法中下降速率均匀，可以以较大步长下降，需要将数据集做归一化处理。

​		即将训练集与测试集数据的**均值化为0，方差化为1** 。测试集数据采用由训练集数据计算出的均值与方差进行归一化。满足如下公式：
$$
x^{\star} = \frac{x-\mu}{\sqrt{\sigma}}
$$

## 五、初始化数据

​		为了避免深度网络层数上涨带来的梯度爆炸或梯度消失问题，改变初始化 $W$ 参数时的方法。该问题产生是由于在层数增加带来 $\hat{y}$ 的值指数变化，故在初始化 $W$ 的时，减小其方差来避免该问题**过快出现**。

​		根据经验而定，在激活函数为 *ReLu* 时，将 $W$ 方差控制在 $\frac{2}{n^{l-1}}$ 。而激活函数为 *tanh* 时， 控制在 $\frac{1}{n^{l-1}}$ 。

## 六、梯度检验

​		使用梯度检验进行DEBUG。检验的本质是导数值的数值估计，采用双边公差的方式实现。估计 $f^{’}(\theta)$ 的公式如下：
$$
\frac{f(\theta+\epsilon)-f(\theta-\epsilon)}{2\epsilon}
$$
误差将控制在 $O(\epsilon^{2})$ 。将上述检验方法应用于深度网络的 *Backprop* 当中时，首先将 $W^{[i]}$， $b^{[i]}$ （$i$ 从$1$到$L$）,化为一个总的向量 $\theta$ ，将 $\mathrm{d}W^{[i]}$， $\mathrm{d}b^{[i]}$ （$i$ 从$1$到$L$）,化为一个总的向量 $\mathrm{d}\theta$ 。 $\theta$ 的每一项是一个参数 $w$ 或 $b$ 。之后对每一项计算：
$$
\mathrm{d}\theta_{approx}^{[i]} = \frac{\mathcal{J}(\theta_{1},\theta_{2},···,\theta_{i}+\epsilon,···)-\mathcal{J}（\theta_{1},\theta_{2},···,\theta_{i}-\epsilon,···)}{2\epsilon}
$$
检验值为：
$$
\frac{||\mathrm{d}\theta_{approx}-\mathrm{d}\theta||_{2}}{||\mathrm{d}\theta_{approx}||_{2}+||\mathrm{d}\theta||_{2}}
$$
若检验值在 $10^{-7}$ 数量级，则可认为当前 *Backprop* 没有差错。

​		注意事项：

- 不要在训练网络的过程中使用梯检验，只在DEBUG中使用。
- cost function $\mathcal{J}(\theta)$ 要进行正则化。
- 不可以和 *Dropout* 一并使用。
- 在初始化参数后的第一次迭代使用梯度检验，可能由于参数本身较小不出现显著差错；可以在几轮迭代过后，当参数值增大后，再次进行梯度检验。

# Part6 Mediapipe模型

​		这一部分介绍，如何使用开源Mediapipe模型实现人体姿态的检测，使用Python环境实现。

## 一、模型的输入参数

​		参考 [Python_API]([Pose - mediapipe (google.github.io)](https://google.github.io/mediapipe/solutions/pose#python-solution-api))，模型有如下输入参数：

- `STATIC_IMAGE_MODE`：如果设置为 false，该解决方案会将输入图像视为视频流。它将尝试在第一张图像中检测最突出的人（人员检测），并在成功检测后进一步定位姿态关键点坐标。在随后的图像中，它只是简单地跟踪那些关键点坐标，而调用另一个检测，直到它失去跟踪，可以减少计算和延迟。如果设置为 true，则对每次输入的图像都会进行一次人员检测，非常适合处理一批静态的、可能不相关的图像。默认为false。
- `MODEL_COMPLEXITY`：模型的复杂度：0、1 或 2。姿态关键点坐标的准确度和计算延迟通常随着模型复杂度的增加而增加。默认为 1。
- `SMOOTH_LANDMARKS`：如果设置为true，将会过滤不同的输入图像上的姿态关键点坐标以减少抖动，但如果static_image_mode也设置为true则忽略。默认为true。
- `UPPER_BODY_ONLY`：是要追踪33个姿态关键点坐标还是只有25个姿态关键点坐标。
- `ENABLE_SEGMENTATION`：如果设置为 true，除了姿态关键点坐标之外，还会生成人物分割蒙版。默认为false。
- `SMOOTH_SEGMENTATION`：如果设置为true，将会过滤不同的输入图像上的人物分割蒙版以减少抖动，但如果 enable_segmentation设置为false或者static_image_mode设置为true则忽略。默认为true。
- `MIN_DETECTION_CONFIDENCE`：来自人员检测模型的最小置信度 ([0.0, 1.0])，即将检测视为成功的概率。默认为 0.5。
- `MIN_TRACKING_CONFIDENCE`：来自姿态关键点跟踪模型的最小置信值 ([0.0, 1.0])，用于判断关键点坐标是否追踪成功。若未追踪成功，则将在下一个输入图像上自动调用人物检测。将其设置为更高的值可以提高稳健性，但会有更高的延迟。如果 static_image_mode 为 true，则忽略，人员检测在每个图像上运行。默认为 0.5。

## 二、参考代码

使用Holistic模型完成图片与视频检测的[代码](https://zhuanlan.zhihu.com/p/353861059)。

使用Pose模型完成视频检测的[代码](https://blog.csdn.net/weixin_43229348/article/details/120541448)。

# Part7 相机标定

## 一、坐标映射关系

### 坐标系构成

- **世界坐标系(world coordinate system)**：用户定义的三维世界的坐标系。单位为m。是客观三维世界的绝对坐标系，也称客观坐标系。世界坐标系作为基准坐标系来描述数码相机的位置，并且用它来描述安放在此三维环境中的其它任何物体的位置，用（Xw, Yw, Zw）表示其坐标值。
- **相机坐标系(camera coordinate system)（光心坐标系）**：单位为m。以相机的光心为坐标原点，X 轴和Y 轴分别平行于图像坐标系的 X 轴和Y 轴，相机的光轴为Z 轴，用（Xc, Yc, Zc）表示其坐标值。
- **图像坐标系(image coordinate system)**： 以CCD 图像平面的中心为坐标原点，X轴和Y 轴分别平行于图像平面的两条垂直边，用( x , y )表示其坐标值。图像坐标系是用物理单位（例如毫米）表示像素在图像中的位置。
- **像素坐标系(pixel coordinate system)：**以图像平面的左上角顶点为原点，X 轴和Y 轴分别平行于图像坐标系的 X 轴和Y 轴，用(u , v )表示其坐标值。每个元素叫像素，像素坐标系就是以像素为单位的图像坐标系。[原文链接](https://blog.csdn.net/qq_30815237/article/details/87530011)

![相机坐标系关系](.\pictures\相机坐标系关系.png)

### 内参矩阵

实现像素坐标系到相机坐标系的变换

![相机坐标系关系](.\pictures\内参.png)

### 外参矩阵

实现相机坐标系到世界坐标系的变换

![相机坐标系关系](.\pictures\外参.png)



### 单应性矩阵

使用标定板，得到的在标定板位置建立的世界坐标系与像素坐标系的变换关系矩阵。
