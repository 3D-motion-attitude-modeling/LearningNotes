# 深度学习笔记

* Convolutional Neural Networks 卷积神经网络

## 一.神经网络与深度学习

* 神经元 `input--(neuron)-->output`

* ReLU: 修正线性单元，线性、非负

* Supervised Learning：监督学习，机器学习的主要方法
* Standard NN: 标准神经网络
  Convolutional NN: 卷积神经网络（通常用于图像数据）
  Recurrent NN: 循环神经网络（适合处理一维序列数据）

![](D:\2021-2022-2\深度学习资料\神经网络.png)

* Structured Data: 结构化数据，如数字等
  Unstructured Data: 非结构化数据，如音频，图片等

### 课程中使用的符号

* 训练集的规模 `m`



## 二.神经网络编程基础 Basics of Neural Network Programming

### 2.1 二分分类

* 用`(x,y)`表示一个样本，其中x为n维特征列向量（作为输入），标签y取0或1（作为输出），由众多样本组成训练集。
  所有的m个x可以组成一个大的n*m矩阵，用X表示。Python检查矩阵规模：`X.shape`
  所有的m个y也可以组成一个1\*m矩阵Y。

### 2.2 logistic回归 logistic regression

* sigmoid函数 σ(z)=1/(1+e<sup>-z</sup>)

* 对于给定的x，我们想知道y=0或是1.因此，令 `y'=P(y=1|x)`。
* logistic回归即为估计y‘的方法之一：`y'=σ(wx+b)`，参数w和b通过训练得到。

### 2.3 损失函数 loss function

* `L(y',y)` 衡量预测值与实际值的差距，越小说明预测越准。
  如：`L(y',y)=(y'-y)^2/2`或`L(y',y)=-(y*ln(y')+(1-y)*ln(1-y'))`

* 成本函数 cost function `J(w,b)=ΣL(y',y)/m`，我们的目标是找到使得成本函数最小的w和b。

### 2.4 梯度下降法

![](D:\2021-2022-2\深度学习资料\梯度下降.jpg)

```
repeat {
	w := w-α*𝜕w;
	b := b-α*𝜕b
}
```

* α:learning rate；𝜕w指代𝜕J(w,b)/dw，表示对w求偏导。

### 2.9 logistic回归中的梯度下降法

* 之前，我们有`a=y'=σ(z)=σ(wx+b)`和`L(a,y)=-(y*ln(a)+(1-y)*ln(1-a))`，其中w和x为行/列向量。

![](D:\2021-2022-2\深度学习资料\logistic 回归中的梯度下降.jpg)

* 按照上述方法计算各项导数，并且更新w和b。

### 2.10 m个样本的梯度下降

* 之前，我们用成本函数`J(w,b)=ΣL(a,y)/m`表示m个样本a<sup>(i)</sup>的平均损失。

因此，我们有 𝜕J(w,b)/𝜕w<sub>1</sub>=Σ𝜕L(a,y)/m𝜕w<sub>1</sub>。

![向量化logistic](D:\2021-2022-2\深度学习资料\向量化logistic.jpg)

```
J=0;dw=0;db=0;
for i=1 to m do
	z(i) = wx(i)+b;
	a(i) = σ(z(i));
	J += -(y(i)*ln(a(i))+(1-y(i))*ln(1-a(i)));
	dz(i) = a(i)-y(i);
	dw += x(i)dz(i); //w为1*n的向量，其中各个元素分别计算
	db += dz(i);
J /= m;dw /= m;db /= m;
w = w-αdw;b = b-αdb;
```

* 上述为一次梯度下降法，我们需要多次使用以不断逼近最小损失。

### 2.11 向量化

* 由于w为1*n的，需要一个for循环来计算w。向量化可以优化这一部分运算。向量化使用了计算机内置的SIMD指令，如python中的numpy，np.dot()。

### 2.13 向量化logistic回归

![向量化logistic](D:\2021-2022-2\深度学习资料\向量化logistic.jpg)

* 使用`Z=np.dot(w,x)+b`即可高效计算所有z(i),然后`A=σ(Z)`计算A。

![](D:\2021-2022-2\深度学习资料\向量化logistic2.jpg)

* `dZ=A-Y,dW=XdZT/m,db=np.sum(dZ)/m`

* 使用上述代码优化2.10中的梯度下降代码。

```
Z=np.dot(w,x)+b;
A=σ(Z);
dZ=A-Y;
dW=XdZT/m;
db=np.sum(dZ)/m;
W:=W-αdW;
b:=b-αdb;
```

### 2.16 关于Python_numpy向量的说明

* `a = np.random.randn(5)`将会创建一个1*5的行向量，但实际上它是用“秩为1的数组”表示的，对其进行矩阵操作时可能出现bug。
* 因此，推荐使用`a = np.random.randn(5,1)`创建列向量，`a.T`即为1*5的行向量。或直接`a = np.random.randn(1,5)`



## 三.神经网络

### 3.2 神经网络表示

![](D:\2021-2022-2\深度学习资料\表示.jpg)

* Input layer: 输入层a[0]
* Hidden layer: 隐藏层a[1]，中间运算时产生的激活值“看不见”。
* Output layer: 输出层a[2]

这是一个双层神经网络：隐藏层一层，输出层一层，输入层不计。在神经网络中，上一层的值作为下一层的输入。



### 3.3 计算神经网络的输出

* 以图3.2.1为例，其中的每一个圆节点的值 a[i]~j~ （表示第i层的第j个节点）都使用公式 `a = σ(wx+b)`计算。
* 注意w和x是一维向量，其大小由输入的数量决定，如计算第一层时大小为3，计算第二层时大小为4。
* 可使用向量化优化计算：

![](D:\2021-2022-2\深度学习资料\3.3.1 多层神经网络向量化.jpg)



### 3.4 多个样本的向量化

* 注意每一个量的规模（常数或一维行列向量或矩阵等）



### 3.6 激活函数

* 之前我们使用的都是sigmoid函数`σ`，其介于0和1中间。

* 其他的激活函数：

  a = tanh(z) = (e^z^ - e^-z^) / (e^z^ + e^-z^) ，其介于-1和1之间。优点是范围更大且输入趋近0时输出趋近0；缺点是输入较大时，输出无限接近1，区分度很小。

  a = max(0, z) ，线性修正单元ReLU。优点是形式简单，计算速度快。缺点是 z < 0 时输出始终为0。

  a = max(0.01z, z) , Leaky ReLU

* 不同层之间的激活函数可以不同。但在二分分类中（最终y取0或1），最后一层的激活函数只能用 σ 。

* 注意到激活函数都是非线性的，这是因为线性函数的乘积仍然是线性函数，这样多层线性激活函数的神经网络可以用一层网络替代，失去了原本的意义。



### 3.8 激活函数的导数

 

### 3.9 神经网络的梯度下降法

* 推导过程与第二章中类似，注意多层神经网络逐层反向传播。

![](D:\2021-2022-2\深度学习资料\3.9.1  神经网络的梯度下降.jpg)



### 3.11 随机初始化

* 不能简单地将w和b初始化为0，这样会导致每一层的所有单元进行完全相同的运算。
* 使用 `w = np.random.randn((m,n))*0.01`随机初始化w，m和n根据神经网络的规模决定，常数可以不取0.01。



 ## WEEK 4 深层神经网络 Deep Neural Network

### 4.3 深层神经网络中的前向传播

* 与之前相同，每一层都使用 z^[i]^ = w^[i]^a^[i-1]^+b^[i]^ ;a^[i]^ = g^[i]^(z^[i]^) 计算，其中 g^[i]^ 为第i层的激活函数。
* 使用for循环一层层推进计算。



### 4.4 核对矩阵的维数

* z^[i]^ 和 b^[i]^ 的维数为 (n^[i]^,1) ，其中 n^[i]^为第i层的单元数；

  w^[i]^的维数为 (n^[i]^,n^[i-1]^) ，因为 a^[i-1]^ 是 (n^[i-1]^,1) 的；

* 考虑同时训练m个样本时的向量化。

  Z^[i]^ 的维数变为 (n^[i]^,m) ，A^[i-1]^ 的维数也变为 (n^[i-1]^,m) ，但 b^[i]^ 的维数仍为 (n^[i]^,1) 。

  

### 4.5 多层神经网络的优势

* 在神经元数量相同的情况下，将它们排布成多层，可以提高学习速度。
* 可以利用多层神经网络实现与或非的逻辑运算或者实现更复杂的函数。
* 层数也不是越多越好，要看实际情况。



### 4.2 前向传播与反向传播

* 前向传播：输入 a^[l-1]^ ,我们要得到 a^[l]^ ,并且记录 z^[l]^ 用于反向传播。

  Z^[l]^ = W^[l]^ * A^[l-1]^ +b^[l]^ ;

  A^[l]^ = g^[l]^(Z^[l]^) ;

* 反向传播：输入da^[l]^ ,我们要得到da^[l-1]^ ,dW^[l]^ ,db^[l]^

  dZ^[l]^ = dA^[l]^ * g^[l]^'(Z^[l]^) ;

  dW^[l]^ = dZ^[l]^ *A^[l-1]T^ / m ;

  db^[l]^ = np.sum(dZ^[l]^, axis = 1, keepdims = True) / m ;

  dA^[l-1]^ = W^[l]T^ * dZ^[l]^ ;

### 4.6 搭建深层神经网络块

### 4.7 参数与超参数

* 参数：W^[i]^, b^[i]^ and so on

* 超参数：learning rate `α`,

  ​				iteration times `i`,

  ​				hidden layer numbers `L`,

  ​				hidden unity `ni`, （第i个隐藏层的神经元数量）

  ​				choice of activation function

  ......and so on

* 超参数决定了最后得到的W和b





## 一阶段总结

神经网络编程的一般步骤：

1. 定义神经网络结构：有几层，每一层有几个神经元，用 `n_i = k` 表示；输入输出层的神经元数量要根据实际问题决定。

2. 初始化模型参数：用for循环初始化每一层的W和b，注意维数。b可以初始化为0。

   ```
   for(i in range(1,len(layer_dims))):
   	parameter['W' + str(i)] = np.random.randn(n[i], n[i-1]) * 0.01
   	parameter['b' + str(i)] = np.random.randn(n[i], 1) * 0.01
   ```
   
3. 实现前向传播
4. 计算损失函数
5. 实现反向传播
6. 更新参数
7. 构建完整神经网络
8. 使用训练集训练参数W和b
9. 对测试集做出预测



## CLASS 2 WEEK 1

### 1.1 训练 开发 测试集

* 将所有数据分为训练集 `training set`、验证集 `development set`和测试集 `test set`

* 对于较小规模的数据集，可以按照6：2：2的比例划分训练集、验证集和测试集；对于较大规模的数据集，验证与测试的比例要适当缩小。

* 如果不需要对算法进行无偏估计，则验证集和测试集可以合二为一。

  

### 1.2 偏差 方差(bias_variance)

![](D:\2021-2022-2\深度学习资料\2-1.2.1 偏差与方差.jpg)

* 高偏差——拟合效果过差——欠拟合 `high bias -> underfitting`

  高方差——拟合效果过好——过拟合 `high variance -> overfitting`

* 可以通过分析在训练集上训练算法产生的误差和验证集上验证算法产生的误差诊断算法是否存在高偏差和高方差。

  当训练集误差远超Bayes误差（或最优误差）时，算法偏差过高；（算法对于训练集都无法很好拟合）

  `Bayes误差：一般情况下能达到的最小误差。比如用清晰的图片识别猫，最小误差可能趋近0；但用模糊的照片识别猫，最小误差可能会是10%`

  当验证集误差远超训练集误差时，算法方差过高。（算法可能过度拟合了训练集，导致测试集中仅与训练集有微小差异的部分被误判）

  

### 1.3 机器学习基础

* 解决高偏差：使用更大的神经网络，或训练更长的时间；
* 解决高方差：使用更大的数据集进行训练，或采用**正则化**。



### 1.4 正则化 Regularization

* L2 regularizaiton: J(w,b) = 1 / m * ΣL(y', y) + λ / 2m * ||w||~2~^2^ 

  其中 ||w||^2^ = w^T^w , λ 为正则化参数(regularization parameter)

  因此 dw 中会增加一项 λ / m * w

* 对于神经网络中向量化的情况：

   J(w^[1]^, b^[1]^, ... , w^[L]^, b^[L]^) = 1 / m * ΣL(y', y) + λ / 2m * Σ||w||~f~^2^ 

  其中 ||w||~f~^2^ = ΣΣ (w^[l]^~ij~)^2^ ，即 Σw^[l]^~i~^T^Σw[l]~j~^T^ ，下标F意为“Frobenius norm(范数)”

* L2正则化也被称为“权重衰减” ，其简单原理为：λ不为0时（采用正则化），dw增大，w -= dw 减小，使得 z = wa + b 更趋近0。对于tanh等函数，在0附近接近线性，因此拟合的结果会更接近线性（具体效果由λ大小决定），因此可以减少过拟合。



### 1.6 Dropout 正则化

* Dropout（随机失活）：随机消除神经网络内部的某些节点

* Inverted Dropout

  ```
  假设对某个神经网络第三层（l=3）实施Inverted Dropout，用a3表示所有节点的输出组成的二维矩阵， keep_prob 设为0.8（保留80％节点）
  设置dropout矩阵 d3 = np.random.randn(a3.shape[0],a3.shape[1])
  随机消除某些节点 a3 = np.multiply(a3,d3)
  向外拓展a3 a3 /= keep_prob （保留下来的节点值要等比例扩大，以保证结果大小不变）
  ```

* Dropout一般只在训练阶段使用，测试阶段不使用。
* 可以对每一层采用不同的keep_prob值。对于参数W矩阵较大的层，keep_prob通常设为较小值；输入层a0不需要dropout。
* Dropout主要用于计算机视觉领域。



### 1.8 其他正则化方法

* Early Stopping：提早结束梯度下降，减少训练时间以降低过拟合。
* L2正则化：注意要尝试不同的λ以找到最佳值。



### 1.9 归一化输入（Normalizing Inputs）

* 归一化输入的两个步骤：（类似于将正态分布变为标准正态分布）
  1. zero out（零均值化）   μ = 1/m * Σ^m^~i=1~x^(i)^
  2. normalize the variances（归一化方差）  σ^2^ = 1/m * Σ^m^~i=1~x^(i)2^

​          最后   x = (x-μ) / σ^2^

* 归一化输入可以使得训练过程更加快速。



### 1.10 梯度消失与梯度爆炸（Vanishing/Exploding）

* 当神经网络层数很多时，由于参数w和b可能很大或很小，经过多次运算后，得到的预测结果y‘可能会非常大（梯度爆炸）或非常小（梯度消失）

  

### 1.11 神经网络的权重初始化

* 初始化方差 `Var(w) = 1 / n`

  初始化参数w w^[l]^ = np.random.randn(shape) * np.sqrt(1 / n^[l-1]^)

* 当使用ReLU作为激活函数时，将上式中的1改为2.

* `np.sqrt(...)` 称为方差参数

* 通过这样初始化，可以解决梯度消失与梯度爆炸的问题。



### 1.12 梯度的数值逼近
