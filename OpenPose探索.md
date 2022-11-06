# OpenPose探索

## 3D重建

- OpenPose指导的[3D重建](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/01_demo.md#3-d-reconstruction)
- 使用Kinect进行3D重建步骤：[获取Kinect dk中彩色图](https://blog.csdn.net/qq_37792894/article/details/111311496)，并将彩色图放入openpose中，得到手部骨骼点。然后从kinect dk的深度图中获取每个骨骼点对应到的深度值，从而得到三维骨骼点。

## 网络训练

- 训练数据集 [LMDB文件介绍](https://zhuanlan.zhihu.com/p/70359311)
  - LMDB [文件制作](https://blog.csdn.net/zxyhhjs2017/article/details/80719103)
- OpenPose部署 [训练指导](https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/tree/master/training) 
- Caffe [学习资源](http://caffe.berkeleyvision.org/)

## OpenPose原理

### multi-stage CNN

![OpenPose NN architecture](F:\Involution\3D_pose_Innovation\LearningNotes\pictures\OpenPose NN architecture.png)
$$
L^t = \phi^t(F, L^{t-1}) \quad 2 \le t \le T_P
$$

$$
\begin{equation}
S^t = 
\begin{cases}
\rho^t(F,L^{T_P}) & t = T_P \\
\rho^t(F,L^{T_P},S^{t-1}) & T_P < t \le T_P + T_C
\end{cases}
\end{equation}
$$

**$\phi^t$ and $\rho^t$ refers to the CNNs for inference at Stage t (Refines over successive stages)** **, we have $J$ confidence maps and $C$ $PAFs$, corresponding to  $S_{j}$ and $L_{c}$ .**

1. Get  feature maps  $F$  from initial image

   - Get  feature maps  $F$ --- `《Very Deep Convolutional Networks》`
     - ConvNets --- `《ImageNet classification with deep convolutional neural networks》`

2. produces a set of part affinity field ( $PAFs$ ---- $L^t$ )

   - Refines the predictions over successive stages（迭代细化）
     - Convolutional Pose Machine 
     - Pose Machine
   - the receptive field is preserved while the computation is reduced（改进卷积层架构）
     - DenseNet --- `《Densely Connected Convolutional Networks》`
     - ResNet

3. the confidence maps detection ( $S^t$ )

   - What is Confidence Map

4. Confidence Maps for Part Detection ( $S^{\star}_{j}$ )**（数据集相关）**

   - generate individual confidence maps (from origin annotated 2D image) . 
   - if multiple people are in the image, there should be a peak in each confidence map corresponding to each visible **part $j$** for each **person $k$** . ( $S^{\star}_{j,k}$ )
   - The groundtruth confidence map predicted by the network is an aggregation of the individual confidence maps via a max operator (通过融合每个人的 Body Part 获得一张图像中所有的 Body Part 的置信图)

5. Part Affinity Fields for Part Association ( $L^{\star}_{c}$ )**（数据集相关）**

   - If a point $p$ lies on the limb, the value at $L^{\star}_{c,k}(p)$ is a unit vector $v$ that points from $j_{1}$ to $j_2$ ; **for all other points, the vector is zero-valued.**

   - How to define the set of points on the limb ?

     ![The set of points on the limb](F:\Involution\3D_pose_Innovation\LearningNotes\pictures\The set of points on the limb.png)

   $$
   \begin{cases}
   0 \le v\ · \ (p - x_{j1, k}) \le l_{c,k} \\
   |v_{vertical} \ · \ (p - x_{j1, k}) | \le \sigma_{l}
   \end{cases}
   $$

   - The groundtruth part affinity field **averages** the affinity
     fields of all people in the image.（对所有人所有“四肢”的 $PAF$​ 求平均，分母为 the number of **non-zero vectors** at point $p$ across all $k$ people)
   - measure association ( $E$ ) between candidate part detections. (通过$PAF$图，衡量 Body part 与 Limb 的关联程度，实际上是对 $PAF$ 向量做 Body part 连接方向上的线积分)

6. Multi-Person Parsing using PAFs

   **(对于本项目而言，多人混合检测，并非重点)**