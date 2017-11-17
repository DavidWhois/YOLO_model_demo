# object detection - YOLO algorithm
本文来自于Andrew Ng的Coursera课程总结，Convolutional Neural Networks:https://www.coursera.org/learn/convolutional-neural-networks/home

YOLO论文: Redmon et al., 2016 (https://arxiv.org/abs/1506.02640)
        Redmon and Farhadi, 2016 (https://arxiv.org/abs/1612.08242)

数据集: drive.ai  (https://www.drive.ai/)

## 1 - 问题背景

以自动驾驶的物体检测为例，假设你想用YOLO模型来识别图像中的物体类别(汽车，摩托车，行人，背景……)和位置，以及在图像中所占的大小。

识别的效果实际上就是在图像上给要识别的物体加上边界框(bounding boxes)，如下图所示。

<img src="images/1.png" style="width:500px;height:250;">

这里我们有两种方式来表示物体的类别，一种就是直接用实数表示(1,2,3),一种就是用one-hot向量表示([0,1,0],[1,0,0],[0,0,1]...),具体如何表示要根据应用开发过程的具体情况来定，两种表示都可以。

## 2 - YOLO

YOLO ("you only look once") 顾名思义，只做一次前向传播就完成预测，具体步骤如下。

### 2.1 - YOLO模型

输入输出：
- **输入** 图片: (m, img_w, img_h, 3)
- **输出** 为一系列边界框，每个边界框用i$(p_c, b_x, b_y, b_h, b_w, c)$ 表示。如上文所述，c也可以one-hot展开为向量形式。

为了应对同一位置上出现两个物体的情况，这里还使用了锚边框(anchor boxes)，其实就是在输出中增加了一个维度，输出多个边界框，具体见下文。

这里的例子使用5个锚边框，图像的尺寸为(m,608,608,3),所以YOLO模型的架构为:IMAGE (m, 608, 608, 3) -> DEEP CNN -> ENCODING (m, 19, 19, 5, 85)，如下图所示。


<img src="images/2.png" style="width:700px;height:400;">

使用19 x19的网格来分割图片，若物体A的中心点落在某网格内，该网格负责预测物体A的边界框。因为这里有5个锚边框，所以每个网格中应包含5个边界框的信息(p_c, b_x, b_y, b_h, b_w, c)

为了计算方便，将最后两个维度压缩为一个维度:(19, 19, 5, 85)-> (19, 19, 425)。所以这里Deep CNN 的输出维度为 (19, 19, 425)，如下图所示。

<img src="images/3.png" style="width:700px;height:400;">

计算每个网格中(19x19个)的每个边界框(5个)的可能性(p_c),这个p_c表示这个具体的框中，存在某类物体的可能性，如下图所示。

<img src="images/4.png" style="width:700px;height:400;">

下图做了YOLO模型预测的可视化，他将每个网格中(19x19个)的每个边界框(5个)中的p_c值最大的那个取出来，并根据这个p_c预测的是具体哪个类，涂上不同的颜色，就是如下效果:

<img src="images/5.png" style="width:300px;height:300;">

或者将所有的边界框都绘制出来，就是下面这个样子:

<img src="images/6.png" style="width:200px;height:200;">
这里模型在一次前向传播中一共预测了19x19x5 = 1805 个边界框。

现在，我们有了很多边界框，但是哪些才是我们的物体所在的那个呢？这里要做两层过滤：
- 将那些可能性（p_c)小于给定阈值(threshold)的都过滤掉。
- 将那些重合的边界框删除，只留下一个，这一步被称为Non-max suppression。这里解释一下，因为我们在同一个网格中使用了多个（这里是5个）锚边框，所以如果该网格中存在一个物体，那5个锚边框都会倾向输出一个结果，但是他们会重叠在一起，我们通过计算IOU值来衡量这个重叠的程度，删除那些重叠的边界框。下文会解释IOU的概念。

### 2.3 - Non-max suppression ###

从下面例子可以很清楚的看出Non-max suppression在干什么。

<img src="images/7.png" style="width:500px;height:400;">
模型预测了3个边界框，但是实际上只有一辆车，所以我们要删除两个框，留下可能性最大的那个。

Non-max suppression 使用IOU **"Intersection over Union"**,来计算重叠率，下图一目了然。
<img src="images/8.png" style="width:500px;height:400;">

## 思考
YOLO算法是利用卷积网络的很强大也很流行的算法，可以看到，通过对输出(labels)的设计，神经网络可以做出各种各样的输出。但是随着输出设计的复杂化，获取训练集的成本也随之提高，如何通过算法实现半监督甚至无监督学习，是机器学习最大的挑战之一。
