# AlphaNet

复现和优化华泰金工在2020年提出的端到端因子挖掘神经网络模型 [`AlphaNet`](https://crm.htsc.com.cn/doc/2020/10750101/74856806-a2e3-41cb-be4c-695dc6cc1341.pdf)，并搭建训练和预测框架，探索模型在多种金融场景下的应用。


## 课题背景

华泰金工于2020年6月设计了一种全新的网络结构：AlphaNet，能够端到端地解决多因子选股中的因子生成和多因子合成步骤，从而有效避免了传统方法中多步骤学习的人工干预和信息损失（例如：人工构造因子表达式、对多因子进行线性加权合成等）。

在传统的机器学习方法中，因子挖掘和多因子合成依然是两个完全分离的步骤，人工干预较多，容易存在不可避免的信息损失。AlphaNet模型（黑色虚线框）则借助深度学习端到端的特性，有效避免这一情况。

<center>
<img src="Images/端到端.png" width="500" align="center"/>
</center>


## 模型复现

这里，我们考虑的模型，指的是 [`AlphaNet-v2`](https://bigquant.com/wiki/doc/rengongzhineng-xilie-AlphaNet-jiegou-tezheng-zhengquan-20200824-gZhImiZjLC)，即华泰金工在提出最初的AlphaNet模型后2个月发布的改进版。

<center>
<img src="Images/net_lstm.png" width="600" align="center"/>
</center>

模型的结构是：

1. **输入**：有个股日频量价数据构成的 “数据图片”
2. **特征提取层**：通过类似卷积的思想，对二维图表数据进行特征提取
3. **LSTM层**：通过LSTM模型，学习特征中的时序信息
4. **输出层**：将特征进行加权和，输出为预测值
