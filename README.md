# AlphaNet

复现和优化华泰金工在2020年提出的端到端因子挖掘神经网络模型 [`AlphaNet`](https://crm.htsc.com.cn/doc/2020/10750101/74856806-a2e3-41cb-be4c-695dc6cc1341.pdf)，并搭建训练和预测框架，探索模型在多种金融场景下的应用。


## 课题背景

华泰金工于2020年6月设计了一种全新的网络结构：AlphaNet，能够端到端地解决多因子选股中的因子生成和多因子合成步骤，从而有效避免了传统方法中多步骤学习的人工干预和信息损失（例如：人工构造因子表达式、对多因子进行线性加权合成等）。

在传统的机器学习方法中，因子挖掘和多因子合成依然是两个完全分离的步骤，人工干预较多，容易存在不可避免的信息损失。AlphaNet模型（黑色虚线框）则借助深度学习端到端的特性，有效避免这一情况。

<center>
<img src="Images/端到端.png" width="500" align="center"/>
</center>


## 模型复现

这里我们考虑的模型，指的是 [`AlphaNet-v2`](https://bigquant.com/wiki/doc/rengongzhineng-xilie-AlphaNet-jiegou-tezheng-zhengquan-20200824-gZhImiZjLC)，即华泰金工在提出最初的AlphaNet模型后2个月发布的改进版。

模型的结构：

1. **数据输入**：由个股日频量价数据构成的 “数据图片”
2. **特征提取层**：通过类似卷积的思想，对二维图表数据进行特征提取
3. **LSTM层**：通过LSTM模型，学习特征中的时序信息
4. **输出层**：将特征进行加权和，输出为预测值

<center>
<img src="Images/net_lstm.png" width="600" align="center"/>
</center>


### 数据输入

股票数据虽然是时间序列数据，直观上用**RNN**（循环神经网络）等传统时序模型对其进行建模会比较有效。但是RNN的递归运算方式过于单一，很难有效地提取到股票数据中较为复杂的特征。

所以，AlphaNet借鉴了计算机视觉领域中最具影响力的**CNN**（卷积神经网络）网络的工作原理，将个股日频量价数据转换为 **`“数据图片”`**，然后通过一种类似卷积的计算方式来提取特征。

<center>
<img src="Images/数据图片.png" width="450" align="center"/>
</center>

“数据图片” 的纵向是特征维度，横向是时间维度。如上图的第一行则是：某只个股，在为期30天的历史回看窗口区间内，每天的开盘价数值。


### 特征提取层

CNN中传统的卷积计算有2个问题：

1. 是基于局部感知的，和输入数据的排布方式有很大关系，然而股票数据和图片不一样，没有固定的排布方式，因此不同的排布方式很有可能会影响模型的效果
2. 本质上只是计算固定特征数据的加权组合，极大程度上限制了因子表达式的可能性

因此，AlphaNet引入了自定义的特征提取层，通过多种**运算符函数**，并通过**完整遍历**的方式，更加丰富且全面地提取 “数据图片” 中的信息。

<center>
<img src="Images/特征提取.png" width="650" align="center"/>
</center>

自定义的特征提取函数可以分为两大类：双变量函数和单变量函数。

例如 **`ts_corr(X, Y, 3)`** 就是双变量函数，对 “数据图片” 中的所有特征进行两两遍历匹配，计算两个窗口之间的相关度：

<center>
<img src="Images/双变量卷积.png" width="650" align="center"/>
</center>

代码实现：

```python
class ts_corr(nn.Module):
    """
    计算过去 d 天 X 值构成的时序数列和 Y 值构成的时序数列的相关系数
    """

    def __init__(self, d=10, stride=10):
        """
        d: 计算窗口的天数
        stride：计算窗口在时间维度上的进步大小
        """
        super(ts_corr, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):

        # n-特征数量，T-时间窗口
        batch_size, n, T = X.shape

        # 初始化输出特征图
        w = int((T - self.d) / self.stride + 1)
        h = int(n * (n - 1) / 2)
        Z = torch.zeros(batch_size, h, w)

        # 遍历每个batch
        for batch in range(batch_size):
            # 主窗口：i 确定时间维度位置，j 确定特征维度位置
            for i in range(w):
                z = []
                start = i * self.stride
                end = start + self.d
                for j in range(n - 1):
                    # 主窗口
                    x = X[batch, j, start:end]
                    # 剩余窗口
                    y = X[batch, j + 1:, start:end]
                    # 计算两个窗口之间的相关系数
                    broadcasted_x = x.expand(len(y), -1)
                    r = pearsonr(broadcasted_x, y)
                    z.append(r)

                # 更新特征图
                Z[batch, :, i] = torch.cat(z, dim=0).T

        return Z
```

而 **`ts_stddev(X, 3)`** 就是单变量函数，会遍历 “数据图片” 中的所有特征窗口，计算窗口内数据的方差：

<center>
<img src="Images/方差特征.png" width="650" align="center"/>
</center>

代码实现：

```python
class ts_stddev(nn.Module):
    """
    过去 d 天 X 值构成的时序数列的标准差
    """

    def __init__(self, d=10, stride=10):
        """
        d: 计算窗口的天数
        stride：计算窗口在时间维度上的进步大小
        """
        super(ts_stddev, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):

        # n-特征数量，T-时间窗口  
        batch_size, n, T = X.shape

        # 初始化输出特征图
        w = int((T - self.d) / self.stride + 1)
        Z = torch.zeros(batch_size, n, w)

        # 遍历每个batch
        for batch in range(batch_size):
            # 窗口：i 确定时间维度位置
            for i in range(w):
                start = i * self.stride
                end = start + self.d
                x = X[batch, :, start:end]
                # 计算窗口的方差
                std = torch.std(x, dim=1)
                # 更新特征图
                Z[batch, :, i] = std
              
        return Z
```

剩下的自定义特征提取层的计算逻辑和上面两种框架基本一致，只是需要根据函数的定义改变一下实际计算的值。


### LSTM层

经过特征提取层得到的特征仍然具有时序信息，所以我们需要用像 **LSTM** 这样的网络结构来捕捉特种中的时序信息。

<center>
<img src="Images/时序信息.png" width="400" align="center"/>
</center>

- lstm怎么捕捉时序信息？

代码实现：

```python
# 初始化LSTM层和批量归一化层
self.lstm = nn.LSTM(n_in, 30, 1, batch_first=True)
self.bn = nn.BatchNorm1d(30)

# LSTM + 批量归一化
features, _ = self.lstm(features)
features = features.transpose(1,2)
features = self.bn(features)
features = features.transpose(1,2)
         
# 取LSTM最后一个时间步的隐藏状态作为最后的特征
features = features[:, -1, :]
```


### 输出层


