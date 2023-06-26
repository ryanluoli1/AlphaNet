import torch
import torch.nn as nn
from audtorch.metrics.functional import pearsonr



# --------------------特征提取层--------------------

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


class ts_cov(nn.Module):
    """
    计算过去 d 天 X 值构成的时序数列和 Y 值构成的时序数列的协方差
    """

    def __init__(self, d=10, stride=10):
        super(ts_cov, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):
        batch_size, n, T = X.shape
        w = int((T - self.d) / self.stride + 1)
        h = int(n * (n - 1) / 2)
        Z = torch.zeros(batch_size, h, w)
        for batch in range(batch_size):
            for i in range(w):
                z = []
                start = i * self.stride
                end = start + self.d
                for j in range(n - 1):
                    x = X[batch, j, start:end]
                    y = X[batch, j + 1:, start:end]
                    x = x.expand(len(y), -1)
                    x_bar = x.mean(dim=1).unsqueeze(dim=1).expand(y.shape[0], y.shape[1])
                    y_bar = y.mean(dim=1).unsqueeze(dim=1).expand(y.shape[0], y.shape[1])
                    cov = torch.sum((x - x_bar) * (y - y_bar), dim=1) / y.shape[1]
                    z.append(cov)
                Z[batch, :, i] = torch.cat(z, dim=0).T
        return Z


class ts_return(nn.Module):
    """
    过去 d 天 X 值构成的时序数列的return
    """

    def __init__(self, d=10, stride=10):
        """
        d: 计算窗口的天数
        stride：计算窗口在时间维度上的进步大小
        """
        super(ts_return, self).__init__()
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
                # 计算窗口的return
                return_d = (x[:, -1] - x[:, 0]) / x[:, 0] - 1
                # 更新特征图
                Z[batch, :, i] = return_d

        return Z


class ts_stddev(nn.Module):
    """
    过去 d 天 X 值构成的时序数列的标准差
    """

    def __init__(self, d=10, stride=10):
        super(ts_stddev, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):
        batch_size, n, T = X.shape
        w = int((T - self.d) / self.stride + 1)
        Z = torch.zeros(batch_size, n, w)
        for batch in range(batch_size):
            for i in range(w):
                start = i * self.stride
                end = start + self.d
                x = X[batch, :, start:end]
                std = torch.std(x, dim=1)
                Z[batch, :, i] = std
        return Z


class ts_zscore(nn.Module):
    """
    过去 d 天 X 值构成的时序数列的z-score
    """

    def __init__(self, d=10, stride=10):
        super(ts_zscore, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):
        batch_size, n, T = X.shape
        w = int((T - self.d) / self.stride + 1)
        Z = torch.zeros(batch_size, n, w)
        for batch in range(batch_size):
            for i in range(w):
                start = i * self.stride
                end = start + self.d
                x = X[batch, :, start:end]
                z_score = torch.mean(x, dim=1) / torch.std(x, dim=1)
                Z[batch, :, i] = z_score
        return Z


class ts_decaylinear(nn.Module):
    """
    过去 d 天 X 值构成的时序数列的加权平均值
    """

    def __init__(self, d=10, stride=10):
        super(ts_decaylinear, self).__init__()
        self.d = d
        self.stride = stride

    def forward(self, X):
        batch_size, n, T = X.shape
        w = int((T - self.d) / self.stride + 1)
        Z = torch.zeros(batch_size, n, w)

        # 权重
        weights = torch.arange(self.d) + 1
        normalized_w = weights / torch.sum(weights)

        for batch in range(batch_size):
            for i in range(w):
                start = i * self.stride
                end = start + self.d
                x = X[batch, :, start:end]
                weighted_avg = torch.mm(normalized_w.unsqueeze(dim=0), x.T)
                Z[batch, :, i] = weighted_avg
        return Z
    
    

    
# --------------------AlphaNet-v1--------------------

class AlphaNet(nn.Module):
    '''
    第一版AlphaNet：输入 + 特征提取层 + 池化层 + 特征展平/残差连接 + 降维度层 + 输出层
    '''

    def __init__(self, d=10, stride=10, d_pool=3, s_pool=3, n=9):
        super(AlphaNet, self).__init__()
        
        # d-回看窗口大小，stride-时间步大小
        self.d = d
        self.stride = stride

        # ts_corr() 和 ts_cov() 输出的特征数量
        h = int(n * (n - 1) / 2)

        # 特征提取层
        self.feature_extractors = nn.ModuleList([
            ts_corr(self.d, self.stride),
            ts_cov(self.d, self.stride),
            ts_stddev(self.d, self.stride),
            ts_zscore(self.d, self.stride),
            ts_return(self.d, self.stride),
            ts_decaylinear(self.d, self.stride),
            nn.AvgPool1d(self.d, self.stride)
        ])

        # 特征提取层后面接的批归一化层
        self.batch_norms1 = nn.ModuleList([
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n)
        ])

        # 池化层（无参数）
        self.avg_pool = nn.AvgPool1d(d_pool, s_pool)
        self.max_pool = nn.MaxPool1d(d_pool, s_pool)
        
        # 池化层后面接的批量归一化层
        self.batch_norms2 = nn.ModuleList([])
        for _ in range(2):
            for _ in range(3):
                self.batch_norms2.append(nn.BatchNorm1d(h))
        for _ in range(5):
            for _ in range(3):
                self.batch_norms2.append(nn.BatchNorm1d(n))

        # 特征展平并拼接后的总数
        n_in = 2 * (h*2*3 + n*5*3)

        # 线性层，输出层，激活函数，失活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.linear_layer = nn.Linear(n_in, 30)
        self.output_layer = nn.Linear(30, 1)

        # 初始化线性层和输出层
        self.initialize_weights()


    def initialize_weights(self):
        
        # 用 truncated_normal 方法初始化线性层和输出层的权重
        nn.init.trunc_normal_(self.linear_layer.weight)
        nn.init.trunc_normal_(self.output_layer.weight)


    def forward(self, X):
        
        features_fe, features_p, i = [], [], 0
        for extractor, batch_norm in zip(self.feature_extractors, self.batch_norms1):
            
            # 特征提取 + 批量归一化 + 展平
            x = extractor(X)
            x = batch_norm(x)
            features_fe.append(x.flatten(start_dim=1))
            
            # 池化层 + 批量归一化 + 展平
            x_avg = self.batch_norms2[i](self.avg_pool(x))
            x_max = self.batch_norms2[i+1](self.max_pool(x))
            x_min = self.batch_norms2[i+2](-self.max_pool(-x))
            features_p.append(x_avg.flatten(start_dim=1))
            features_p.append(x_max.flatten(start_dim=1))
            features_p.append(x_min.flatten(start_dim=1))
            i += 3
         
        # 残差连接
        f1 = torch.cat(features_fe, dim=1)
        f2 = torch.cat(features_p, dim=1)
        features = torch.cat([f1, f2], dim=1)
        
        # 线性层 + 激活 + 失活 + 输出层
        features = self.linear_layer(features)
        features = self.relu(features)
        features = self.dropout(features)
        output = self.output_layer(features)

        return output

    
    

# --------------------AlphaNet-v2--------------------

class AlphaNet_v2(nn.Module):
    '''
    研报中的改进版，相比AlphaNet-v1:
        1. 扩充了6比率类特征
        2. 用LSTM替换池化层和全连接层
    '''

    def __init__(self, d=10, stride=10, n=15):
        super(AlphaNet_v2, self).__init__()
        
        # d-回看窗口大小，stride-时间步大小
        self.d = d
        self.stride = stride

        # ts_corr() 和 ts_cov() 输出的特征数量
        h = int(n * (n - 1) / 2)

        # 特征提取层
        self.feature_extractors = nn.ModuleList([
            ts_corr(self.d, self.stride),
            ts_cov(self.d, self.stride),
            ts_stddev(self.d, self.stride),
            ts_zscore(self.d, self.stride),
            ts_return(self.d, self.stride),
            ts_decaylinear(self.d, self.stride)
        ])

        # 特征提取层后面接着的批量归一化
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n)
        ])
        
        
        # 特征总数
        n_in = 2 * h + 4 * n

        # LSTM层，批量归一化，输出层
        self.lstm = nn.LSTM(n_in, 30, 1, batch_first=True)
        self.bn = nn.BatchNorm1d(30)
        self.output_layer = nn.Linear(30, 1)

        # 初始化输出层的权重
        self.initialize_weights()


    def initialize_weights(self):
        # 用 truncated_normal 方法初始化输出层的权重
        nn.init.trunc_normal_(self.output_layer.weight)


    def forward(self, X):

        # 特征提取 + 批量归一化
        features = []
        for extractor, batch_norm in zip(self.feature_extractors, self.batch_norms):
            x = extractor(X)
            x = batch_norm(x)
            features.append(x)
        features = torch.cat(features, dim=1)

        # 将输入转换为: (batch_size, sequence_length, feature_size)
        features = features.transpose(1, 2)

        # LSTM + 批量归一化
        features, _ = self.lstm(features)
        features = features.transpose(1,2)
        features = self.bn(features)
        features = features.transpose(1,2)
         
        # 取LSTM最后一个时间步的隐藏状态作为最后的特征
        features = features[:, -1, :]
        
        # 输出层
        output = self.output_layer(features)

        return output
    
   
    

# --------------------AlphaNet-Attention--------------------

class AlphaNet_att(nn.Module):
    '''
    在AlphaNet-v2的基础上，用Multi-head Self-Attention层替换掉LSTM层
    '''

    def __init__(self, d=10, stride=10, n=15):
        super(AlphaNet_att, self).__init__()

        # d-回看窗口大小，stride-时间步大小
        self.d = d
        self.stride = stride

        # ts_corr() 和 ts_cov() 输出的特征数量
        h = int(n * (n - 1) / 2)

        # 特征提取层
        self.feature_extractors = nn.ModuleList([
            ts_corr(self.d, self.stride),
            ts_cov(self.d, self.stride),
            ts_stddev(self.d, self.stride),
            ts_zscore(self.d, self.stride),
            ts_return(self.d, self.stride),
            ts_decaylinear(self.d, self.stride)
        ])

        # 特征提取层后面接的批量归一化层
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(h),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n),
            nn.BatchNorm1d(n)
        ])
        

        # 特征总数
        n_in = 2 * h + 4 * n
        
        # 多头自注意力机制 + 层归一化
        self.mha = nn.MultiheadAttention(embed_dim=n_in,
                                         num_heads=3,
                                         dropout=0.1,
                                         batch_first=True)
        self.ln = nn.LayerNorm(n_in)
        
        # 输出层
        self.output_layer = nn.Linear(n_in, 1)

        # 初始化输出层的权重
        self.initialize_weights()


    def initialize_weights(self):
        # 用 truncated_normal 方法初始化输出层的权重
        nn.init.trunc_normal_(self.output_layer.weight)


    def forward(self, X):
        
        # 特征提取 + 批量归一化
        features = []
        for extractor, batch_norm in zip(self.feature_extractors, self.batch_norms):
            x = extractor(X)
            x = batch_norm(x)
            features.append(x)
        features = torch.cat(features, dim=1)

        # 将输入转换为: (batch_size, sequence_length, feature_size)
        features = features.transpose(1,2)
        
        # 自注意力计算 + 层归一化
        features, _ = self.mha(features, features, features)
        features = self.ln(features)
        
        # 取所有时间步的均值作为最后的特征
        features = torch.mean(features, dim=1)
        
        # 输出层
        output = self.output_layer(features)

        return output
