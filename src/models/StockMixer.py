import torch
import torch.nn as nn
import torch.nn.functional as F

# 激活函数
acv = nn.GELU()


class MixerBlock(nn.Module):
    def __init__(self, mlp_dim, hidden_dim, dropout=0.0):
        super(MixerBlock, self).__init__()
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.LN = acv
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.LN(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        return x


class TriU(nn.Module):
    def __init__(self, time_step):
        super(TriU, self).__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList(
            [
                nn.Linear(i + 1, 1)
                for i in range(time_step)
            ]
        )

    def forward(self, inputs):
        x = self.triU[0](inputs[:, :, 0].unsqueeze(-1))
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[:, :, 0:i + 1])], dim=-1)
        return x


class Mixer2dTriU(nn.Module):
    def __init__(self, time_steps, channels):
        super(Mixer2dTriU, self).__init__()
        self.LN_1 = nn.LayerNorm([time_steps, channels])
        self.LN_2 = nn.LayerNorm([time_steps, channels])
        self.timeMixer = TriU(time_steps)
        self.channelMixer = MixerBlock(channels, channels)

    def forward(self, inputs):
        x = self.LN_1(inputs)
        x = x.permute(0, 2, 1)
        x = self.timeMixer(x)
        x = x.permute(0, 2, 1)

        x = self.LN_2(x + inputs)
        y = self.channelMixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    def __init__(self, time_step, channel, scale_dim=8):
        super(MultTime2dMixer, self).__init__()
        self.mix_layer = Mixer2dTriU(time_step, channel)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channel)

    def forward(self, inputs, y):
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)


class NoGraphMixer(nn.Module):
    """适配batch方式的股票混合器
    
    使用动态LayerNorm替代固定stocks维度的LayerNorm
    """
    def __init__(self, feature_dim, hidden_dim=20):
        super(NoGraphMixer, self).__init__()
        # 使用特征维度进行LayerNorm
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dense1 = nn.Linear(feature_dim, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, inputs):
        # inputs: [batch_size, feature_dim]
        x = self.layer_norm(inputs)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        return x


class StockMixerCore(nn.Module):
    """StockMixer核心模型（适配batch方式）
    
    Args:
        time_steps: 时间步长
        channels: 特征数量
        market: 市场维度隐藏层大小
        scale: 缩放因子数量（未使用，保留兼容性）
    """
    def __init__(self, time_steps, channels, market, scale):
        super(StockMixerCore, self).__init__()
        # scale_dim 动态计算为 time_steps // 2（Conv1d stride=2 后的维度）
        scale_dim = time_steps // 2
        self.time_steps = time_steps
        self.channels = channels
        self.scale_dim = scale_dim
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale_dim)
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(time_steps * 2 + scale_dim, 1)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        # 股票混合器使用时间维度作为特征维度
        self.stock_mixer = NoGraphMixer(time_steps * 2 + scale_dim, market)
        self.time_fc_ = nn.Linear(time_steps * 2 + scale_dim, 1)

    def forward(self, inputs):
        # inputs: [batch_size, time_steps, channels]
        x = inputs.permute(0, 2, 1)  # [batch_size, channels, time_steps]
        x = self.conv(x)  # [batch_size, channels, time_steps//2]
        x = x.permute(0, 2, 1)  # [batch_size, time_steps//2, channels]
        y = self.mixer(inputs, x)  # [batch_size, time_steps*2 + scale_dim, channels]
        y = self.channel_fc(y).squeeze(-1)  # [batch_size, time_steps*2 + scale_dim]

        z = self.stock_mixer(y)  # [batch_size, time_steps*2 + scale_dim]
        y = self.time_fc(y)  # [batch_size, 1]
        z = self.time_fc_(z)  # [batch_size, 1]
        return y + z


class StockMixer(nn.Module):
    """StockMixer模型包装类
    
    用于与QniverseModel框架集成的StockMixer模型。
    接受configs对象作为参数，从中提取所需配置。
    
    配置参数:
        - enc_in: 输入特征数量 (fea_num)
        - seq_len: 时间步长 (lookback_length)  
        - c_out: 输出维度 (steps)
        - market_num: 市场隐藏层维度
        - scale_factor: 缩放因子
        - dropout: dropout比率
        - tau_hat_init: tau初始化值
    """
    
    def __init__(self, configs):
        super(StockMixer, self).__init__()
        
        # 从configs中提取参数
        self.input_size = configs.enc_in  # 特征数量 (fea_num)
        self.seq_len = configs.seq_len  # 时间步长 (lookback_length)
        self.output_size = configs.c_out  # 输出维度 (steps)
        self.market_num = getattr(configs, 'market_num', 20)  # 市场隐藏层维度
        self.scale_factor = getattr(configs, 'scale_factor', 3)  # 缩放因子
        self.dropout = getattr(configs, 'dropout', 0.1)
        
        # alpha参数用于WeightedMSELoss
        self.alpha = torch.nn.Parameter(
            torch.tensor(float(getattr(configs, 'tau_hat_init', 0.0)), dtype=torch.float32)
        )
        
        # 输入dropout
        self.input_drop = nn.Dropout(self.dropout)
        
        # 核心StockMixer模型（适配batch方式）
        # 输入shape: [batch_size, time_steps, channels]
        self.core_model = StockMixerCore(
            time_steps=self.seq_len,
            channels=self.input_size,
            market=self.market_num,
            scale=self.scale_factor
        )
        
        # 最终投影层
        self.projection = nn.Linear(1, self.output_size)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入张量, shape [batch_size, seq_len, enc_in]
            
        Returns:
            输出张量, shape [batch_size, c_out]
        """
        # 应用输入dropout
        x = self.input_drop(x)
        
        # 核心模型处理
        out = self.core_model(x)  # [batch_size, 1]
        
        # 投影到输出维度
        out = self.projection(out)  # [batch_size, c_out]
        
        return out
