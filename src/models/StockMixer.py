import torch
import torch.nn as nn
import torch.nn.functional as F

# activation function
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
    """Batch-friendly stock mixer.
    
    Uses a dynamic LayerNorm instead of a LayerNorm over a fixed stocks dimension.
    """
    def __init__(self, feature_dim, hidden_dim=20):
        super(NoGraphMixer, self).__init__()
        # LayerNorm over the feature dimension
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
    """StockMixer core model (batch-friendly).
    
    Args:
        time_steps: number of time steps
        channels: number of features
        market: market-dimension hidden size
        scale: number of scale factors (unused, kept for compatibility)
    """
    def __init__(self, time_steps, channels, market, scale):
        super(StockMixerCore, self).__init__()
        # scale_dim is computed dynamically as time_steps // 2 (the dimension after Conv1d with stride=2)
        scale_dim = time_steps // 2
        self.time_steps = time_steps
        self.channels = channels
        self.scale_dim = scale_dim
        self.mixer = MultTime2dMixer(time_steps, channels, scale_dim=scale_dim)
        self.channel_fc = nn.Linear(channels, 1)
        self.time_fc = nn.Linear(time_steps * 2 + scale_dim, 1)
        self.conv = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=2, stride=2)
        # the stock mixer uses the time dimension as the feature dimension
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
    """StockMixer wrapper class.
    
    StockMixer model integrated with the QniverseModel framework.
    Takes a configs object and reads the required settings from it.
    
    Config parameters:
        - enc_in: number of input features (fea_num)
        - seq_len: number of time steps (lookback_length)  
        - c_out: output dimension (steps)
        - market_num: market hidden dimension
        - scale_factor: scale factor
        - dropout: dropout rate
        - tau_hat_init: initial tau value
    """
    
    def __init__(self, configs):
        super(StockMixer, self).__init__()
        
        # extract parameters from configs
        self.input_size = configs.enc_in  # number of features (fea_num)
        self.seq_len = configs.seq_len  # number of time steps (lookback_length)
        self.output_size = configs.c_out  # output dimension (steps)
        self.market_num = getattr(configs, 'market_num', 20)  # market hidden dimension
        self.scale_factor = getattr(configs, 'scale_factor', 3)  # scale factor
        self.dropout = getattr(configs, 'dropout', 0.1)
        
        # alpha parameter used by WeightedMSELoss
        self.alpha = torch.nn.Parameter(
            torch.tensor(float(getattr(configs, 'tau_hat_init', 0.0)), dtype=torch.float32)
        )
        
        # input dropout
        self.input_drop = nn.Dropout(self.dropout)
        
        # core StockMixer model (batch-friendly)
        # input shape: [batch_size, time_steps, channels]
        self.core_model = StockMixerCore(
            time_steps=self.seq_len,
            channels=self.input_size,
            market=self.market_num,
            scale=self.scale_factor
        )
        
        # final projection layer
        self.projection = nn.Linear(1, self.output_size)
        
    def forward(self, x):
        """Forward pass.
        
        Args:
            x: input tensor, shape [batch_size, seq_len, enc_in]
            
        Returns:
            output tensor, shape [batch_size, c_out]
        """
        # apply input dropout
        x = self.input_drop(x)
        
        # core model
        out = self.core_model(x)  # [batch_size, 1]
        
        # project to the output dimension
        out = self.projection(out)  # [batch_size, c_out]
        
        return out
