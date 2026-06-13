"""
MERA (Mixture of Experts with Routing Attention) Model

Adapted from the original MERA implementation to be compatible with the current project framework.
This implementation uses pure PyTorch without fmoe dependency.

Reference: MERA-main/MERA
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class NoisyGate(nn.Module):
    """
    Noisy Top-K Gating for Mixture of Experts.
    Implements the noisy gating mechanism with load balancing loss.
    """
    
    def __init__(self, d_model, num_expert, top_k=2, noise_std=1.0, no_noise=False):
        super().__init__()
        self.num_expert = num_expert
        self.top_k = top_k
        self.noise_std = noise_std
        self.no_noise = no_noise
        
        # Gate weights
        self.w_gate = nn.Parameter(torch.zeros(d_model, num_expert), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        
        # For loss tracking
        self.loss = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))
    
    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates."""
        return (gates > 0).sum(0)
    
    def cv_squared(self, x):
        """The squared coefficient of variation of a sample."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0.0], device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)
    
    def set_loss(self, loss):
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss
    
    def get_loss(self):
        loss = self.loss
        self.loss = None
        return loss
    
    @property
    def has_loss(self):
        return self.loss is not None
    
    def forward(self, inp):
        """
        Args:
            inp: Input tensor of shape [batch_size, d_model]
        
        Returns:
            top_k_indices: Indices of selected experts [batch_size, top_k]
            top_k_gates: Gate values for selected experts [batch_size, top_k]
        """
        # Reshape input
        shape_input = list(inp.shape)
        channel = shape_input[-1]
        other_dim = shape_input[:-1]
        inp_flat = inp.reshape(-1, channel)
        
        # Compute gate logits
        clean_logits = inp_flat @ self.w_gate
        
        # Add noise during training
        noise_stddev = (self.noise_std / self.num_expert) * self.training
        if self.no_noise:
            noise_stddev = 0
        
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
        
        # Apply softmax
        logits = self.softmax(noisy_logits)
        
        # Get top-k experts
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.num_expert), dim=1
        )
        
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = top_k_logits
        
        # Compute load balancing loss
        if self.training:
            zeros = torch.zeros_like(logits, requires_grad=True)
            gates = zeros.scatter(1, top_k_indices, top_k_logits)
            
            load = self._gates_to_load(gates)
            importance = gates.sum(0)
            loss = self.cv_squared(importance) + self.cv_squared(load)
            self.set_loss(loss)
        
        # Reshape outputs
        top_k_indices = top_k_indices.reshape(other_dim + [self.top_k]).contiguous()
        top_k_gates = top_k_gates.reshape(other_dim + [self.top_k]).contiguous()
        
        return top_k_indices, top_k_gates


class Expert(nn.Module):
    """Single expert implemented as a GRU."""
    
    def __init__(self, d_model, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
        )
    
    def forward(self, x):
        out, _ = self.gru(x)
        return out


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer.
    Routes input to top-k experts and combines their outputs.
    """
    
    def __init__(self, d_model, d_gate, num_expert=8, top_k=2, dropout=0.1, noise_std=1.0):
        super().__init__()
        self.num_expert = num_expert
        self.top_k = top_k
        self.d_model = d_model
        
        # Create experts (shared GRU for efficiency, like in original MERA)
        self.shared_expert = Expert(d_model, num_layers=2)
        self.experts = nn.ModuleList([self.shared_expert for _ in range(num_expert)])
        
        # Gate network
        self.gate = NoisyGate(d_gate, num_expert, top_k=top_k, noise_std=noise_std)
        
        # Activation
        self.activation = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, inp, gate_inp):
        """
        Args:
            inp: Input tensor [batch_size, seq_len, d_model]
            gate_inp: Gate input tensor [batch_size, d_gate]
        
        Returns:
            output: Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = inp.shape
        
        # Get gate outputs
        top_k_indices, top_k_gates = self.gate(gate_inp)  # [batch, top_k]
        
        # Process through experts
        # For efficiency, we process all samples through the shared expert
        # and weight by gate values
        expert_outputs = []
        for k in range(self.top_k):
            # Get expert indices for this k
            expert_idx = top_k_indices[:, k]  # [batch_size]
            gate_value = top_k_gates[:, k:k+1, None]  # [batch_size, 1, 1]
            
            # Process through expert (using shared expert)
            expert_out = self.shared_expert(inp)  # [batch_size, seq_len, d_model]
            expert_out = expert_out * gate_value
            expert_outputs.append(expert_out)
        
        # Combine expert outputs
        output = sum(expert_outputs)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MERA(nn.Module):
    """
    MERA: Mixture of Experts with Routing Attention
    
    A Transformer-based model with Mixture of Experts (MoE) layer for stock prediction.
    
    Args:
        configs: Configuration namespace containing:
            - enc_in: Input feature dimension
            - d_model: Model hidden dimension
            - e_layers: Number of transformer encoder layers
            - n_heads: Number of attention heads
            - dropout: Dropout rate
            - c_out: Output dimension
            - num_expert: Number of experts in MoE layer (default: 8)
            - top_k: Top-k experts to route to (default: 4)
            - gate_dim: Gate embedding dimension (default: 16)
            - noise_level: Noise level for initialization (default: 0.0)
            - tau_hat_init: Initial tau value (default: 0.0)
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # Model dimensions
        self.input_size = configs.enc_in
        self.hidden_size = configs.d_model
        self.num_layers = getattr(configs, 'e_layers', 2)
        self.num_heads = getattr(configs, 'n_heads', 4)
        self.dropout = configs.dropout
        self.output_size = getattr(configs, 'c_out', 1)
        
        # MoE parameters
        self.num_expert = getattr(configs, 'num_expert', 8)
        self.top_k = getattr(configs, 'top_k', 4)
        self.gate_dim = getattr(configs, 'gate_dim', 16)
        self.noise_level = getattr(configs, 'noise_level', 0.0)
        
        # Learnable parameter for loss weighting (similar to LSTM)
        self.alpha = nn.Parameter(
            torch.tensor(float(getattr(configs, 'tau_hat_init', 0.0)), dtype=torch.float32)
        )
        
        # Input projection
        self.input_proj = nn.Linear(self.input_size, self.hidden_size)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(self.input_size)
        
        # Positional encoding
        self.pe = PositionalEncoding(self.hidden_size, self.dropout)
        
        # Transformer encoder layers
        encoder_layers = []
        for _ in range(self.num_layers):
            encoder_layers.append(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=self.num_heads,
                    dim_feedforward=self.hidden_size * 4,
                    dropout=self.dropout,
                    batch_first=False,
                    norm_first=True
                )
            )
        self.encoder = nn.ModuleList(encoder_layers)
        
        # MoE layer
        self.moe = MoELayer(
            d_model=self.hidden_size,
            d_gate=self.gate_dim,
            num_expert=self.num_expert,
            top_k=self.top_k,
            dropout=self.dropout,
            noise_std=1.0
        )
        
        # Gate input embedding (used for routing)
        # Maps last hidden state to gate dimension
        self.gate_proj = nn.Linear(self.hidden_size, self.gate_dim)
        
        # Output projection
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.fc2 = nn.Linear(self.hidden_size // 2, self.output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass of MERA model.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_features]
        
        Returns:
            output: Prediction tensor of shape [batch_size, c_out]
        """
        batch_size, seq_len, num_features = x.shape
        
        # Apply batch normalization
        x = x.reshape(-1, self.input_size)
        x = self.bn(x)
        x = x.reshape(batch_size, seq_len, num_features)
        
        # Permute to [seq_len, batch_size, features] for transformer
        x = x.permute(1, 0, 2).contiguous()
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pe(x)
        
        # Pass through transformer encoder layers
        for layer in self.encoder:
            x = layer(x)
        
        # x shape: [seq_len, batch_size, hidden_size]
        # Get last time step for gate input
        last_hidden = x[-1]  # [batch_size, hidden_size]
        
        # Project to gate dimension
        gate_inp = self.gate_proj(last_hidden)  # [batch_size, gate_dim]
        
        # Pass through MoE layer
        # Permute back to [batch_size, seq_len, hidden_size]
        x = x.permute(1, 0, 2).contiguous()
        moe_out = self.moe(x, gate_inp)  # [batch_size, seq_len, hidden_size]
        
        # Take last time step
        out = moe_out[:, -1, :]  # [batch_size, hidden_size]
        
        # Output projection
        out = self.relu(self.fc1(out))
        out = self.fc2(out)  # [batch_size, c_out]
        
        return out
    
    def collect_gate_loss(self, weight=0.01):
        """Collect load balancing loss from gate."""
        loss = 0
        if self.moe.gate.has_loss:
            loss = self.moe.gate.get_loss() * weight
        return loss
