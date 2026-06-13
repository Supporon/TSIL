"""
LSR-IGRU: Stock Trend Prediction Based on Long Short-Term Relationships and Improved GRU

Based on the paper: https://arxiv.org/abs/2409.08282
This model enhances GRU by incorporating long short-term relationship matrices
among stocks for better stock trend prediction.

Key components:
1. Long-term relationship matrix (based on industry/sector information)
2. Short-term relationship matrix (based on recent price dynamics)
3. Improved GRU (IGRU) that integrates relationship information into inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelationshipModule(nn.Module):
    """
    Module to compute long-short term relationship matrix.
    - Long-term: Based on feature similarity (can represent industry relationships)
    - Short-term: Based on recent price dynamics (overnight returns)
    """
    
    def __init__(self, input_dim, hidden_dim, num_heads=4, dropout=0.1):
        super(RelationshipModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Long-term relationship projection
        self.long_term_query = nn.Linear(input_dim, hidden_dim)
        self.long_term_key = nn.Linear(input_dim, hidden_dim)
        
        # Short-term relationship projection (using recent dynamics)
        self.short_term_query = nn.Linear(input_dim, hidden_dim)
        self.short_term_key = nn.Linear(input_dim, hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.scale = math.sqrt(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def compute_long_term_relation(self, x):
        """
        Compute long-term relationship based on overall feature similarity.
        Input: x [batch_size, seq_len, input_dim]
        Output: relation matrix [batch_size, batch_size]
        """
        # Use mean of sequence as representation
        x_mean = x.mean(dim=1)  # [batch_size, input_dim]
        
        query = self.long_term_query(x_mean)  # [batch_size, hidden_dim]
        key = self.long_term_key(x_mean)  # [batch_size, hidden_dim]
        
        # Compute attention scores as relationship
        attention = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        relation = F.softmax(attention, dim=-1)
        
        return relation
    
    def compute_short_term_relation(self, x):
        """
        Compute short-term relationship based on recent dynamics.
        Input: x [batch_size, seq_len, input_dim]
        Output: relation matrix [batch_size, batch_size]
        """
        # Use last few timesteps for short-term dynamics
        x_recent = x[:, -5:, :].mean(dim=1)  # [batch_size, input_dim]
        
        query = self.short_term_query(x_recent)
        key = self.short_term_key(x_recent)
        
        attention = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        relation = F.softmax(attention, dim=-1)
        
        return relation
    
    def forward(self, x):
        """
        Compute combined long-short term relationship.
        Returns enhanced features incorporating relationship information.
        """
        long_rel = self.compute_long_term_relation(x)  # [batch_size, batch_size]
        short_rel = self.compute_short_term_relation(x)  # [batch_size, batch_size]
        
        # Combine relationships
        combined_rel = (long_rel + short_rel) / 2
        
        return combined_rel, long_rel, short_rel


class ImprovedGRUCell(nn.Module):
    """
    Improved GRU Cell that incorporates relationship information.
    The key improvement is enhancing inputs with relationship-aware features.
    """
    
    def __init__(self, input_dim, hidden_dim):
        super(ImprovedGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Standard GRU gates
        self.Wz = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.Wr = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.Wh = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Relationship-aware input transformation
        self.relation_transform = nn.Linear(input_dim, input_dim)
        
    def forward(self, x, h_prev, relation_weight=None):
        """
        x: [batch_size, input_dim]
        h_prev: [batch_size, hidden_dim]
        relation_weight: [batch_size, batch_size] - relationship matrix
        """
        # Enhance input with relationship information
        if relation_weight is not None:
            # Apply relationship-weighted aggregation
            x_enhanced = torch.matmul(relation_weight, x)
            x = x + self.relation_transform(x_enhanced)
        
        combined = torch.cat([x, h_prev], dim=-1)
        
        # Update gate
        z = torch.sigmoid(self.Wz(combined))
        # Reset gate  
        r = torch.sigmoid(self.Wr(combined))
        # Candidate hidden state
        combined_reset = torch.cat([x, r * h_prev], dim=-1)
        h_tilde = torch.tanh(self.Wh(combined_reset))
        # Final hidden state
        h = (1 - z) * h_prev + z * h_tilde
        
        return h


class IGRU(nn.Module):
    """
    Improved GRU layer that processes sequences with relationship awareness.
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(IGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(ImprovedGRUCell(cell_input_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, relation_matrix=None):
        """
        x: [batch_size, seq_len, input_dim]
        relation_matrix: [batch_size, batch_size]
        """
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        outputs = []
        
        # Initialize hidden states
        h = [torch.zeros(batch_size, self.hidden_dim, device=device) 
             for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx] = cell(x_t, h[layer_idx], relation_matrix)
                x_t = self.dropout(h[layer_idx])
            
            outputs.append(h[-1])
        
        outputs = torch.stack(outputs, dim=1)  # [batch_size, seq_len, hidden_dim]
        
        return outputs, h[-1]


class LSR_IGRU(nn.Module):
    """
    LSR-IGRU: Long Short-Term Relationship Improved GRU Model
    
    This model combines:
    1. Relationship Module: Computes long-short term relationship matrix
    2. IGRU: Improved GRU that incorporates relationship information
    3. Prediction Head: Outputs stock trend prediction
    """
    
    def __init__(self, configs):
        super(LSR_IGRU, self).__init__()
        
        self.input_size = configs.enc_in
        self.output_size = configs.c_out
        self.hidden_size = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout
        self.seq_len = configs.seq_len
        
        # Learnable parameter for weighted loss (if using MSE_with_weak)
        self.alpha = torch.nn.Parameter(
            torch.tensor(float(configs.tau_hat_init), dtype=torch.float32)
        )
        
        # Input processing
        self.input_drop = nn.Dropout(self.dropout)
        self.input_proj = nn.Linear(self.input_size, self.hidden_size)
        
        # Relationship module
        self.relationship_module = RelationshipModule(
            input_dim=self.input_size,
            hidden_dim=self.hidden_size,
            num_heads=4,
            dropout=self.dropout
        )
        
        # Improved GRU
        self.igru = IGRU(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # Attention for sequence aggregation
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1, bias=False)
        )
        
        # Prediction head
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size * 2, configs.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(configs.d_ff, configs.c_out, bias=True),
        )
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, input_size]
        """
        batch_size, seq_len, _ = x.size()
        
        # Input dropout
        x = self.input_drop(x)
        
        # Compute relationship matrix
        relation_matrix, long_rel, short_rel = self.relationship_module(x)
        
        # Project input
        x_proj = self.input_proj(x)  # [batch_size, seq_len, hidden_size]
        
        # Process through Improved GRU with relationship awareness
        igru_out, final_hidden = self.igru(x_proj, relation_matrix)
        # igru_out: [batch_size, seq_len, hidden_size]
        
        # Attention-weighted aggregation
        attn_scores = self.attention(igru_out)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_scores, dim=1)
        context = (igru_out * attn_weights).sum(dim=1)  # [batch_size, hidden_size]
        
        # Combine with last hidden state
        combined = torch.cat([context, final_hidden], dim=-1)
        
        # Prediction
        output = self.projection(combined)
        
        return output


class LSRIGRU(LSR_IGRU):
    """Alias for LSR_IGRU to match common naming conventions."""
    pass
