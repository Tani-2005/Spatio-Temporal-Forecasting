import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """
    Graph Attention Layer (GAT).
    Learns dynamic connection weights between regions instead of relying purely on physical distance.
    """
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        # The attention mechanism vector 'a'
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        # 1. Apply linear transformation to all nodes
        h = self.W(x) 
        batch_size, num_nodes, features = h.shape
        
        # 2. Prepare combinations of node features to calculate attention
        h_repeated_in = h.repeat_interleave(num_nodes, dim=1)
        h_repeated_out = h.repeat(1, num_nodes, 1)
        
        # Concatenate features of node i and node j: [W*h_i || W*h_j]
        concat_h = torch.cat([h_repeated_in, h_repeated_out], dim=-1)
        concat_h = concat_h.view(batch_size, num_nodes, num_nodes, 2 * features)
        
        # 3. Calculate Attention Scores (e_ij)
        e = self.leakyrelu(self.a(concat_h).squeeze(-1))
        
        # 4. Mask out disconnected nodes using the original adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e) # Extremely small number for Softmax
        attention = torch.where(adj > 0, e, zero_vec)
        
        # 5. Normalize with Softmax to get final attention weights (alpha)
        attention_weights = F.softmax(attention, dim=-1)
        
        # 6. Output the attention-weighted features
        h_prime = torch.matmul(attention_weights, h)
        return F.elu(h_prime)

class STGAT(nn.Module):
    """
    Upgraded Spatio-Temporal Model combining GAT and LSTM.
    (Replace the STGCN class in your model.py with this)
    """
    def __init__(self, num_nodes, num_features, gat_hidden_dim, lstm_hidden_dim, pred_len):
        super(STGAT, self).__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        
        # Replaced the GCN with our new GAT layer
        self.gat = GATLayer(in_features=num_features, out_features=gat_hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=gat_hidden_dim * num_nodes, 
            hidden_size=lstm_hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        self.fc = nn.Linear(lstm_hidden_dim, num_nodes * pred_len)

    def forward(self, x, adj):
        batch_size, seq_len, num_nodes, num_features = x.shape
        gat_outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :, :] 
            gat_out = self.gat(x_t, adj) # Pass through Graph Attention Layer
            gat_out = gat_out.view(batch_size, -1) 
            gat_outputs.append(gat_out)
            
        lstm_input = torch.stack(gat_outputs, dim=1)
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        
        predictions = self.fc(hidden[-1])
        return predictions.view(batch_size, self.pred_len, self.num_nodes)