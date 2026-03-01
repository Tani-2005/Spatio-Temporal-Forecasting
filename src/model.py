import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """
    A custom Graph Convolutional Network Layer.
    Formula: H^{(l+1)} = ReLU(A * H^{(l)} * W)
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # W: The learnable weights of the GCN
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Initialize weights using Xavier initialization (best practice for deep learning)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        """
        Args:
            x (Tensor): Node features. Shape: (Batch, Nodes, Features)
            adj (Tensor): Adjacency Matrix. Shape: (Nodes, Nodes)
        """
        # 1. Feature Transformation: H * W
        support = torch.matmul(x, self.weight) 
        
        # 2. Spatial Aggregation: A * (H * W)
        # We multiply by the adjacency matrix to blend a city's data with its neighbors
        output = torch.matmul(adj, support)    
        
        return F.relu(output)

class STGCN(nn.Module):
    """
    The full Spatio-Temporal Model combining GCN and LSTM.
    """
    def __init__(self, num_nodes, num_features, gcn_hidden_dim, lstm_hidden_dim, pred_len):
        super(STGCN, self).__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        
        # Spatial Component: Extracts features from neighboring cities
        self.gcn = GCNLayer(in_features=num_features, out_features=gcn_hidden_dim)
        
        # Temporal Component: Learns patterns over time
        # Setting batch_first=True makes our tensor shapes easier to manage (Batch, Seq, Features)
        self.lstm = nn.LSTM(
            input_size=gcn_hidden_dim * num_nodes, 
            hidden_size=lstm_hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # Output Layer: Maps the LSTM's hidden state to our future predictions
        self.fc = nn.Linear(lstm_hidden_dim, num_nodes * pred_len)

    def forward(self, x, adj):
        """
        Args:
            x (Tensor): Input sequence. Shape: (Batch, Seq_Len, Nodes, Features)
            adj (Tensor): Adjacency Matrix. Shape: (Nodes, Nodes)
        """
        batch_size, seq_len, num_nodes, num_features = x.shape
        
        # We need to apply the GCN to every single week (time step) in our window
        gcn_outputs = []
        for t in range(seq_len):
            # Extract data for week 't'
            x_t = x[:, t, :, :] 
            
            # Pass through GCN
            gcn_out = self.gcn(x_t, adj) 
            
            # Flatten the spatial dimension so the LSTM can process it
            gcn_out = gcn_out.view(batch_size, -1) 
            gcn_outputs.append(gcn_out)
            
        # Stack the spatial outputs back into a time sequence
        # Shape becomes: (Batch, Seq_Len, GCN_Hidden_Dim * Num_Nodes)
        lstm_input = torch.stack(gcn_outputs, dim=1)
        
        # Pass the sequence through the LSTM
        lstm_out, (hidden, cell) = self.lstm(lstm_input)
        
        # We only care about the very last hidden state to make our future prediction
        last_hidden_state = hidden[-1] 
        
        # Generate the future predictions
        predictions = self.fc(last_hidden_state)
        
        # Reshape to our target format: (Batch, Future_Weeks, Nodes)
        return predictions.view(batch_size, self.pred_len, self.num_nodes)