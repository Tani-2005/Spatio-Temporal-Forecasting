import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def train_stgcn(model, train_loader, val_loader, adj_matrix, epochs=50, learning_rate=0.001):
    """
    Trains the Spatio-Temporal Graph Convolutional Network.
    
    Args:
        model (nn.Module): The STGCN model.
        train_loader (DataLoader): PyTorch DataLoader for training data.
        val_loader (DataLoader): PyTorch DataLoader for validation data.
        adj_matrix (Tensor): The spatial adjacency matrix.
        epochs (int): Number of times to loop through the entire dataset.
        learning_rate (float): Step size for the optimizer.
    """
    # 1. Setup Loss Function and Optimizer
    # Huber Loss is less sensitive to massive anomaly spikes (outbreaks) than MSE
    criterion = nn.HuberLoss(delta=1.0) 
    
    # AdamW incorporates weight decay natively, which prevents overfitting
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Convert adjacency matrix to PyTorch tensor if it isn't already
    if not isinstance(adj_matrix, torch.Tensor):
        adj_matrix = torch.FloatTensor(adj_matrix)
        
    print("--- Starting Model Training ---")
    
    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train() # Set model to training mode
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            # Step 1: Zero the gradients so they don't accumulate from previous batches
            optimizer.zero_grad()
            
            # Step 2: Forward pass (Generate predictions)
            predictions = model(batch_x, adj_matrix)
            
            # Step 3: Calculate the error
            loss = criterion(predictions, batch_y)
            
            # Step 4: Backpropagation (Calculate gradients)
            loss.backward()
            
            # Step 5: Update the weights
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # --- VALIDATION PHASE ---
        model.eval() # Set model to evaluation mode (turns off dropout, etc.)
        val_loss = 0.0
        
        with torch.no_grad(): # Disable gradient tracking to save memory/compute
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x, adj_matrix)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("--- Training Complete ---")
    return model