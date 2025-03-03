import torch
import torch.nn as nn
import torch.optim as optim

class GNNTrainer:
    """GNN Training Module"""
    def __init__(self, epochs=100, lr=0.01, patience=5):
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        
    def train(self, model, graph):
        """Train GNN model"""
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        best_loss = float('inf')
        no_improve = 0
        
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(graph)
            loss = self.compute_loss(outputs, graph)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= self.patience:
                break
                
        return model
    
    def compute_loss(self, outputs, graph):
        # not currently implemented still have to think about the strategies to evaluate the loss.
        pass