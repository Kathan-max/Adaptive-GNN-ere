import torch
import torch.nn as nn

class PrototypicalNetworks(nn.Module):
    """Few-shot learning with prototypical networks"""
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim))
        
    def compute_prototypes(self, support_set):
        """Compute class prototypes"""
        return torch.mean(support_set, dim=1)
        
    def forward(self, support, query):
        # support: [n_classes, n_support, features]
        # query: [n_query, features]
        
        prototypes = self.compute_prototypes(support)
        distances = torch.cdist(query, prototypes)
        return -distances
    
    def adapt_model(self, model, support_set, graph):
        """Adapt model using few-shot examples"""
        # Extract features for support set
        with torch.no_grad():
            features = model.get_embeddings(graph)
            
        # Update model parameters based on prototypes
        # Implementation details would depend on model architecture
        return model