import networkx as nx
import torch

class DynamicGraphUpdater:
    """Handles temporal graph updates"""
    def __init__(self, update_method='tgcn', hidden_dim=64):
        self.update_method = update_method
        self.hidden_dim = hidden_dim
        self.current_graph = None
        
    def update_graph(self, model, prev_graph, new_data):
        """Update graph with new data"""
        if self.current_graph is None:
            self.current_graph = prev_graph
            
        # Get node embeddings
        with torch.no_grad():
            embeddings = model.get_embeddings(self.current_graph)
            
        # Apply temporal update
        updated_embeddings = self.temporal_update(embeddings)
        
        # Update node features
        nx.set_node_attributes(self.current_graph, 
                              {n: {'embedding': e} 
                               for n, e in zip(self.current_graph.nodes(), updated_embeddings)})
        
        return self.current_graph
    
    def temporal_update(self, embeddings):
        """Apply temporal update rule"""
        # Simple RNN-based update for demonstration
        if not hasattr(self, 'rnn'):
            self.rnn = torch.nn.GRUCell(self.hidden_dim, self.hidden_dim)
            
        return self.rnn(embeddings)