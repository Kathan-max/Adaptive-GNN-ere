import networkx as nx
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MultiViewGraphBuilder:    
    def __init__(self, 
                 use_semantic_view: bool = True,
                 use_structural_view: bool = True,
                 use_temporal_view: bool = True,
                 use_meta_graph_view: bool = True):
        self.use_semantic_view = use_semantic_view
        self.use_structural_view = use_structural_view
        self.use_temporal_view = use_temporal_view
        self.use_meta_graph_view = use_meta_graph_view
        
        # Initialize TF-IDF vectorizer for semantic similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=2, max_df=0.95, stop_words='english', 
            ngram_range=(1, 2), max_features=10000
        )
    
    def build_multi_view_graphs(self, 
                               base_graph: nx.DiGraph, 
                               document_texts: Optional[Dict[str, str]] = None,
                               timestamps: Optional[Dict[str, str]] = None) -> Dict[str, nx.DiGraph]:
        
        views = {}
        
        if self.use_semantic_view:
            views['semantic'] = self.build_semantic_view(base_graph)
            
        if self.use_structural_view:
            views['structural'] = self.build_structural_view(base_graph)
            
        if self.use_temporal_view and timestamps:
            views['temporal'] = self.build_temporal_view(base_graph, timestamps)
            
        if self.use_meta_graph_view:
            views['meta_graph'] = self.build_meta_graph_view(base_graph)
            
        return views
    
    def build_semantic_view(self, graph: nx.DiGraph) -> nx.DiGraph:
        semantic_graph = nx.DiGraph()
        
        # Copy nodes from the original graph
        for node, data in graph.nodes(data=True):
            semantic_graph.add_node(node, **data)
        
        # Extract text content from all nodes
        node_texts = {}
        for node, data in graph.nodes(data=True):
            if 'text' in data:
                node_texts[node] = data['text']
        
        if not node_texts:
            print("Warning: No text content found in nodes for semantic view")
            return semantic_graph
        
        # Compute semantic similarity between nodes using TF-IDF and cosine similarity
        nodes = list(node_texts.keys())
        texts = [node_texts[node] for node in nodes]
        
        # Fit and transform texts to TF-IDF vectors
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            # Compute pairwise cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Add edges for similar nodes (above threshold)
            similarity_threshold = 0.3
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    if similarity_matrix[i, j] > similarity_threshold:
                        semantic_graph.add_edge(
                            nodes[i], nodes[j], 
                            weight=float(similarity_matrix[i, j]),
                            type="semantic_similarity"
                        )
                        semantic_graph.add_edge(
                            nodes[j], nodes[i], 
                            weight=float(similarity_matrix[i, j]),
                            type="semantic_similarity"
                        )
        except Exception as e:
            print(f"Error computing semantic similarity: {str(e)}")
        
        print(f"Built semantic view with {semantic_graph.number_of_nodes()} nodes and {semantic_graph.number_of_edges()} edges")
        return semantic_graph
    
    def build_structural_view(self, graph: nx.DiGraph) -> nx.DiGraph:
        
        structural_graph = nx.DiGraph()
        
        # Copy nodes from the original graph
        for node, data in graph.nodes(data=True):
            structural_graph.add_node(node, **data)
        
        # Copy existing edges of certain types that represent structural relationships
        structural_relation_types = {"part_of", "contains", "is_a", "located_in"}
        
        for source, target, data in graph.edges(data=True):
            if 'type' in data and data['type'] in structural_relation_types:
                structural_graph.add_edge(source, target, **data)
        
        # Add hierarchical relationships based on entity types
        # For example, sections contain paragraphs which contain sentences
        hierarchy = {
            "SECTION": "PARAGRAPH",
            "PARAGRAPH": "SENTENCE",
            "DOCUMENT": "SECTION",
            "CHAPTER": "SECTION"
        }
        
        # Group nodes by their entity type
        nodes_by_type = {}
        for node, data in graph.nodes(data=True):
            entity_type = data.get('type', '')
            if entity_type not in nodes_by_type:
                nodes_by_type[entity_type] = []
            nodes_by_type[entity_type].append((node, data))
        
        # Add hierarchical edges
        for parent_type, child_type in hierarchy.items():
            if parent_type in nodes_by_type and child_type in nodes_by_type:
                # Add edges from all parent nodes to child nodes within a reasonable distance
                for parent_node, parent_data in nodes_by_type[parent_type]:
                    parent_start = parent_data.get('start', 0)
                    parent_end = parent_data.get('end', float('inf'))
                    
                    for child_node, child_data in nodes_by_type[child_type]:
                        child_start = child_data.get('start', 0)
                        child_end = child_data.get('end', float('inf'))
                        
                        # Check if child is contained within parent
                        if parent_start <= child_start and child_end <= parent_end:
                            structural_graph.add_edge(
                                parent_node, child_node,
                                type="contains",
                                weight=1.0
                            )
        
        print(f"Built structural view with {structural_graph.number_of_nodes()} nodes and {structural_graph.number_of_edges()} edges")
        return structural_graph
    
    def build_temporal_view(self, graph: nx.DiGraph, timestamps: Dict[str, str]) -> nx.DiGraph:
        temporal_graph = nx.DiGraph()
        
        # Copy nodes from the original graph and add timestamp attribute
        for node, data in graph.nodes(data=True):
            node_data = data.copy()
            
            # Add timestamp if available
            if node in timestamps:
                node_data['timestamp'] = timestamps[node]
            
            temporal_graph.add_node(node, **node_data)
        
        # Copy edges from the original graph
        for source, target, data in graph.edges(data=True):
            temporal_graph.add_edge(source, target, **data)
        
        
        # Group nodes by entity text
        nodes_by_text = {}
        for node, data in graph.nodes(data=True):
            entity_text = data.get('text', '').lower()
            if entity_text:
                if entity_text not in nodes_by_text:
                    nodes_by_text[entity_text] = []
                # Store node, data, and timestamp (if available)
                timestamp = timestamps.get(node, "")
                nodes_by_text[entity_text].append((node, data, timestamp))
        
        # Link temporally related nodes
        for entity_text, node_list in nodes_by_text.items():
            if len(node_list) > 1:
                # Sort by timestamp if available
                sorted_nodes = sorted(node_list, key=lambda x: x[2] if x[2] else "")
                
                # Add temporal edges
                for i in range(len(sorted_nodes) - 1):
                    current_node = sorted_nodes[i][0]
                    next_node = sorted_nodes[i+1][0]
                    
                    temporal_graph.add_edge(
                        current_node, next_node,
                        type="temporal_next",
                        weight=1.0
                    )
                    temporal_graph.add_edge(
                        next_node, current_node,
                        type="temporal_previous",
                        weight=1.0
                    )
        
        print(f"Built temporal view with {temporal_graph.number_of_nodes()} nodes and {temporal_graph.number_of_edges()} edges")
        return temporal_graph
    
    def build_meta_graph_view(self, graph: nx.DiGraph) -> nx.DiGraph:
        meta_graph = nx.DiGraph()
        
        # Extract entity types
        entity_types = set()
        for _, data in graph.nodes(data=True):
            if 'type' in data:
                entity_types.add(data['type'])
        
        # Extract relation types
        relation_types = set()
        for _, _, data in graph.edges(data=True):
            if 'type' in data:
                relation_types.add(data['type'])
        
        # Create nodes for entity types
        for entity_type in entity_types:
            meta_graph.add_node(f"ENTITY_{entity_type}", 
                               type="entity_type",
                               name=entity_type)
        
        # Create nodes for relation types
        for relation_type in relation_types:
            meta_graph.add_node(f"RELATION_{relation_type}", 
                               type="relation_type",
                               name=relation_type)
        
        # Connect entity types based on their relations in the original graph
        for source, target, data in graph.edges(data=True):
            if 'type' in data and data['type'] in relation_types:
                source_type = graph.nodes[source].get('type', 'UNKNOWN')
                target_type = graph.nodes[target].get('type', 'UNKNOWN')
                relation_type = data['type']
                
                if source_type in entity_types and target_type in entity_types:
                    # Add edge from source entity type to relation type
                    meta_graph.add_edge(
                        f"ENTITY_{source_type}",
                        f"RELATION_{relation_type}",
                        type="source_of",
                        weight=1.0
                    )
                    
                    # Add edge from relation type to target entity type
                    meta_graph.add_edge(
                        f"RELATION_{relation_type}",
                        f"ENTITY_{target_type}",
                        type="target_of",
                        weight=1.0
                    )
        
        print(f"Built meta-graph view with {meta_graph.number_of_nodes()} nodes and {meta_graph.number_of_edges()} edges")
        return meta_graph
    
    def combine_views(self, views: Dict[str, nx.DiGraph]) -> nx.MultiDiGraph:
        multi_view_graph = nx.MultiDiGraph()
        
        # Add all nodes from all views
        for view_name, view_graph in views.items():
            for node, data in view_graph.nodes(data=True):
                if not multi_view_graph.has_node(node):
                    multi_view_graph.add_node(node, **data)
        
        # Add all edges from all views with view information
        for view_name, view_graph in views.items():
            for source, target, data in view_graph.edges(data=True):
                edge_data = data.copy()
                edge_data['view'] = view_name
                multi_view_graph.add_edge(source, target, **edge_data)
        
        print(f"Combined multi-view graph has {multi_view_graph.number_of_nodes()} nodes and {multi_view_graph.number_of_edges()} edges")
        return multi_view_graph