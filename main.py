import os
import argparse
import networkx as nx
import torch
import yaml
from datetime import datetime
from typing import Dict, List

# Import custom modules
from data.document_processor import DocumentProcessor
from data.multi_view_builder import MultiViewGraphBuilder
from nas.controller import NASController
from few_shot.prototypical import PrototypicalNetworks
from temporal.dynamic_graph import DynamicGraphUpdater
from training.trainer import GNNTrainer

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Adaptive MVGNAS for Knowledge Graph Construction")
    parser.add_argument("-i", "--input", required=True, help="Path to input PDF documents directory")
    parser.add_argument("-c", "--config", default="config/default_config.yaml", help="Path to config file")
    parser.add_argument("-o", "--output", default="output_graphs", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    config = load_config(args.config)
    
    doc_processor = DocumentProcessor(**config['document_processor'])
    view_builder = MultiViewGraphBuilder(**config['multi_view'])
    nas_controller = NASController(**config['nas'])
    graph_updater = DynamicGraphUpdater(**config['temporal'])
    few_shot_learner = PrototypicalNetworks(**config['few_shot'])
    
    # Process documents
    pdf_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".pdf")]
    base_graph = doc_processor.process_documents(pdf_paths)
    
    # Build multi-view graphs
    views = view_builder.build_multi_view_graphs(base_graph)
    multi_view_graph = view_builder.combine_views(views)
    
    # Neural Architecture Search
    print("\nStarting Neural Architecture Search...")
    best_architecture = nas_controller.search(multi_view_graph)
    print(f"Best architecture found: {best_architecture}")
    
    # Initialize and train final model
    trainer = GNNTrainer(**config['training'])
    model = nas_controller.build_model(best_architecture)
    trainer.train(model, multi_view_graph)
    
    # Few-shot learning if prior knowledge exists
    if config['few_shot'].get('prior_knowledge_path'):
        print("\nApplying few-shot learning...")
        support_set = few_shot_learner.load_support_set(
            config['few_shot']['prior_knowledge_path']
        )
        model = few_shot_learner.adapt_model(model, support_set, multi_view_graph)
    
    # Temporal adaptation
    if config['temporal']['enable']:
        print("\nProcessing temporal updates...")
        time_steps = graph_updater.get_time_steps(args.input)
        for t in time_steps:
            updated_graph = graph_updater.update_graph(
                model, multi_view_graph, t
            )
            nx.write_graphml(updated_graph, 
                            os.path.join(args.output, f"graph_{t}.graphml"))
    
    # Save final graph
    final_graph = graph_updater.current_graph if config['temporal']['enable'] else multi_view_graph
    nx.write_graphml(final_graph, os.path.join(args.output, "final_graph.graphml"))
    
    print("\nProcessing completed successfully!")

if __name__ == "__main__":
    main()
    