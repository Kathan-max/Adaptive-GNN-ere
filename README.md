# Adaptive MVGNAS Framework  
**Generalized Entity and Relation Extraction using Adaptive Graph Neural Networks**  

---

## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Installation](#installation)  
4. [Quick Start](#quick-start)  
5. [Configuration](#configuration)  
6. [Contributing](#contributing)  
---

## Project Overview  
The **Adaptive MVGNAS Framework** is a state-of-the-art system for generalized entity and relation extraction across diverse domains (e.g., finance, law, education). It combines multi-view graph representation learning with neural architecture search (NAS) and temporal adaptation to construct dynamic knowledge graphs from unstructured documents.  

The system processes input documents (PDFs), extracts entities and relations, constructs multi-view graphs, and adapts to new domains using few-shot learning. It is designed for scalability, flexibility, and high performance in real-world applications.  

---

## Key Features  
- **Hybrid NLP Pipeline**: Combines spaCy and transformer models for robust entity and relation extraction.  
- **Multi-View Graph Construction**: Builds semantic, structural, temporal, and meta-graph views for comprehensive representation.  
- **Neural Architecture Search (NAS)**: Automatically discovers optimal GNN architectures for each domain/task.  
- **Few-Shot Learning**: Handles new entity/relation types with minimal labeled examples.  
- **Temporal Adaptation**: Updates knowledge graphs dynamically using GRU-based mechanisms.  
- **Modular Design**: Easy to extend with new components or domain-specific configurations.  

---

## Installation  

### Prerequisites  
- Python 3.8 or later  
- pip3 and virtualenv  
- CUDA (optional, for GPU acceleration)  

### Steps  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/adaptive-gnn-ere/proposed-solution.git  
   cd proposed-solution  
2. Run the Installation Script:
    ```bash
    chmod +x automateInstallation.sh
    ./automateInstallation.sh
    or (with command line args for main.py)
    ./install_dependencies.sh --input ./documents --config finance_config.yaml
3. Verify the installation:
The script will automatically verify all components.
If any issues occur, check the error messages and ensure all prerequisites are met.

## Quick Start
1. Place your input documents (PDFs) in the data/ directory.

2. Update the configuration file (config/default_config.yaml) if needed.

3. run:
python3 main.py --input ./data --config finance_config.yaml

4. Outputs:
Knowledge graphs are saved in the output/ directory
logs are stored in logs/

## Configuration  
The system is highly configurable through YAML files. Key configuration options include:  

- **Document Processing**:  
  - NER models (spaCy, transformers)  
  - Relation extraction thresholds  

- **Multi-View Graphs**:  
  - Semantic similarity thresholds  
  - Document hierarchy rules  
  - Temporal update intervals  

- **NAS**:  
  - Search space (GNN layers, attention mechanisms)  
  - Controller parameters  

- **Few-Shot Learning**:  
  - Prototype dimensions  
  - Similarity thresholds  

Example domain-specific configurations are provided in `config/domain_configs/`.  

---

## Contributing  
We welcome contributions! Here's how you can help:  

### Code Contributions  
1. Fork the repository.  
2. Create a new branch for your feature/bugfix:
   ```bash
   git checkout -b feature/your-feature-name  
3. Commit your changes:
    ```bash
    git commit -m "Add your feature description"  
4. Push to your branch:
    ```bash
    git push origin feature/your-feature-name
5. Open a pull request with a detailed description of your changes.

#### Coding Guidelines
- Follow PEP 8 style guidelines
- Document new functionality in the relevant Readme section

## Contact
For questions or collaborations, please contact:

- Kathan Piyushkumar Bhavsar: bhavsa85@uwindsor.ca
- Ashiqur Rahman: rahman6s@uwindsor.ca
- Foysal Rahman Nitu: nituf@uwindsor.ca

Thank you for using Adaptive MVGNAS!
We hope this framework helps you build powerful knowledge graphs across diverse domains. Let us know how it works for your use case!