import os
import re
import PyPDF2
import spacy
import torch
import networkx as nx
from typing import List, Dict, Tuple, Set, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification

class DocumentProcessor:
    
    def __init__(self, spacy_model: str = "en_core_web_lg", 
                 transformer_model: str = "Jean-Baptiste/camembert-ner-with-dates",
                 use_spacy: bool = True,
                 use_transformers: bool = True,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):

        self.device = device
        self.use_spacy = use_spacy
        self.use_transformers = use_transformers
        
        # Initialize spaCy
        if use_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
                print(f"Loaded spaCy model: {spacy_model}")
            except OSError:
                print(f"SpaCy model {spacy_model} not found. Downloading...")
                spacy.cli.download(spacy_model)
                self.nlp = spacy.load(spacy_model)
        
        # Initialize transformers
        if use_transformers:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
            self.model = AutoModelForTokenClassification.from_pretrained(transformer_model)
            self.model.to(device)
            print(f"Loaded transformer model: {transformer_model} on {device}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n"
                    
            # Basic cleaning
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def extract_entities_spacy(self, text: str) -> List[Dict]:

        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'type': ent.label_,
                'confidence': 1.0,  # spaCy doesn't provide confidence scores
                'source': 'spacy'
            })
            
        return entities

    def extract_entities_transformers(self, text: str) -> List[Dict]:
        """
        Extract entities from text using transformers.
        
        Args:
            text: Input text
            
        Returns:
            List[Dict]: List of extracted entities with their metadata
        """
        # Break text into chunks to handle transformer token limits
        max_length = self.tokenizer.model_max_length - 10
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        all_entities = []
        offset = 0
        
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            predictions = torch.argmax(outputs.logits, dim=2)
            input_ids = inputs["input_ids"].cpu().numpy()[0]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            # Extract entities
            current_entity = {"text": "", "start": -1, "end": -1, "type": ""}
            
            for i, (token, prediction) in enumerate(zip(tokens, predictions[0].cpu().numpy())):
                if prediction == 0:  # 'O' tag
                    if current_entity["text"]:
                        current_entity["end"] = self.get_token_end_position(chunk, current_entity["text"], current_entity["start"])
                        current_entity["source"] = "transformers"
                        current_entity["confidence"] = 0.9  # Placeholder
                        all_entities.append(current_entity.copy())
                        current_entity = {"text": "", "start": -1, "end": -1, "type": ""}
                elif token.startswith("##"):
                    current_entity["text"] += token[2:]
                else:
                    # B tag (beginning of entity)
                    if prediction % 2 == 1:
                        if current_entity["text"]:
                            current_entity["end"] = self.get_token_end_position(chunk, current_entity["text"], current_entity["start"])
                            current_entity["source"] = "transformers"
                            current_entity["confidence"] = 0.9  # Placeholder
                            all_entities.append(current_entity.copy())
                        
                        entity_type = self.model.config.id2label[prediction]
                        if entity_type.startswith("B-"):
                            entity_type = entity_type[2:]
                        
                        token_start = self.get_token_start_position(chunk, token)
                        if token_start != -1:
                            current_entity = {
                                "text": token,
                                "start": token_start + offset,
                                "end": -1,
                                "type": entity_type
                            }
                    # I tag (inside entity)
                    else:
                        if current_entity["text"]:
                            current_entity["text"] += " " + token
            
            # Don't forget the last entity in the chunk
            if current_entity["text"]:
                current_entity["end"] = self.get_token_end_position(chunk, current_entity["text"], current_entity["start"])
                current_entity["source"] = "transformers"
                current_entity["confidence"] = 0.9  # Placeholder
                all_entities.append(current_entity.copy())
            
            offset += len(chunk)
            
        return all_entities

    def get_token_start_position(self, text: str, token: str) -> int:
        # Remove special tokens for matching
        clean_token = token.replace("Ġ", "")
        match = re.search(r'\b' + re.escape(clean_token) + r'\b', text)
        return match.start() if match else -1
    
    def get_token_end_position(self, text: str, entity_text: str, start_pos: int) -> int:
        if start_pos == -1:
            return -1
        # Look for the entity text starting from start_pos
        pattern = re.escape(entity_text)
        substring = text[start_pos:start_pos + len(entity_text) + 20]  # Add some margin
        match = re.search(pattern, substring)
        return start_pos + match.end() if match else start_pos + len(entity_text)

    def extract_entities(self, text: str) -> List[Dict]:
        entities = []
        
        if self.use_spacy:
            entities.extend(self.extract_entities_spacy(text))
            
        if self.use_transformers:
            entities.extend(self.extract_entities_transformers(text))
            
        # Merge overlapping entities, preferring those with higher confidence
        return self.merge_overlapping_entities(entities)
    
    def merge_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        if not entities:
            return []
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: (x['start'], -x['end']))
        merged = [sorted_entities[0]]
        
        for current in sorted_entities[1:]:
            previous = merged[-1]
            
            # Check for overlap
            if current['start'] <= previous['end']:
                # If current entity has higher confidence, replace previous
                if current['confidence'] > previous['confidence']:
                    merged[-1] = current
                # If equal confidence but current is longer, prefer current
                elif current['confidence'] == previous['confidence'] and (current['end'] - current['start']) > (previous['end'] - previous['start']):
                    merged[-1] = current
            else:
                merged.append(current)
                
        return merged
    
    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        # Simple relation extraction based on proximity
        relations = []
        
        # Sort entities by their position in text
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        # For each pair of entities that are close to each other, create a relation
        for i in range(len(sorted_entities) - 1):
            for j in range(i + 1, min(i + 5, len(sorted_entities))):  # Consider up to 5 subsequent entities
                entity1 = sorted_entities[i]
                entity2 = sorted_entities[j]
                
                # Check if entities are within a reasonable distance
                distance = entity2['start'] - entity1['end']
                if 0 <= distance <= 100:  # Within 100 characters
                    # Extract the text between the two entities
                    between_text = text[entity1['end']:entity2['start']]
                    
                    # Simple relation extraction based on keywords
                    relation_type = self.identify_relation_type(between_text)
                    
                    if relation_type:
                        relations.append({
                            'source': entity1,
                            'target': entity2,
                            'type': relation_type,
                            'text': between_text.strip(),
                            'confidence': 0.7  # Placeholder confidence
                        })
        
        return relations
    
    def identify_relation_type(self, text: str) -> Optional[str]:
        # Define some patterns for relation types
        relation_patterns = {
            'contains': r'contains|comprises|includes|has',
            'is_a': r'is a|type of|kind of|class of',
            'part_of': r'part of|component of|element of',
            'related_to': r'related to|associated with|connected to',
            'causes': r'causes|leads to|results in',
            'located_in': r'located in|found in|situated in',
            'temporal': r'before|after|during|when',
            'owner_of': r'owns|possesses|has possession of',
            'created_by': r'created by|developed by|authored by'
        }
        
        for relation_type, pattern in relation_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return relation_type
        
        # If no specific relation is found, check if there's any verb connecting the entities
        doc = self.nlp(text) if self.use_spacy else None
        if doc:
            for token in doc:
                if token.pos_ == "VERB":
                    return f"action:{token.lemma_}"
                
        # Default relation
        return "related_to" if text.strip() else None
    
    def build_initial_graph(self, entities: List[Dict], relations: List[Dict]) -> nx.DiGraph:
        G = nx.DiGraph()
        
        # Add entities as nodes
        for entity in entities:
            node_id = f"{entity['type']}_{entity['text']}_{entity['start']}"
            G.add_node(node_id, **entity)
        
        # Add relations as edges
        for relation in relations:
            source_id = f"{relation['source']['type']}_{relation['source']['text']}_{relation['source']['start']}"
            target_id = f"{relation['target']['type']}_{relation['target']['text']}_{relation['target']['start']}"
            
            G.add_edge(source_id, target_id, **{
                'type': relation['type'],
                'text': relation['text'],
                'confidence': relation['confidence']
            })
            
        return G
    
    def process_document(self, pdf_path: str) -> nx.DiGraph:
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print(f"Warning: No text extracted from {pdf_path}")
            return nx.DiGraph()
            
        # Extract entities
        entities = self.extract_entities(text)
        print(f"Extracted {len(entities)} entities from {pdf_path}")
        
        # Extract relations
        relations = self.extract_relations(text, entities)
        print(f"Extracted {len(relations)} relations from {pdf_path}")
        
        # Build initial graph
        graph = self.build_initial_graph(entities, relations)
        print(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        return graph
    
    def process_documents(self, pdf_paths: List[str]) -> nx.DiGraph:
        merged_graph = nx.DiGraph()
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Warning: File {pdf_path} does not exist")
                continue
                
            document_graph = self.process_document(pdf_path)
            merged_graph = nx.compose(merged_graph, document_graph)
        
        print(f"Final merged graph has {merged_graph.number_of_nodes()} nodes and {merged_graph.number_of_edges()} edges")
        return merged_graph