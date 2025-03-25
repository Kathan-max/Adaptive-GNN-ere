import json

with open("E:/Windsor/Sem-2/Topics In Applied AI-GNN/Project/Code/data/Datasets/ade_full.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    
print(len(data))
print(data[0])
print('-'*10)
print(data[1])