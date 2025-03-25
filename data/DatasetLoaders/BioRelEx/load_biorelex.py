import requests

url_train = "https://github.com/YerevaNN/BioRelEx/releases/download/1.0alpha7/1.0alpha7.train.json"
url_test = "https://github.com/YerevaNN/BioRelEx/releases/download/1.0alpha7/1.0alpha7.dev.json"

response_train = requests.get(url_train)
response_test = requests.get(url_test)

with open("E:/Windsor/Sem-2/Topics In Applied AI-GNN/Project/Code/data/Datasets/biorelex_train_full.json", "wb") as file:
    file.write(response_train.content)

with open("E:/Windsor/Sem-2/Topics In Applied AI-GNN/Project/Code/data/Datasets/biorelex_test_full.json", "wb") as file:
    file.write(response_test.content)
    
print("Download complete!")