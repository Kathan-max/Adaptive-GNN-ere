import requests

url = "https://lavis.cs.hs-rm.de/storage/spert/public/datasets/ade/ade_full.json"
response = requests.get(url)

with open("E:/Windsor/Sem-2/Topics In Applied AI-GNN/Project/Code/data/Datasets/ade_full.json", "wb") as file:
    file.write(response.content)

print("Download complete!")