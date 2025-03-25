import json

with open("../../Datasets/biorelex_train_full.json") as f:
    data_train = json.load(f)

print(len(data_train))
#print(data_train[0])

print(json.dumps(data_train[0], indent=4))