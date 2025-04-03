import json
import random

class FullTextViewer:
    def __init__(self, path):
        self.path = path
        
    def firstObjViewer(self):
        with open(self.path) as f:
            data = json.load(f)
        return json.dumps(data[f"chunk-{0}"], indent=4) 

    def randomReader(self):
        with open(self.path) as f:
            data = json.load(f)
        finalIdx = len(data) - 1 
        idx = random.randint(0, finalIdx)
        return json.dumps(data[f"chunk-{idx}"], indent=4)
    
    def totalChunks(self):
        with open(self.path) as f:
            data = json.load(f)
        return len(data)
    
    def totalInterChunks(self):
        with open(self.path) as f:
            data = json.load(f)
        dict_ = {}
        for key, value in data.items():
            if value['fileName'] not in dict_:
                dict_[value['fileName']] = 1
            else:  
                dict_[value['fileName']] += 1
        return dict_

    def dataTrimer(self, percent):
        with open(self.path) as f:
            data = json.load(f)
        finalIdx = len(data)
        finalCount = int((percent/100) * finalIdx)
        trimmedData = {}
        igData = {}
        for key, value in data.items():
            if finalCount == 0:
                break
            rn = random.randint(0, 100)
            if rn%2 == 0:
                trimmedData[key] = value
                finalCount -= 1
            else:
                igData[key] = value
        if finalCount > 0:
            while(finalCount > 0):
                for key, value in igData.items():
                    trimmedData[key] = value
                    finalCount -= 1
        return trimmedData
    
    def saveData(self, trimmedData, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trimmedData, f, indent=4)
        
if __name__ == "__main__":
    viewer = FullTextViewer("E:/Windsor/Sem-2/Topics In Applied AI-GNN/Project/Code/data/Datasets/LawFull.json")
    # viewer.firstObjViewer()
    # random_ele = viewer.randomReader()
    # print(random_ele)
    
    # inter_chunks = viewer.totalInterChunks()
    # total_chunks = viewer.totalChunks()
    # print(json.dumps(inter_chunks, indent=4))
    
    trimmedData = viewer.dataTrimer(25)
    viewer.saveData(trimmedData, "E:/Windsor/Sem-2/Topics In Applied AI-GNN/Project/Code/data/Datasets/LawTrimmed_25.json")