import os

files = os.listdir("../Datasets/raw/BigData/downloaded_papers/")

class_dir = {}
for file in files:
    fname = ""
    for i in file:
        if '_' == i:
            break
        fname += i
    if fname not in class_dir:
        class_dir[fname] = 1
    else:
        class_dir[fname]+=1
for key, value in class_dir.items():
    print(key, value)   

classes_toKeep = ['math', 'physics', 'q-fin', 'q-bio', 'econ']

# for file in files:
#     fname = ""
#     for i in file:    
#         if '_' == i:
#             break    
#         fname += i
#     if fname not in classes_toKeep:
#         os.remove("../Datasets/raw/BigData/downloaded_papers/"+file)    