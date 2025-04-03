import os
import shutil

classes_toKeep = ['math', 'physics', 'q-fin', 'q-bio', 'econ']
files = os.listdir("../Datasets/raw/BigData/downloaded_papers/")
for file in files:
    fname = ""
    for i in file:
        if '_' == i:
            break
        fname += i
    if fname == 'math' or fname == 'physics':
        destination = "../Datasets/raw/Education_/"+file
    elif fname == 'q-fin':
        destination = "../Datasets/raw/Finance_/"+file
    elif fname == 'q-bio':
        destination = "../Datasets/raw/Biology_/"+file    
    elif fname == 'econ':    
        destination = "../Datasets/raw/Law_/"+file
    source = "../Datasets/raw/BigData/downloaded_papers/"+file
    shutil.move(source, destination)