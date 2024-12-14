import os
import sys
import fileinput

dataset_folder = '/home/bruno/github/CT-Super-Resolution/Dataset/'
# dataset_folder = 'C:\\Users\\bruno\\Downloads\\Dataset\\'
images_folder = '/home/bruno/github/CT-Super-Resolution/Images/'
# images_folder = 'C:\\Users\\bruno\\Downloads\\Images\\'
models_folder = '/home/bruno/github/CT-Super-Resolution/Models/'
# models_folder = 'C:\\Users\\bruno\\Downloads\\Models\\'

if os.name == "nt":
    dataset_folder = dataset_folder.replace("\\","\\\\")
    images_folder = images_folder.replace("\\","\\\\")
    models_folder = models_folder.replace("\\","\\\\")
    
file = open('CT Super Resolution_TEMPLATE.chn','r')
outFile = open('CT Super Resolution_edited.chn','w')

for line in file:
    line = line.replace("[YOUR_MODEL_PATH_HERE]",models_folder)
    line = line.replace("[YOUR_IMAGE_PATH_HERE]",images_folder)
    line = line.replace("[YOUR_DATASET_PATH_HERE]",dataset_folder)
    outFile.write(line)

file.close()
outFile.close()

