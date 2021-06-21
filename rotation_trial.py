import os
from pathlib import Path
import cv2
import numpy as np
from data_loader import DataLoader
from numpy.core.fromnumeric import size
from PIL import Image
from matplotlib import pyplot as plt




data_loader = DataLoader()
dict_result = data_loader.get_characters_train_data(
    path = "c:\\Users\\kaany\\Documents\\rug/HWR\\project\\venv11\\monkbrill2", 
    num_samples = 1
)
"""
for k, v in dict_result.items():
    print(k)
    print(len(v))
    print("----------")
"""

#print(dict_result["Alef"])
arr = np.array
arr =dict_result["Alef"]
print(arr[0])
arr = arr[0]
theta = 355

im = Image.fromarray(arr)
im_rotate = im.rotate(theta)
plt.imshow(im_rotate)
plt.show()