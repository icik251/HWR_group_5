import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

## CODE FROM: https://www.tutorialfor.com/blog-282403.htm

img = cv2.imread(
    "D:\\RUG courses\\2B\\Handwriting Recognition\\Project\\lines_nice\\roi(0).png"
)  # Read the picture,Converted to operable array
grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert bgr image to grayscale
ret, thresh1 = cv2.threshold(
    grayimage, 130, 255, cv2.THRESH_BINARY
)  # Binary the image (130,255). The points between the two become 255 (background).
# print (thresh1 [0,0]) #250 Output the pixel value of this point [0,0] #Return value ret is the threshold
# print (ret) #130
(h, w) = thresh1.shape  # Return height and width
# print (h, w) #s output height and width
a = [0 for z in range(0, w)]
print(
    a
)  # a=[0,0,0,0,0,0,0,0,0,0, ..., 0,0] initializes an array of length w to record each Number of black dots
# Record the peaks of each column
for j in range(0, w):  # traverse a column
    for i in range(0, h):  # traverse a line
        if thresh1[i, j] == 0:  # If the point is changed to a black point
            a[j] += 1  # The counter in this column is incremented by one
            thresh1[i, j] = 255  # turn it to white after recording
    # print (j)
#
for j in range(0, w):  # Iterate through each column
    for i in range(
        (h - a[j]), h
    ):  # Blacken to the bottom from the top point where the column should turn black
        thresh1[i, j] = 0  # black
# Thresh1 at this time is a histogram of an image projected in the vertical direction
# If i want to split characters,Actually, there is no need to draw this picture.
# Just go to a=[] to get the information i want
# img2=Image.open ("D:\\RUG courses\\2B\\Handwriting Recognition\\Project\\lines_nice\\roi(0).png")
# img2.convert ("l")
# plt.imshow(a, cmap=plt.gray())
curr_idx = 0
list_of_cropped_images = list()
for column_idx in range(len(a)-2):
    if a[column_idx] < 5 and a[column_idx+1] >=5 and a[column_idx+2] >= 5:
        list_of_cropped_images.append(grayimage[:,curr_idx:column_idx])
        curr_idx = column_idx

for i, image in enumerate(list_of_cropped_images):
    cv2.imwrite('D:\\RUG courses\\2B\\Handwriting Recognition\\Project\\lines_nice\\roi(0)\\char_{}.png'.format(i), image)
    
plt.imshow(thresh1, cmap=plt.gray())
plt.show()
#cv2.imshow("img", thresh1)
#cv2.waitkey(0)
#cv2.destroyallwindows()
