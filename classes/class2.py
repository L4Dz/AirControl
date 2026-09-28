import cv2
import random

img = cv2.imread('assets/shanghai.jpg' , -1)
img = cv2.resize(img, (400, 400))
for i in range (img.shape[0]):
    for j in range (100):
        img[i][j]= [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#print (type(img))