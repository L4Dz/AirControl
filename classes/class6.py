import numpy as np
import cv2

img = cv2.imread('assets/Chess_board.svg.webp', -1)

img = cv2.resize(img, (0, 0), fx=0.75, fy=0.75)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.intp(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)

for i in range(len(corners)):
    for j in range(i+1, len(corners)):
        x1, y1 = corners[i].ravel()
        x2, y2 = corners[j].ravel()
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)   

cv2.imshow('Chess Board', img)

cv2.waitKey(0)
cv2.destroyAllWindows()