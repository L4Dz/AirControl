import numpy as np 
import cv2

cap = cv2.VideoCapture(0)

while True :
    ret, frame = cap.read()
    w = int(cap.get(3))
    h = int(cap.get(4))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(frame, frame, mask=mask_inv)




    cv2.imshow('Cool_screen', result)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 