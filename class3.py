import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True :
    ret, frame = cap.read()
    w = int(cap.get(3))
    h = int(cap.get(4))

    img = np.zeros(frame.shape, np.uint8)


    smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    img[:h//2, :w//2] = cv2.rotate(smaller_frame, cv2.ROTATE_180)
    img[h//2:, :w//2] = smaller_frame
    img[:h//2, w//2:] = cv2.rotate(smaller_frame, cv2.ROTATE_180)
    img[h//2:, w//2:] = smaller_frame

    img = cv2.line(img, (0,0), (w,h), (255, 0, 0), 10)
    cv2.imshow('Cool_screen', img)

    if cv2.waitKey(1) == ord('q'):
        break
#first commit on mac os
cap.release()
cv2.destroyAllWindows()