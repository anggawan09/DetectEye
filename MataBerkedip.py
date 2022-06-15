from itertools import count
from tkinter import font    
import cv2

from matplotlib.pyplot import flag
# import cv2
cascade_wajah   =   cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cascade_mata    =   cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
cap = cv2.VideoCapture (0)
font = cv2.FONT_HERSHEY_PLAIN
count = 0
flag = False

while(1):
    ret, frame  =   cap.read()
    gray    =   cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray    =   cv2.bilateralFilter(gray, 5, 1, 1)
    faces   =   cascade_wajah.detectMultiScale(gray, 1.3, 5, minSize=(200,200))
    if(len(faces)>0):
        for(x, y, w, h) in faces:
            frame = cv2.retangle(frame, (x,y),(x+w, y+h), (255, 255, 0), 3)

            roi_gray    =   gray[y:y+h, x:x+w]
            roi_color   =   frame[y:y+h, x:x+w]
            mata        = cascade_mata.detectMultiScale(roi_gray, 1.3, 5, minSize=(20,20))
            jumlah      = len(mata)
            if(jumlah == 0 and flag == False):
                count += 1; flag = True
            if(jumlah == 2):
                flag = False

    cv2.putText(frame,"Mata Berkedip : " + str(count), (70,70),font, 3,(0,0,255),2)
    cv2.imshow('Face Detection',frame)
    a = cv2.waitKey(1)
    if(a==ord('q')):
        break