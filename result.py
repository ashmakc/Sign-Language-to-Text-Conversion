import cv2
import numpy as np
from keras.models import model_from_json
import operator
import sys, os
import random
import string


from tensorflow.python.keras.saving.save import load_model

from tensorflow.keras.models import model_from_json
with open("general1-bw.json","r") as file:
  general1_json=file.read()
  loaded_model=model_from_json(general1_json)
  loaded_model.load_weights("general1-bw.h5")
  print("Loaded model")

cam = cv2.VideoCapture(0)
while(cam.isOpened()):
    
    
    ret, frame = cam.read()
    frame=cv2.flip(frame,1)
    
    
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,2)
    
    roi = frame[y1:y2, x1:x2]
     
 
    #cv2.imshow("Frame", frame)
    minvalue=20
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),2)
    th3=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret,test_image=cv2.threshold(th3,minvalue,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    test_image=cv2.resize(test_image,(128,128))
    cv2.imshow("roi",test_image)

    result= loaded_model.predict(test_image.reshape(1,128,128,1))
    prediction={'A': result[0][0], 
                  'B': result[0][1], 
                  'C': result[0][2],
                  'D': result[0][3],
                  'E': result[0][4],
                  'F': result[0][5],
                  'G': result[0][6], 
                  'H': result[0][7], 
                  'I': result[0][8],
                  'J': result[0][9],
                  'K': result[0][10],
                  'L': result[0][11],
                  'M': result[0][12], 
                  'N': result[0][13], 
                  'O': result[0][14],
                  'P': result[0][15],
                  'Q': result[0][16],
                  'R': result[0][17],
                  'S': result[0][18], 
                  'T': result[0][19], 
                  'U': result[0][20],
                  'V': result[0][21],
                  'W': result[0][22],
                  'X': result[0][23],
                  'Y': result[0][24], 
                  'Z': result[0][25],
                  } 
    
    
    prediction=sorted(prediction.items(),key=operator.itemgetter(1),reverse=True)
    #prediction=26
    
    cv2.putText(frame,prediction[0][0],(100,120),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5,cv2.LINE_AA)

    cv2.putText(frame,'Predicted Text:',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_4)
    cv2.imshow("FRAME",frame)
    #cv2.putText(frame,  ":" + str(result[int(prediction)+1]), (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255, 255, 255), 2)
    #prediction=prediction+1
    interrupt=cv2.waitKey(10)
    if interrupt & 0xFF==27:
       break
        
cam.release()
cv2.destroyAllWindows()