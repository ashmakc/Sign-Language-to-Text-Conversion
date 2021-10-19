#import imp
import cv2
import numpy as np 
import math
import os
import string

from tensorflow.python.keras.saving.save import load_model

from tensorflow.keras.models import model_from_json
with open("model1.json","r") as file:
  model1_json=file.read()
  loaded_model=model_from_json(model1_json)
  loaded_model.load_weights("new1_model.h5")
  print("Loaded model")

if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/train"):
    os.makedirs("data/train")
    
if not os.path.exists("data/test"):
    os.makedirs("data/test")
#for i in range(1):
   # if not os.path.exists("data/train/" + str(i)):
    #    os.makedirs("data/train/"+str(i))
    #if not os.path.exists("data/test/" + str(i)):
     #   os.makedirs("data/test/"+str(i))
for i in string.ascii_uppercase:
    if not os.path.exists("data/train/" +i):
        os.makedirs("data/train/" +i)
    if not os.path.exists("data/test/" + i):
        os.makedirs("data/test/"+i)    
       

cam = cv2.VideoCapture(0)
#cv2.namedWindow("Trackbars")
#def nothing(x):
 #   pass
#cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
#cv2.createTrackbar("U - H", "Trackbars", 0, 255, nothing)
#cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
#cv2.createTrackbar("U - S", "Trackbars", 179, 179, nothing)
#cv2.createTrackbar("L - V", "Trackbars", 255, 255, nothing)
#cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)
 
while(cam.isOpened()):
    #Get frame
    
    ret, frame = cam.read()
    frame=cv2.flip(frame,1)
    
    # Show frame
    #if cv2.waitKey(1)&0xFF == ord('q'):
       # break
    
      # Coordinates of the ROI
    
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,2)
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]
    #roi = cv2.resize(roi, (64, 64)) 
 
    cv2.imshow("Frame", frame)
    minvalue=20
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),2)
    #test=cv2.resize(blur,(minvalue,minvalue))
    #th3=cv2.threshold(blur,120,255,cv2.THRESH_BINARY)
    th3=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret,test_image=cv2.threshold(th3,minvalue,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    test_image=cv2.resize(test_image,(300,300))
    #_,roi=cv2.threshold(blur,130,2050,cv2.THRESH_BINARY)
    cv2.imshow("roi",test_image)
    
    #gray=cv2.resize(gray,)
    
    #_, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    #kernel = np.ones((1, 1), np.uint8)
    #img = cv2.dilate(mask, kernel, iterations=1)
    #img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
   # hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    #l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    #u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    #l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    #u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    #l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    #u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    #lower_range = np.array([l_h, l_s, l_v])
    #upper_range = np.array([u_h, u_s, u_v])
    #imcrop = img[102:298, 427:623]
   
    #mask = cv2.inRange(hsv, lower_range, upper_range)

    #result = cv2.bitwise_and(roi, roi, mask=mask)
    #blur = cv2.GaussianBlur(result,(9,9),0)
    #_, roi = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)
   # cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))

    #cv2.imshow("mask", mask)
    #cv2.imshow("ROI", roi)
    mode='train' 
    directory='data/'+mode+'/'

    #getting count
    #cv2.putText(frame,"mode : "+mode,(10,50),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255,1))
    count={
        'a':len(os.listdir(directory+"/A")),
        'b':len(os.listdir(directory+"/B")),
        'c':len(os.listdir(directory+"/C")),
        'd':len(os.listdir(directory+"/D")),
        'e':len(os.listdir(directory+"/E")),
        'f':len(os.listdir(directory+"/F")),
        'g':len(os.listdir(directory+"/G")),
        'h':len(os.listdir(directory+"/H")),
        'i':len(os.listdir(directory+"/I")),
        'j':len(os.listdir(directory+"/J")),
        'k':len(os.listdir(directory+"/K")),
        'l':len(os.listdir(directory+"/L")),
        'm':len(os.listdir(directory+"/M")),
        'n':len(os.listdir(directory+"/N")),
        'o':len(os.listdir(directory+"/O")),
        'p':len(os.listdir(directory+"/P")),
        'q':len(os.listdir(directory+"/Q")),
        'r':len(os.listdir(directory+"/R")),
        's':len(os.listdir(directory+"/S")),
        't':len(os.listdir(directory+"/T")),
        'u':len(os.listdir(directory+"/U")),
        'v':len(os.listdir(directory+"/V")),
        'w':len(os.listdir(directory+"/W")),
        'x':len(os.listdir(directory+"/X")),
        'y':len(os.listdir(directory+"/Y")),
        'z':len(os.listdir(directory+"/Z")),
        
        
    }

    #printing count
   # cv2.putText(frame, "Image count : "+str(count), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "a : "+str(count['a']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "b : "+str(count['b']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "c : "+str(count['c']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "d : "+str(count['d']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "e : "+str(count['e']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "f : "+str(count['f']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "g : "+str(count['g']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "h : "+str(count['h']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "i : "+str(count['i']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    #cv2.putText(frame, "j : "+str(count['j']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)


    interrupt=cv2.waitKey(10)
    if interrupt & 0xFF==27:
        break
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory+'A/'+str(count['a'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'B/'+str(count['b'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'C/'+str(count['c'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'D/'+str(count['d'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'E/'+str(count['e'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(directory+'F/'+str(count['f'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory+'G/'+str(count['g'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'H/'+str(count['h'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'I/'+str(count['i'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory+'J/'+str(count['j'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(directory+'K/'+str(count['k'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory+'L/'+str(count['l'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(directory+'M/'+str(count['m'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'N/'+str(count['n'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory+'O/'+str(count['o'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(directory+'P/'+str(count['p'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(directory+'Q/'+str(count['q'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory+'R/'+str(count['r'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory+'S/'+str(count['s'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory+'T/'+str(count['t'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(directory+'U/'+str(count['u'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(directory+'V/'+str(count['v'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(directory+'W/'+str(count['w'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(directory+'X/'+str(count['x'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory+'Y/'+str(count['y'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(directory+'Z/'+str(count['z'])+'.jpg', test_image)
    

    
   
    mode='test'
    directory='data/'+mode+'/'
 
    #getting count
    
    count={
        'A':len(os.listdir(directory+"/A")),
        'B':len(os.listdir(directory+"/B")),
        'C':len(os.listdir(directory+"/C")),
        'D':len(os.listdir(directory+"/D")),
        'E':len(os.listdir(directory+"/E")),
        'F':len(os.listdir(directory+"/F")),
        'G':len(os.listdir(directory+"/G")),
        'H':len(os.listdir(directory+"/H")),
        'I':len(os.listdir(directory+"/I")),
        'J':len(os.listdir(directory+"/J")),
        'K':len(os.listdir(directory+"/K")),
        'L':len(os.listdir(directory+"/L")),
        'M':len(os.listdir(directory+"/M")),
        'N':len(os.listdir(directory+"/N")),
        'O':len(os.listdir(directory+"/O")),
        'P':len(os.listdir(directory+"/P")),
        'Q':len(os.listdir(directory+"/Q")),
        'R':len(os.listdir(directory+"/R")),
        'S':len(os.listdir(directory+"/S")),
        'T':len(os.listdir(directory+"/T")),
        'U':len(os.listdir(directory+"/U")),
        'V':len(os.listdir(directory+"/V")),
        'W':len(os.listdir(directory+"/W")),
        'X':len(os.listdir(directory+"/X")),
        'Y':len(os.listdir(directory+"/Y")),
        'Z':len(os.listdir(directory+"/Z")),
       
    }

    interrupt=cv2.waitKey(10)
    if interrupt & 0xFF==27:
        break
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory+'A/'+str(count['A'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'B/'+str(count['B'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'C/'+str(count['C'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'D/'+str(count['D'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'E/'+str(count['E'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(directory+'F/'+str(count['F'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory+'G/'+str(count['G'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'H/'+str(count['H'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'I/'+str(count['I'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory+'J/'+str(count['J'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(directory+'K/'+str(count['K'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory+'L/'+str(count['L'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(directory+'M/'+str(count['M'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory+'N/'+str(count['N'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory+'O/'+str(count['O'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(directory+'P/'+str(count['P'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(directory+'Q/'+str(count['Q'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory+'R/'+str(count['R'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory+'S/'+str(count['S'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory+'T/'+str(count['T'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(directory+'U/'+str(count['U'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(directory+'V/'+str(count['V'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(directory+'W/'+str(count['W'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(directory+'X/'+str(count['X'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory+'Y/'+str(count['Y'])+'.jpg', test_image)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(directory+'Z/'+str(count['Z'])+'.jpg', test_image)
    

   
   
cam.release()
cv2.destroyAllWindows()

    
