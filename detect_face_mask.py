import numpy as np
import cv2
import random
import time

# multiple cascades could be found here: https://github.com/Itseez/opencv/tree/master/data/haarcascades

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')


# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 80

# User message
font = cv2.FONT_HERSHEY_SIMPLEX
org = (100, 30)
weared_mask_font_color = (255, 255, 255)
not_weared_mask_font_color = (0, 0, 255)
thickness = 1
font_scale = 1

weared_mask = "Good"
not_weared_mask = "BAD"

# Read video
cap = cv2.VideoCapture(0)
time.sleep(2.0)

starting_time = time.time()
frame_id = 0

while 1:
    # Get individual frame
    ret, img = cap.read()
    img = cv2.flip(img,1)
    frame_id += 1

    # Convert Image into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert image in black and white
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)

    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Face prediction for black and white
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)


    if(len(faces) == 0 and len(faces_bw) == 0):
        cv2.putText(img, "No face found...", org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Detect nose
            nose_rects = nose_cascade.detectMultiScale(gray, 1.5, 5)

        # Face detected but Lips not detected which means person is wearing mask
        if( len(nose_rects) == 0 ):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, weared_mask, org, font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        else:
            for (nx, ny, nw, nh) in nose_rects:
                if(y < ny < y + h):
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, not_weared_mask, org, font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
                    cv2.rectangle(img, (nx, ny), (nx + nh, ny + nw), (0, 0, 255), 3)
                    break


    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(img, "FPS=" + str(round(fps,2)), (10,10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    # Show frame with results
    cv2.imshow('Mask Detection', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release video
cap.release()
cv2.destroyAllWindows()
