import cv2
import numpy as np

img_untold = cv2.imread('untold-ar.jpeg')
video_untold = cv2.VideoCapture('untold-intro.mp4')
webcam = cv2.VideoCapture(0)

success, videoImg = video_untold.read()
hI, wI, cI = img_untold.shape
videoImg = cv2.resize(videoImg, (wI, hI))

orb = cv2.ORB_create(nfeatures=1000)
kp, desc = orb.detectAndCompute(img_untold, None)
img_untold = cv2.drawKeypoints(img_untold, kp, None)

while True:
    success2, webcamImg = webcam.read()
    kp2, desc2 = orb.detectAndCompute(webcamImg, None)
    webcamImg = cv2.drawKeypoints(webcamImg, kp2, None)
    cv2.imshow('Webcam', webcamImg)
    cv2.imshow('Untold Image', img_untold)
    cv2.imshow('Untold Video', videoImg)
    cv2.waitKey(1)

