import cv2
import numpy as np

img_untold = cv2.imread('untold-ar.jpeg')
video_untold = cv2.VideoCapture('untold-intro.mp4')
webcam = cv2.VideoCapture(0)

success, videoImg = video_untold.read()
hI, wI, cI = img_untold.shape
videoImg = cv2.resize(videoImg, (wI, hI))

while True:
    success2, webcamImg = webcam.read()
    cv2.imshow('Webcam', webcamImg)
    cv2.imshow('Untold Image', img_untold)
    cv2.imshow('Untold Video', videoImg)
    cv2.waitKey(1)

