import cv2
import numpy as np

MIN_MATCH_COUNT = 15

img_untold = cv2.imread('untold-ar.jpeg')
img_untold = cv2.resize(img_untold, (0,0), fx=0.25, fy=0.25)
video_untold = cv2.VideoCapture('untold-intro.mp4')
webcam = cv2.VideoCapture(0)

success, videoImg = video_untold.read()
hI, wI, cI = img_untold.shape
videoImg = cv2.resize(videoImg, (wI, hI))

orb = cv2.ORB_create(nfeatures=1000)
kp, desc = orb.detectAndCompute(img_untold, None)
# img_untold = cv2.drawKeypoints(img_untold, kp, None)

#If you want to select the best matches based on CrossCheck
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#If you want to select the best matches based on Ratio Test by D.Lowe
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


while True:
    success2, webcamImg = webcam.read()
    kp2, desc2 = orb.detectAndCompute(webcamImg, None)
    # webcamImg = cv2.drawKeypoints(webcamImg, kp2, None)
    if desc2 is not None:
        # matches = bf.match(desc, desc2)
        # matches = sorted(matches, key=lambda x: x.distance)
        # img3 = cv2.drawMatches(img_untold, kp, webcamImg, kp2, matches[:10], flags=2, outImg=None)
        matches = bf.knnMatch(desc, desc2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        img3 = cv2.drawMatchesKnn(img_untold, kp, webcamImg, kp2, good, flags=2, outImg=None)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp[m.queryIdx].pt for [m] in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for [m] in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            #Calculate the outliers
            # src_pts_kp = [kp[m.queryIdx].pt for [m] in good]
            #
            # correct_matched_kp = [src_pts_kp[i] for i in range(len(src_pts_kp)) if mask[i]]
            # print(correct_matched_kp)

        cv2.imshow('Maching', img3)

    cv2.imshow('Webcam', webcamImg)
    cv2.imshow('Untold Image', img_untold)
    cv2.imshow('Untold Video', videoImg)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

