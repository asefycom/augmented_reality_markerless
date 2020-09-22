import cv2
import numpy as np

MIN_MATCH_COUNT = 15

img_untold = cv2.imread('untold-ar.jpeg')
img_untold = cv2.resize(img_untold, (0,0), fx=0.25, fy=0.25)
video_untold = cv2.VideoCapture('untold-intro.mp4')
video_untold.set(1, 854)
success, videoImg = video_untold.read()
hI, wI, cI = img_untold.shape
videoImg = cv2.resize(videoImg, (wI, hI))
webcam = cv2.VideoCapture(0)

orb = cv2.ORB_create(nfeatures=1000)
kp, desc = orb.detectAndCompute(img_untold, None)
# img_untold = cv2.drawKeypoints(img_untold, kp, None)

#If you want to select the best matches based on CrossCheck
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#If you want to select the best matches based on Ratio Test by D.Lowe
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


while True:
    success2, webcamImg = webcam.read()
    webcamImg_ar = webcamImg.copy()
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

            src_bnd_pts = np.float32([[0,0],[0,hI],[wI,hI],[wI,0]]).reshape(-1,1,2)
            dst_bnd_pts = cv2.perspectiveTransform(src_bnd_pts,M)
            cv2.polylines(webcamImg, [np.int32(dst_bnd_pts)], True, (0, 0, 255), thickness=2)

            videoWarped = cv2.warpPerspective(videoImg,M, (webcamImg.shape[1], webcamImg.shape[0]))
            maskWin = np.zeros((webcamImg.shape[0], webcamImg.shape[1]), np.uint8)
            cv2.fillPoly(maskWin,[np.int32(dst_bnd_pts)], (255,255,255))
            maskWinInv = cv2.bitwise_not(maskWin)
            webcamImg_ar = cv2.bitwise_and(webcamImg_ar, webcamImg_ar, mask=maskWinInv)
            webcamImg_ar = cv2.bitwise_or(videoWarped, webcamImg_ar)

            cv2.imshow('Video Warped', videoWarped)
            cv2.imshow('Mask Window', webcamImg_ar)

        # cv2.imshow('Matching', img3)

    # cv2.imshow('Webcam', webcamImg)
    # cv2.imshow('Untold Image', img_untold)
    # cv2.imshow('Untold Video', videoImg)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

