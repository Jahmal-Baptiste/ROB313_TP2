import cv2
import numpy as np

def ransac(N, pts1, pts2):
    for k in range(N):
        selected_indexes             = np.random.choice(pts1.shape[0], 7, replace=False)
        selected_pts1, selected_pts2 = pts1[selected_indexes], pts2[selected_indexes]
        FRansac, mask = cv2.findFundamentalMat(selected_pts1, selected_pts2, cv2.FM_7POINT)
        print(FRansac.shape)

        lines2 = cv2.computeCorrespondEpilines(selected_pts1.reshape(-1,1,2), 2, FRansac)

