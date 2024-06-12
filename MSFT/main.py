import os
import time

import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, CSRTensor, COOTensor
import cv2
import pydegensac
from PIL import Image
from pyramid_calor import pyramid_calor
import pysift


if __name__ == '__main__':
    im1_path = 'images/0018_nir.tiff'
    im2_path = 'images/0018_rgb.tiff'
    im1_rgb = cv2.imread(im1_path)
    im2_rgb = cv2.imread(im2_path)

    if im1_rgb.shape[2] == 3:
        im1 = cv2.cvtColor(im1_rgb, cv2.COLOR_RGB2GRAY)
        im1 = im1 / 255
    else:
        im1 = im1_rgb / 255

    if im2_rgb.shape[2] == 3:
        im2 = cv2.cvtColor(im2_rgb, cv2.COLOR_RGB2GRAY)
        im2 = im2 / 255
    else:
        im2 = im2_rgb / 255

    # im1 = Tensor(im1)
    # im2 = Tensor(im2)
    t0 = time.time()
    kp1, desc1 = pysift.computeKeypointsAndDescriptors(im1)
    kp2, desc2 = pysift.computeKeypointsAndDescriptors(im2)
    print('Average match time: ', time.time() - t0)

    # match
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 1 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx] for m in good]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx] for m in good]).reshape(-1, 2)
    H_pred, inliers = pydegensac.findHomography(src_pts, dst_pts, 5.0)
    matchesMask = inliers.ravel().tolist()
    matchesMask = [int(boo) for boo in matchesMask]
    print('inlier nums: ', np.array(matchesMask).sum())
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    kp1 = [cv2.KeyPoint(point[0], point[1], 1) for point in kp1]
    kp2 = [cv2.KeyPoint(point[0], point[1], 1) for point in kp2]
    img3 = cv2.drawMatches(np.array(im1_rgb), kp1, np.array(im2_rgb), kp2, good, None, **draw_params)
    Image.fromarray(img3).save('matches.png')


