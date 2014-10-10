import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('images/frog.png')
rows,cols,ch = img.shape

pts1 = np.float32([[63,14],[172,50],[10,85],[110,159]])
pts2 = np.float32([[0,0],[114,0],[0,84],[114,84]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(114,84))
print len(img[0][0])
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
