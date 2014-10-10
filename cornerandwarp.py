import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = 'images/frog.png'
img = cv2.imread(filename)
img_copy = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01 * dst.max()] = [0,0,255]

cv2.imshow('dst',img)

img = img_copy.copy()

mypts = dst > 0.01 * dst.max()
h, w = mypts.shape
print h, w, mypts

top = None
bottom = None
left = None
right = None

for y in range(0,h):
	for x in range(0,w):
		if mypts[y][x]:
			top = [x,y]
			break
	if top is not None:
		break

for y in range(h-1,-1,-1):
	for x in range(0,w):
		if mypts[y][x]:
			bottom = [x,y]
			break
	if bottom is not None:
		break

for x in range(0,w):
	for y in range(0,h):
		if mypts[y][x]:
			left = [x,y]
			break
	if left is not None:
		break

for x in range(w-1,-1,-1):
	for y in range(0,h):
		if mypts[y][x]:
			right = [x,y]
			print "right is = " + str(right)
			break
	if right is not None:
		break

rows,cols,ch = img.shape
#  y,   x
#pts1 = np.float32([[63,14],[172,50],[10,85],[110,159]])
print top, left, right, bottom
pts1 = np.float32([top, right, left, bottom])
pts2 = np.float32([[0,0],[114,0],[0,84],[114,84]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(114,84))
print len(img[0][0])
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
