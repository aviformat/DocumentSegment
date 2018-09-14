# coding=utf-8
# coding=utf-8
import cv2
import numpy as np
import sys

size=512
def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
def draw(img2):
    size = 512
    img1 = img2
    img2 = cv2.resize(img2, (size, size))
    img1 = cv2.resize(img1, (size, size))
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (21, 21), 0)

    img = cv2.Canny(img, 200, 300, apertureSize=5, L2gradient=True)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, 1)
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 30, minLineLength=150)
    allAngle = []
    true_Angle = []
    coordi = []
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(img1, (x1, y1), (x2, y2), (0, 0, 255), 2)
            angle = (y1 - y2) / float(x2 - x1)
            angle = np.rad2deg(np.arctan(angle))
            if angle >= -45 and angle <= 45:
                true_Angle.append(angle)
                coordi.append((x1, y1, x2, y2))
            allAngle.append(angle)
    return (img1)
def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
def four_point_transform(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
def preprocess(img2):
    size=512
    img1 = img2
    img2 = cv2.resize(img2, (size, size))
    img1 = cv2.resize(img1, (size, size))
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (21, 21), 0)

    img = cv2.Canny(img, 200, 300, apertureSize=5, L2gradient=True)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel, 1)
    return (img)
def houghTransform(img):
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 30, minLineLength=150)
    #
    #
    allAngle = []
    true_Angle = []
    coordiS, coordiE = [], []
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            # cv2.line(origImg, (x1, y1), (x2, y2), (0, 0, 255), 2)
            angle = (y1 - y2) / float(x2 - x1)
            angle = np.rad2deg(np.arctan(angle))
            if angle >= -45 and angle <= 45:
                true_Angle.append(angle)
                coordiS.append((x1, y1))
                coordiE.append((x2, y2))
            allAngle.append(angle)
    return (true_Angle,coordiS,coordiE)

def perspectiveTransform(img1,true_Angle,coordiS,coordiE):
    topLeft = []
    topRight = []
    bottomLeft = []
    bottomRight = []
    sumtotal = 100000
    for i in range(len(coordiS)):
        if coordiS[i][0] + coordiS[i][1] < sumtotal:
            sumtotal = coordiS[i][0] + coordiS[i][1]
            topLeft = (coordiS[i][0], coordiS[i][1])
    sumtotal = 0
    for i in range(len(coordiE)):
        if coordiE[i][0] + coordiE[i][1] > sumtotal:
            sumtotal = coordiE[i][0] + coordiE[i][1]
            bottomRight = (coordiE[i][0], coordiE[i][1])
    if np.mean(true_Angle) > 0:
        sumy = 10000
        for i in range(len(coordiE)):
            if coordiE[i][1] <= sumy:
                sumy = coordiE[i][1]
                topRight = (coordiE[i][0], coordiE[i][1])
        sumy = 0
        for i in range(len(coordiS)):
            if coordiS[i][1] >= sumy:
                sumy = coordiS[i][1]
                bottomLeft = (coordiS[i][0], coordiS[i][1])

    if np.mean(true_Angle) <= 0:
        sumx = 0
        for i in range(len(coordiE)):
            if coordiE[i][0] >= sumx:
                sumx = coordiE[i][0]
                topRight = (coordiE[i][0], coordiE[i][1])
        sumx = 100000
        coordiS = sorted(coordiS, key=lambda x: x[1])
        for i in range(len(coordiS)):
            if coordiS[i][0] <= sumx:
                sumx = coordiS[i][0]
                bottomLeft = (coordiS[i][0], coordiS[i][1])
    # print coordiS, np.mean(true_Angle)
    # print topLeft, topRight, bottomLeft, bottomRight
    blah = four_point_transform(img1, np.array([topLeft, topRight, bottomRight, bottomLeft]))
    return (blah)
original_img=cv2.imread(str(sys.argv[1]))
original_img=cv2.resize(original_img,(size,size))
preprocessedImg=preprocess(original_img)
true_Angle,coordinate_start,coordinate_end=houghTransform(preprocessedImg)
final_img=perspectiveTransform(original_img,true_Angle,coordinate_start,coordinate_end)

cv2.imshow('final_img',final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('CroppedImage.jpg',final_img)
# print original_img.shape,final_img.shape
final_img=cv2.resize(final_img,(512,512))
final_img = np.concatenate((original_img, final_img), axis=1)
cv2.imshow('final image',final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

