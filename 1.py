from PIL import Image, ImageOps
import pytesseract as pt
import cv2
import numpy as np
import re


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal


def remove_noise(image):
    return cv2.medianBlur(image, 5)

# thresholding


def thresholding(image):
    return cv2.threshold(image, 0, 255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# dilation


def dilate(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

# erosion


def erode(image):
    kernel = np.ones((6, 6), np.uint8)
    print(kernel)
    return cv2.erode(image, kernel, iterations=1)

# opening - erosion followed by dilation


def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection


def canny(image):
    return cv2.Canny(image, 100, 200)

# skew correction


def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# template matching


def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


def sharpen(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.filter2D(src=image, ddepth=-1, kernel=kernel)


def preprocess(img):
    # pt.pytesseract.tesseract_cmd = r'C:\Users\Sanjay T\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    image = get_grayscale(img)
    # image=sharpen(image)
    image = thresholding(image)
    print(image.shape)
    # image=erode(image)
    # image=dilate(image)
    # image = opening(image)
    # image=canny(image)
    # image = cv2.bitwise_not(image)
    # cv2.imshow('AV CV- Winter Wonder Sharpened', image)

    # image=match_template(image,img)
    return image


img = cv2.imread('Tablet/Capture.JPG')
image = img
custom_config = r'--psm 11 tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
numberplateRegex = re.compile(
    "(([A-Z]\s*){2}([0-9]\s*){2}([A-Z]\s*){2}([0-9]\s*){4})")

textNoPreprocess = pt.image_to_string(image, config=custom_config)

match = numberplateRegex.findall(textNoPreprocess)
if len(match) > 0:
    for match in match:
        print("Found match: ", match)
        print(textNoPreprocess)
    exit()

image = preprocess(image)
textPreprocess = pt.image_to_string(image, config=custom_config)

match = numberplateRegex.findall(textPreprocess)
if len(match) > 0:
    for match in match:
        print("Found match after preprocessing: ", match)
    exit()

print("No matches found after preprocessing, displaying full text")
print(textPreprocess)
print("Without preprocessing:")
print(textNoPreprocess)
# image = preprocess(img)
# cv2.imshow("gray", image)
# cv2.waitKey(0)
# img = get_grayscale(img)

# text2 = pt.image_to_string(image, config=custom_config)
# print(text2)

# result = re.findall(numberplateRegex, text2)
# print(result)
