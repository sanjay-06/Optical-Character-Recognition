import cv2
import matplotlib.pyplot as plt
import numpy as np
import  pytesseract as pt
import regex
from imageRecognizer import ImageRecognizer

lic=cv2.CascadeClassifier('numberplate.xml')
def plt_show(image,title='',gray=False,size=(100,100)):
    temp=image
    if gray == False:
        temp=cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
        plt.title(title)
        # plt.imshow(temp)
        # plt.show()

def detect(img):
    temp=img
    gray=cv2.cvtColor(temp,cv2.COLOR_RGB2GRAY)
    number=lic.detectMultiScale(img,1.2)
    print("number plate detected: "+str(len(number)))
    for numbers in number:
        (x,y,w,h)=numbers
        print(x,y,w,h)
        roi_color=img[y:y+h,x:x+w]
        # cv2.rectangle(temp,(x,y),(x+w,y+h),(0,255,0),3)
    # plt_show(temp)
    kernel=np.ones((5,5),np.uint8)
    erosion=cv2.erode(roi_color,kernel,iterations=1)
    # plt.subplot(1,1,1)
    # plt.imshow(erosion)
    cv2.imshow("j",roi_color)
    plt.show()
    return roi_color

img=cv2.imread(r'C:\Users\Sanjay T\Desktop\sem5\ml package\Optical-Character-Recognition\Car\CarFull\car5.jpg')
plt_show(img)
a=detect(img)
pt.pytesseract.tesseract_cmd = r'C:\Users\Sanjay T\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
NUMERPLATE_PATTERN=regex.compile(
    r"(([A-Z]\s*){2}([0-9]\s*){1,2}([A-Z]\s*){2}([0-9]\s*){2,4}){s<3}")
def get_numberplate_matches(text):
  matches = NUMERPLATE_PATTERN.findall(text)
  return matches

ir=ImageRecognizer(a)
ir.recognize()
text=ir.get_text()
print(text)
matches=get_numberplate_matches(text)
print(matches)
print(matches[0][0])

