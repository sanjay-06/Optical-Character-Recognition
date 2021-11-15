import pytesseract as pt
import cv2
import spacy
from pytesseract import Output

nlp = spacy.load("en_core_web_sm")

img = cv2.imread('Tablet/Capture.jpg')
custom_config = r'--psm 11 tessedit_char_whitelist = ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

d = pt.image_to_data(img, output_type=Output.DICT)
print("Data extracted")
print(d['height'])
print(d['text'])
n_boxes = len(d['level'])
maxHeight = -1
bigText = ""

for i in range(n_boxes):
    h = d['height'][i]
    text = d['text'][i]
    if not text.strip() or not text.isalnum():
        continue
    if h > maxHeight:
        maxHeight = h
        bigText = text

print(f"Biggest text: {bigText} with height {maxHeight}")
outputText = pt.image_to_string(img)
# print(outputText)
doc = nlp(outputText)
for word in doc.ents:
    print(word.text, word.label_)
# TODO handle all characters for biggest word
