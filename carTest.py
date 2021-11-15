from PIL import Image
import regex

from imageRecognizer import ImageRecognizer

car = Image.open("./Car/CarPlate/carcrop.jpg")
ir = ImageRecognizer(car, psm=6)
ir.recognize()
text = ir.get_text()
print("Text: ", text)

numberplateRegex = regex.compile(
    r"(([A-Z]\s*){2}([0-9]\s*){1,2}([A-Z]\s*){2}([0-9]\s*){1,4}){s<2}")
matches = numberplateRegex.findall(text)
print(matches)
if len(matches) > 0:
    for match in matches:
        print("Found match: ", match)
        print(text)
else:
    print("No match")
print("Match: ", matches[0][0].strip())
