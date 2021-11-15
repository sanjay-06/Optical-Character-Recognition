import pytesseract as pt
from pytesseract import Output


class ImageRecognizer:
    def __init__(self, image, *, psm=11, char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
        self.image = image
        self.config = f"--psm {psm} tessedit_char_whitelist = {char_whitelist}"
        self.processed = False

    def recognize(self):
        self.boxes = pt.image_to_boxes(
            self.image, config=self.config, output_type=Output.DICT)
        self.data = pt.image_to_data(
            self.image, config=self.config, output_type=Output.DICT)
        self.text = pt.image_to_string(self.image, config=self.config)
        self.processed = True

    def get_text(self):
        if not self.processed:
            return False
        return self.text

    def get_boxes(self):
        if not self.processed:
            return False
        return self.boxes

    def get_data(self):
        if not self.processed:
            return False
        return self.data
