import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Paracetamol is a drug|")

for word in doc.ents:
    print(word.text, word.label_)
