import spacy

nlp=spacy.load("en_core_web_sm")

# incelenek olan kelime ve kelimeler

word="book"

# kelimeyi nlp işleminden geçir

doc=nlp(word) 

for token in doc:
    
    print(f"Txt: {token.text}")          # kelimenin kendsi
    print(f"Lemma: {token.lemma_}")       # kelimenin kök hali
    print(f"POS: {token.pos_}")          # k3limenin dil bilgisel özelliği
    print(f"Tag: {token.tag_}")          # kelimenin detaylı dilbilgisel özelliği
    print(f"Dependency: {token.dep_}")   # kelimenin rolü
    print(f"shape: {token.shape_}")      # karekter yapisi
    print(f"is alpha: {token.is_alpha}") # kelimenin yalnizca alfabetik karekterlerden oluşup oluşmadığını kontrol eder
    print(f"is stop: {token.is_stop}")   # kelimenin stop word olup olmadığı
    print(f"Morfoloji: {token.morph}")   # kelimenin morfolojik özelliklerini verir
    print(f"is plural: {'Number=Plur' in token.morph}") # kelimenin çoğul olup olmadığı
    
    