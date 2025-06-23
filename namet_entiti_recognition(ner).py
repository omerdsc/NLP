
"""
varlık ismi tanima: metin (cumle) -> metin içerisinde bulunan varlık isimilerini tanimla
"""

# import libraries"
import pandas as pd
import spacy


# spacy modeli ile varlık tanıma

nlp=spacy.load("en_core_web_sm") # spacy kutuphanesi ingilizce dil modeli

content="Alşce work at Amazon and lives in London. She visited the British Museum Last weekend."

doc=nlp(content) # b isime metindeki varliklari (entities) analiz eder


for ent in doc.ents:
    # ent.text: varlık ismi (Alice, Amazon)
    # nt.start_char  ve ent.end_char: varligin baslangic ve bitis karekterleri
    # ent.label_: varlık turu
    # print(end.text, ent.start_char, ent.end_char, ent_label_)
    print(ent.text,ent.label_)
    
entities=[(ent.text,ent.label_,ent.lemma_) for ent in doc.ents]

df=pd.DataFrame(entities, columns=["text","type","lemma"])