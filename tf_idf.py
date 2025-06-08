# import libraries

import pandas as pd
import numpy as np


from sklearn.feature_extraction.text import TfidfVectorizer


#ornek belge olustur

documents=[
    "Köpek çok tatlı bir hayvandır",
    "Köpek ve kuşlar çok tatlı hayvanlardır.",
    "inekler süt üretirler."]


# vektorizer tanimla
tfidf_vectorize=TfidfVectorizer()


# metinleri sayiyisal hale cevir
x=tfidf_vectorize.fit_transform(documents)


# kelime skumesini ıncele

feature_names=tfidf_vectorize.get_feature_names_out()

# vektor temsili incele

vektor_temsili=x.toarray()
print(f"tf-ı0df: {vektor_temsili}")

df_tfidf=pd.DataFrame(vektor_temsili,columns=feature_names)

# ortalama tf ıdf degerlerine bakalım

tf_idf=df_tfidf.mean(axis=0)