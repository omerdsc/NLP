# import libraries
import pandas as pd
import numpy  as np
from  sklearn.feature_extraction.text import TfidfVectorizer

# veri seti yükle
df=pd.read_csv("spam.csv",encoding="ISO-8859-1")


# tf-idf

vectorizer=TfidfVectorizer()

x=vectorizer.fit_transform(df.v2)


# kelimem kümesini incele

feature_names=vectorizer.get_feature_names_out()
tfidf_score=x.mean(axis=0).A1

# tfidf skorlarını içeren bir df olustur

df_tfidf=pd.DataFrame({"word":feature_names,"tfidf score":tfidf_score})

# skorlari sirala ve sonuçları incele

df_tfidf_sorted=df_tfidf.sort_values(by="tfidf_score",ascending=False)