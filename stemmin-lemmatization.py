# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 15:28:23 2025

@author: omer_
"""

import nltk

nltk.download("wordnet") #wordnet: lemmatization işlmei için gerekli veri tabani

from nltk.stem import PorterStemmer #stemming için fonksiyon


#Porte stemmer nennesii oluştur

stemmer=PorterStemmer()

words=["runing","runner","ran","runs","better","go","went"]

# kelimelerim köklerini buluyoruz, bunu yaparkende porter stemmerinın steam(9 fonksiyonunu kullanıyoruz)
stems=[stemmer.stem(w) for w in words]

print(f"stem: {stems}")

#%% lemmatization

from nltk.stem import WordNetLemmatizer


lemmatizer=WordNetLemmatizer()

words=["running","runner","ran","runs","better","go","went"]
lemmas=[lemmatizer.lemmatize(w,pos="v") for w in words]

print(f"lemma: {lemmas}")