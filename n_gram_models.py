# import libraries
import pandas as pd
import numpy as np

import nltk
from nltk.util import ngrams # n gram modeli olustumak için
from nltk.tokenize import word_tokenize # tokenization

from collections import Counter

# ornek veri seti oluştur
corpus=["I love apple",
        "I love him",
        "I love NLP",
        "You love me",
        "He loves apple",
        "They love apple",
        "I love you and you love me"]

""" 
dil modeli yapmak istiyoruz
amaç 1 kelimeden sonra gelecek kelimeyi tahmin etmek: metin turetmek/olustumak
bunun için h-gram dil modelini kullanıcaz

example: I ...(love)
"""
# verileri token haline getir

tokens=[word_tokenize(sentence.lower()) for sentence in corpus]


# bigramm2 li kelime gurupları oluştur
bigrams=[]
for token_list in tokens:
    bigrams.extend(list(ngrams(token_list,2)))

bigrams_freq=Counter(bigrams)

# trigram

trigrams=[]

for token_list in tokens:
    trigrams.extend(list(ngrams(token_list,3)))
    
trigrams_freq= Counter(trigrams)


# model testing

# ı love bigrams dan sonra you veya apple gelme olasılığı hesaplayalım

bigram=("i","love") # hedef bigram 

# ı love you olma olasılığı

prob_you=trigrams_freq[("i","love","you")]/bigrams_freq[bigram]

print(f"you kelimesinin olma olasılığı: {prob_you}")

# ı love apple olma olasılığı

prob_apple=trigrams_freq[("i","love","apple")]/bigrams_freq[bigram]

print(f"apple kelimesinin olma olasılığı: {prob_apple}")

