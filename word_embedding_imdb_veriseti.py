# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# veri seti yükleme


df=pd.read_csv("IMDB Dataset.csv")

documents=df["review"]

# metim temizleme 

def clean_text(text):
    
    text=text.lower() # kucuk harf
    text=re.sub(r"\d+","",text) # sayilari temizle
    text=re.sub(r"[^\w\s]","",text) # özel karekterleri temizle
    text=" ".join([word for word in text.split() if len(word)>2 ])
    
    return text

cleaned_documents = [clean_text(doc) for doc in documents]

# metin tokenization

tokenized_documnet=[simple_preprocess(doc) for doc in cleaned_documents]



# %%


# word2vec model tanimi
model=Word2Vec(sentences=tokenized_documnet,vector_size=50, window=5, min_count=1,sg=0)
word_vectors=model.wv

words=list(word_vectors.index_to_key)[:500]

vectors=[word_vectors[word] for word in words]

#clustring KMeans k=2

kmenas=KMeans(n_clusters=2)
kmenas.fit(vectors)
clusters=kmenas.labels_



# PCA 50 -> 2

pca=PCA(n_components=2)

reduce_vectors=pca.fit_transform(vectors)


# 2 boyutlu bir görselleştirme

plt.figure()
plt.scatter(reduce_vectors[:,0],reduce_vectors[:,1],c=clusters, cmap="viridis")

centers=pca.transform(kmenas.cluster_centers_)
plt.scatter(centers[:,0], centers[:,1], c="red",marker="X",s=130, label="center")
plt.show()