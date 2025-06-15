"""
word2vec (google
          fasttext (facebook- meta))
"""

# import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA # principle component analysis: Noyut indirgeme
    
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess # metinden büyük küçük harfleri çıkartır ve tokenizasyon işlemini yapar

# ornek veri seti olustur

sentences=["Köpek çok tatalı bir hayvandır.",
           "Köpekler evil hayvanlardır.",
           "KEdiler genellikle bağımsız hareket etmeyi severler.",
           "Köpekler sadık ve dost canlısı hayvanlardır.",
           "Hayvanlar insanlar için iyi arkadaşlardır."
           ]

tokenized_sentences=[simple_preprocess(sentence) for sentence in sentences]

# word2vec
word2_vec_model=Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)


# fasttext
fasttext_model=FastText(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1, sg=0)


# gorsellestirme: PCA

def plot_word_embedding(model,title):
    
    word_vector=model.wv
    
    words=list(word_vector.index_to_key)[:1000]
    vectors= [word_vector[word] for word in words]
    
    #PCA
    pca=PCA(n_components=3)
    reduced_vectors=pca.fit_transform(vectors)
    
    
    #♥3d gorsellestirme
    fig=plt.figure(figsize=(12,10))
    ax=fig.add_subplot(111, projection="3d")
    
    # vektorleri çiz
    ax.scatter(reduced_vectors[:,1], reduced_vectors[:,2])
    
    # kelimeleri etketle
    for i, word in enumerate((words)):
        ax.text(reduced_vectors[i,0],reduced_vectors[i,1],reduced_vectors[i,2],word,fontsize=12)
        
    ax.set_title(title)
    ax.set_xlabel("componenet 1")
    ax.set_ylabel("componenet 2")
    ax.set_zlabel("componenet 3")
    plt.show()
    
plot_word_embedding(word2_vec_model, "Word2vec")
plot_word_embedding(fasttext_model, "FastTExt")
