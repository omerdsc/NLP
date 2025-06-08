import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Stopwords'i bir kez indir
nltk.download("stopwords")

# veri setinin içeriye aktarılması
df = pd.read_csv("IMDB Dataset.csv")

# metin verilerini alalım
documents = df["review"]
labels = df["sentiment"]  # positive veya negative

# Stopword listesi bir kez alınır
stop_words_eng = set(stopwords.words("english"))

def clean_text(text):
    # Küçük harfe çevir
    text = text.lower()
    
    # Rakamları temizle
    text = re.sub(r"\d+", "", text)
    
    # Özel karakterleri kaldır
    text = re.sub(r"[^\w\s]", "", text)
    
    # Kısa kelimeleri temizle
    words = [word for word in text.split() if len(word) > 2]
    
    # Stopword'leri kaldır
    words = [word for word in words if word not in stop_words_eng]
    
    return " ".join(words)

# metinleri temizle
cleaned_doc = [clean_text(row) for row in documents]



# %% bow
vectorizer=CountVectorizer()

# metin -> sayisal hale getir
x=vectorizer.fit_transform((cleaned_doc[:75])) # çok uzun süreceğinden başlangıç için ilk 75 taneyi yapıyoruz


# kelime kumesi göster
feature_names=vectorizer.get_feature_names_out()

# vektor temisli goster

vector_temisil=x.toarray()

# kelime frekanslarını goster
df_bow=pd.DataFrame(vector_temisil,columns=feature_names)

word_count=x.sum(axis=0).A1
word_freq=dict(zip(feature_names,word_count))

#ilk 5 kelimeyi print ettir
most_common_5_words= Counter(word_freq).most_common(5)
print(f"most_common_5_words: {most_common_5_words}")