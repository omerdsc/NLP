"""
spam veri seti -> spam ve ham -> binary classification with desicion tree
"""

# import libraries
import pandas as pd


# veri setini yükle 

data=pd.read_csv("spam.csv", encoding="latin-1")

data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1, inplace=True)
data.columns=["label","text"]




# EDA: Keşifsel veri analizi: missing value
print(data.isnull().sum())
print(data.duplicated().sum())


# %% text cleaning and processing: ozel karektereli , lowercase, tokenization, stopwords, lemmatize

import nltk

nltk.download("stopwords") 
nltk.download("wordnet") # lemma bulmak için gerekli olan veriseti
nltk.download("omw-1.4") # wordanet e ait farklı dillerin kelime anlamlarini içeren bir veri seti

import re
from nltk.corpus import stopwords # stopwordslerden kurtulmak için
from nltk.stem import WordNetLemmatizer

text=list(data.text)

lemmatizer=WordNetLemmatizer()
corpus=[]

for i in range(len(text)):
    r=re.sub("[^a-zA-Z]", " ", text[i]) # metiniçerisinde harf olmayan tüm karekterlerden kurtul
    r=r.lower() # buyuk harfleri küçük harfe çevirin
    r=r.split() # kelimeleri ayır
    
    r=[word for word in r if word not in stopwords.words("english")] # stopwordslerden kurtul
    
    r=[lemmatizer.lemmatize(word) for word in r]
    
    r= " ".join(r)
    
    corpus.append(r)
    
data["text2"]=corpus

# %% model training and evaluation

x=data["text2"]
y=data["label"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.2,random_state=21)

# feature extraciton: bag of words

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train_cv=cv.fit_transform(x_train)

# classifier training: model training and evaluation

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
dt.fit(x_train_cv,y_train)

x_test_cv=cv.transform(x_test)

#prediction

prediction=dt.predict(x_test_cv)


from sklearn.metrics import confusion_matrix

c_matrix=confusion_matrix(y_test,prediction)

score=dt.score(x_test_cv, y_test)


