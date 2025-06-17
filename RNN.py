"""
solve classificationproblem (Sentiment Analysisi in NLP) with RNN (Deep Learning based language model)
duygu analizi -> bir cümlenin etiketlenmesi (positive ve negative)
reatautanr yorumları değerlendirme

"""


# import libraries

import pandas as pd
import numpy as np
from gensim.models import Word2Vec # metin temsili
from keras.preprocessing.sequence import pad_sequences # Amaç: Dizi uzunluklarını eşitlemek.
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# create dataset
data = {
    "text": [
        "Yemekler çok lezzetliydi, tekrar geleceğim.",
        "Garsonlar çok ilgisizdi, memnun kalmadım.",
        "Tatlı olarak baklava yedim, harikaydı.",
        "Servis çok yavaştı, buz gibi yemek geldi.",
        "Ortam çok ferah ve huzurluydu.",
        "Siparişim yanlış geldi, sinirlendim.",
        "Çorba sıcaktı ve tadı harikaydı.",
        "Çorba tuzdan içilmiyordu, berbattı.",
        "Müzikler çok güzeldi, ortam mükemmeldi.",
        "Masalar pis ve düzensizdi.",
        "Fiyatlar uygundu, kalite şaşırtıcıydı.",
        "Garsonlar kaba ve ilgisizdi.",
        "Tatlı olarak sütlaç yedim, mükemmeldi.",
        "Pizza çok tuzluydu, yiyemedim.",
        "Makarnalar tam kıvamında pişmişti.",
        "Tatlılar bayattı, hiç beğenmedim.",
        "Müzikler çok keyifliydi.",
        "Yemekler soğuk ve tatsızdı.",
        "Tatlılar çok taze ve güzeldi.",
        "Tatlı çok kötüydü, yiyemedim bile.",
        "Sunum çok şıktı, lezzet de yerindeydi.",
        "Yemek çok geç geldi, soğumuştu.",
        "Tatlı efsaneydi, ellerinize sağlık.",
        "Servis berbattı, tekrar gelmem.",
        "Yemekler çok hızlı servis edildi.",
        "Garsonlar kayıptı, servis çok yavaş.",
        "Tatlı olarak kazandibi yedim, bayıldım.",
        "Masalar kirliydi, hijyen eksikti.",
        "Tatlıdan sonra kahve ikramı harikaydı.",
        "Yemek çok yağlıydı, midemi bozdu.",
        "Tatlılar harikaydı, porsiyonlar büyüktü.",
        "Servis geç ve ilgisizdi.",
        "Tatlı nefisti, tekrar sipariş ederim.",
        "Makarna hamurdu, hiç güzel değildi.",
        "Garson çok ilgiliydi, ne istersek getirdi.",
        "Masalar düzensizdi ve yerler yapışıktı.",
        "Ortam çok sakindi, keyif aldım.",
        "Lahmacun çok pişmişti, tadı kaçmıştı.",
        "Pizza tam kıvamındaydı, sıcak geldi.",
        "Fiyat performans açısından kötüydü.",
        "Sunum muazzamdı, lezzet de aynı şekilde.",
        "Çorba çok acıydı, içemedim.",
        "Tatlı olarak tiramisu yedim, çok iyiydi.",
        "Yemeklerin porsiyonu çok küçüktü.",
        "Tatlı ikramı bizi çok mutlu etti.",
        "Tatlılar donmuştu, tad alamadım.",
        "Garson çok nazikti, teşekkür ettik.",
        "Menü çok sınırlıydı, çeşit azdı.",
        "Tatlı çok hafifti, bayıldım.",
        "Garsonlar ilgisizdi, siparişler karıştı."
    ],
    "label": [
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative"
    ]
}

df=pd.DataFrame(data)



# %% metin temizleme ve preprocessing: tokenize, label encoding, train test_split

# tokenization
tokenizer=Tokenizer()
tokenizer.fit_on_texts(df["text"])
sequence=tokenizer.texts_to_sequences(df["text"])
word_index=tokenizer.word_index # sayısallaştırlıran kelimelerin hangi sayıyıa denk geldiğini gösteriyor

# padding proces
maxlen=max(len(seq) for seq in sequence)
x=pad_sequences(sequence, maxlen=maxlen)
print(x.shape)

# label encoding

label_Encoder=LabelEncoder()
y=label_Encoder.fit_transform(df["label"])

# train test splir

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2,random_state=21)

# %% metin temsili: word embedding :word2vec

sentences=[text.split() for text in df["text"]]
word2vec_model=Word2Vec(sentences,vector_size=50,window=5,min_count=1) # Uyarı

embedding_dim=50
embedding_matrix=np.zeros((len(word_index)+1,embedding_dim))
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i]=word2vec_model.wv[word]


# %% modelling: build, train ve test rnn modeli

#build model
model=Sequential()

# embedding
model.add(Embedding(input_dim=len(word_index)+1,output_dim=embedding_dim,weights=[embedding_matrix],input_length=maxlen,trainable=False))

# RNN layer
model.add(SimpleRNN(50,return_sequences=False))

# output layer
model.add(Dense(1,activation="sigmoid"))

# compile model

model.compile(optimizer="adam", loss="binary_crossentropy",metrics=["accuracy"])

#train model

model.fit(x_train, y_train, epochs=10, batch_size=2, validation_data=(x_test,y_test))


# evaluate rnn model

loss, accuracy = model.evaluate(x_test,y_test)

print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")


# cule siniflandirma çalışması

def classify_sentence(sentence):
    
    seq=tokenizer.texts_to_sequences([sentence])
    padded_seq=pad_sequences(seq,maxlen=maxlen)
    
    prediction=model.predict(padded_seq)
    predicred_class=(prediction >0.5).astype(int)
    label="positive" if predicred_class[0][0]==1 else "negative"
    return label

sentence= "Servis geç geldi"

result=classify_sentence(sentence)

print(f"Result: {result}")
