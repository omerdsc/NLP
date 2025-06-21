"""
metin üretimi
lstm train with test data
text data = gpt ile olustur
"""

# import libraries 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



# eğitim verisi chatgpr ile olustur

texts = [
    "Bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum.",
    "Kitap okumak beni gerçekten mutlu ediyor.",
    "Sabah kahvemi içmeden güne başlayamıyorum.",
    "Arkadaşlarımla sinemaya gitmek için sabırsızlanıyorum.",
    "Yarınki sınav için biraz daha çalışmam gerekiyor.",
    "Müzik dinlemek ruh halimi hemen değiştiriyor.",
    "Bu hafta sonu ailemi ziyaret etmeyi planlıyorum.",
    "Yeni bir diziye başladım, konusu çok sürükleyici.",
    "Evde bitki yetiştirmek bana huzur veriyor.",
    "Akşam yemeğinde ne pişirsem diye düşünüyorum.",
    "Yürüyüş yaparken podcast dinlemeyi seviyorum.",
    "Yeni tarifler denemek mutfağı daha eğlenceli hale getiriyor.",
    "İşe erken gitmek günümü daha verimli kılıyor.",
    "Film geceleri arkadaşlarla daha keyifli oluyor.",
    "Sıcak bir çay eşliğinde kitap okumak harika.",
    "Bilgisayar oyunu oynamak bazen tüm stresimi alıyor.",
    "Güne meditasyonla başlamak zihnimi temizliyor.",
    "Hafta içi yoğun geçse de hafta sonları dinleniyorum.",
    "Yeni bir hobi edinmek beni heyecanlandırıyor.",
    "Günlük tutmak duygularımı anlamama yardımcı oluyor.",
    "Sabahları yürüyüş yapınca tüm gün enerjik hissediyorum.",
    "Kahve kokusu bile beni mutlu etmeye yetiyor.",
    "Bugün güneşli hava sayesinde moralim yerine geldi.",
    "Yatağa erken gitmek istiyorum ama hep geç kalıyorum.",
    "Sevdiğim müziği duyunca tüm yorgunluğum geçiyor.",
    "Yarın için plan yapmayı çok seviyorum.",
    "Yeni yerler keşfetmek beni her zaman heyecanlandırıyor.",
    "Evde vakit geçirmek bazen dışarı çıkmaktan daha keyifli.",
    "Pencereden dışarı bakmak bile iyi geliyor.",
    "Kitapçıda saatlerce zaman geçirebilirim.",
    "Kendi yemeğimi yapınca ayrı bir tat alıyorum.",
    "Sessiz bir ortamda çalışmak verimliliğimi artırıyor.",
    "Sabah sporunu ihmal ettiğimde kendimi kötü hissediyorum.",
    "Yalnız kalmak bazen çok iyi geliyor.",
    "Sıcak bir battaniyeye sarınıp film izlemek harika.",
    "Hafta sonu planları yaparken bile mutlu oluyorum.",
    "Küçük şeylerle mutlu olmayı öğreniyorum.",
    "Sevdiğim insanlarla vakit geçirmek en büyük lüksüm.",
    "Kedilerle oynamak günümün en güzel anı olabilir.",
    "Kahvaltı en sevdiğim öğündür.",
    "Yeni kitaplar almak bana terapi gibi geliyor.",
    "Baharda yürüyüş yapmanın tadı bambaşka.",
    "Bir günlüğüne telefonumu kapatmak istiyorum.",
    "Bugün biraz tembellik yapmaya ihtiyacım var.",
    "Çalışma masamı düzenleyince motive oluyorum.",
    "Yeni müzik listeleri hazırlamak beni heyecanlandırıyor.",
    "Anılarla dolu bir fotoğraf albümüne bakmak içimi ısıtıyor.",
    "Bazen sessizlik en iyi arkadaştır.",
    "Bir fincan bitki çayıyla günü kapatmak huzur verici.",
    "Kendime zaman ayırmak çok kıymetliymiş.",
    "Bugün kendimi ödüllendireceğim bir tatlıyla.",
    "Güne güzel bir notla başlamak harika olurdu.",
    "Yağmur sesini dinleyerek uyumayı çok seviyorum.",
    "Uzun zamandır ertelediğim işleri nihayet yapmaya başladım.",
    "Yeni hedefler belirlemek bana umut veriyor.",
    "Güzel anıları hatırlamak ruhuma iyi geliyor.",
    "Evde mum yakmak ortama sıcaklık katıyor.",
    "Sıcak çorba içmek bile bazen yeterlidir mutlu olmak için.",
    "İyi bir uyku sonrası her şey daha kolay geliyor.",
    "Hayal kurmak için her zaman biraz zaman ayırırım.",
    "Eski müzikleri dinleyip anılara dalmak hoşuma gidiyor.",
    "Bugün üretken bir gün geçirdim, çok memnunum.",
    "Pijamalarla bütün gün evde olmak güzel hissettiriyor.",
    "Yalnızca kendimle vakit geçirmek bana iyi geliyor.",
    "Küçük notlar yazıp etrafa yapıştırmak hoşuma gidiyor.",
    "Sabah güneşini penceremden izlemek çok keyifli.",
    "Yeni bir şey öğrendiğimde kendimle gurur duyuyorum.",
    "Taze kahve kokusu beni güne hazırlıyor.",
    "Renkli defterlere not almak çalışmayı eğlenceli kılıyor.",
    "Hayat bazen durup sadece nefes almak kadar basit.",
    "Kendime güzel bir playlist hazırladım, ruhum dans ediyor.",
    "Bugün aynaya bakıp kendime gülümsedim.",
    "Doğada yürüyüş yapmak en iyi terapi.",
    "Kendi sınırlarımı aşmak bana güç veriyor.",
    "Evde dans etmek moralimi yerine getiriyor.",
    "Bazen sadece oturup düşünmek bile yeterli oluyor.",
    "Sevdiğim dizinin yeni bölümü çıkmış, akşam planım belli.",
    "Sade bir gün geçirdim ama çok huzurluydu.",
    "Bugün küçük şeylere şükretmeyi seçtim.",
    "Güzel haberler aldım, kalbim kıpır kıpır.",
    "Kendi kendime vakit geçirmek özgür hissettiriyor.",
    "Plan yapmadan bir gün geçirmek iyi gelebiliyor.",
    "Gün batımını izlemek en sevdiğim anlardan biri.",
    "Bugün biraz yavaşlamak istiyorum.",
    "Kendime bir çiçek aldım, odama neşe kattı.",
    "En son neye güldüğümü hatırlayıp yine güldüm.",
    "Kendi hikayemi yazmak için bugün güzel bir gün.",
    "İçimden geldiği gibi yaşamak istiyorum.",
    "Yeni bir dil öğrenmeye karar verdim, çok heyecanlıyım.",
    "Bugün sade ama anlamlı bir gün geçirdim.",
    "Sevdiğim bir filmi tekrar izlemek iyi geldi.",
    "Küçük bir yürüyüş bile zihnimi açıyor.",
    "Bir fincan çay, bir güzel söz, hepsi yeter.",
    "Güzel bir rüya gördüm, hala etkisindeyim.",
    "Bazen her şeyden uzaklaşmak istiyor insan.",
    "Kendi halimde mutluyum bugün.",
    "Bu sabah erkenden uyanmak iyi geldi.",
    "Bugün telefonuma daha az baktım ve daha huzurluydum.",
    "Bir şeyi başarmak beni gururlandırıyor.",
    "Sadeleşmek istiyorum, hem zihnen hem bedenen.",
    "Bugün içim kıpır kıpır, nedensiz bir mutlulukla doluyum.",
    "Yeni başlangıçlar için harika bir gün.",
    "Kendimi sevmeyi öğreniyorum, yavaş yavaş ama sağlam adımlarla."
]


# %% metin temizleme ve preprocessing: tokenization, padding,label encoding

# tokenization

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts) #metinler üzerindeki kelime frekansını öğren fit et
total_words=len(tokenizer.word_index)+1 # toplam kelime sayısı

# n-gram dizileri olustur ve padding uygula
input_sequences=[]

for text in texts:
    # metinleri kelime indexlerine çevir
    token_list=tokenizer.texts_to_sequences([text])[0]
    
    # her metin için n-gram dizisi olusturalım
    for i in range(1,len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        
# en uzun diziyi bulalım, tüm dizileri aynı uzunluğa getirelim

max_sequnece_length= max(len(x) for x in input_sequences)

# dzilere padding işlemi uygula, hepsinin ayni uzunlukta olmasini sağla
input_sequences=pad_sequences(input_sequences,maxlen=max_sequnece_length, padding="pre")

# x ve y 

x=input_sequences[:,:-1]
y=input_sequences[:,-1]

y=tf.keras.utils.to_categorical(y,num_classes=total_words) # one hot encoding




# %% LSTM modeli olustur, compile, train  vve evaluate

model=Sequential()

# embedding
model.add(Embedding(total_words, 50 , input_length=x.shape[1]))

# lstm
model.add(LSTM(100, return_sequences=False))


#output
model.add(Dense(total_words,activation="softmax"))

# model compile

model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])

# model training
model.fit(x,y,epochs=100,verbose=1)



# %% model prediction

def generate_text(seed_text, next_words):
    for _ in range(next_words):
        
        # girdi metnini sayisal verilere dönüştür
        token_list=tokenizer.texts_to_sequences([seed_text])[0]
        
        # padding
        token_list=pad_sequences([token_list],maxlen=max_sequnece_length-1,padding="pre")
        
        #prediction
        prediction_probabilities = model.predict(token_list, verbose=0)
        
        # en tüksek olasiliğa sahip kelimenin indexini bul
        predicted_word_index= np.argmax(prediction_probabilities,axis=-1)
        
        #tokenizer ile keilme indexinden asil kelime bulunur
        predicted_word=tokenizer.index_word[predicted_word_index[0]]
        
        # tahmin edilen kelimeyi seed_text e ekleyelim
        
        seed_text=seed_text + " "+ predicted_word
    
    return seed_text

seed_text="Bu hafta sonu"

print(generate_text(seed_text, 4))
    