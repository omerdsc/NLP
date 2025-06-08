import nltk

from nltk.corpus import stopwords

nltk.download("stopwords") # farklı dillerde en çok kullanılan stop words iceren veri eti

# ingilizce stop words analizi nltk

stop_words_eng=set(stopwords.words("english"))

# ornek ingilizce metin

text="there are same example of handling stop words from same texts."

text_list=text.split()
filtered_words=[word for word in text_list if word.lower() not in stop_words_eng]

print(f"filtered_words: {filtered_words}")
# turkce stopwords analizi nltk
stop_words_tr=set(stopwords.words("turkish"))

metin=" merhaba arkdaslar çok güzel bir ders işliyoruz. Bu ders faydalı mı"

metin_list=metin.split()

filtered_words_tr=[word for word in metin_list if word.lower() not in stop_words_tr]

print(f"filtered_words: {filtered_words_tr}")
# kutuphanesiz stop words cikarimi


# %% kutuphaneisz stop word cikartimi

tr_stopwords=["için","bu","ile","mu","mi","özel"]


metin="BU bir denemedir. Amacamız bu metinde bulunan özel karekterleri elemekmi acaba"
metin_list=metin.split()

filtered_words_tr2=[word for word in metin_list if word.lower() not in tr_stopwords]
