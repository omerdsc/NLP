# import library

from sklearn.feature_extraction.text import CountVectorizer

# ornek metin
document=[
    "Bu çalışma NGram çalışmasıdır.",
    "Bu çalışma doğal dil işleme çalışmasıdır."]

# unigram, bigram, trigram şeklinde 3 farkli N degerinde sahip gram modeli
vectorize_unigram=CountVectorizer(ngram_range=(1,1))
vectorize_bigram=CountVectorizer(ngram_range=(2,2))
vectorize_trigram=CountVectorizer(ngram_range=(3,3))

# unigram
x_unigram=vectorize_unigram.fit_transform(document)
unigram_features=vectorize_unigram.get_feature_names_out()

# bigram
x_bigram=vectorize_bigram.fit_transform(document)
bigram_features=vectorize_bigram.get_feature_names_out()

# bigram
x_trigram=vectorize_trigram.fit_transform(document)
trigram_features=vectorize_trigram.get_feature_names_out()


