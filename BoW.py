# count vectorizer içeriye aktar

from sklearn.feature_extraction.text import CountVectorizer


# veri setini oluştur
documents=["kedi bahçede","kedi evde"]


# vectorize taimla
vectorize=CountVectorizer()


# metni sayısal vektorlere çevir
X=vectorize.fit_transform(documents)

# kelime kümesi olusturma
feature_names=vectorize.get_feature_names_out() #kelime kümesini oluştrma
print(f"kelime kümesi: {feature_names}")
#vector temsili

vector_temsili=X.toarray()

print(f"kelime kümesi: {vector_temsili}")

