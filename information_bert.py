# import library
from transformers import BertTokenizer, BertModel

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# tokenizer and model create
model_name="bert-base-uncased" # kucuk boyurlu modeli
tokenizer= BertTokenizer.from_pretrained(model_name)
model=BertModel.from_pretrained(model_name)



# veri olustur: karşılaştırılacak belgeleeri ve sorgu cumlenizi olustur

documents=[
    "machine learning is a field of artificial intelligence",
    "Natural language processing involves human language",
    "Artificial intelligence encomppases machine learning and natural language processing (nlp)",
    "Deep learning is a subset of machine learning",
    "Data scence combines statictic, data analysisi and machine learning",
    "I go to shop"
    ]

query="What is deep learning?"



# bert ile bilgi getirme

def get_embedding(text):
    #tokenization
    inputs=tokenizer(text,return_tensors="pt",truncation=True, padding=True)
    
    # modeli çalıştır
    outputs=model(**inputs)
    
    # songizli katman alalım
    
    last_hidden_state=outputs.last_hidden_state
    
    # metin temsili olustur
    embedding=last_hidden_state.mean(dim=1)
    
    # vektoru numpy olarak return et
    return embedding.detach().numpy()

# belge ve sorgu vektörlerini al

doc_embeddings=np.vstack([get_embedding(doc) for doc in documents])
query_embedding=get_embedding(query)

# kosinüs benzerliği ile belgeler arasında benzerliği hesapalyalım

similirities=cosine_similarity(query_embedding,doc_embeddings)


# ger belgenin benzerlik skoru

for i, score in enumerate(similirities[0]):
    print(f"{documents[i]} : {score}")