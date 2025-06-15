# import librares
from transformers import AutoTokenizer, AutoModel
import torch



# model ve tokenizasyon

model_name="bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModel.from_pretrained(model_name)

# input text (metin) tanimla
text="Trasnformers can be used for natura Language processing"


# metni tokenlara çevirmek
imputs=tokenizer(text, return_tensors="pt") # çıktı pytorch tensoru olarak return edilir


# modeli kullanarak metin temisli oluştur

with torch.no_grad(): # gradyanların hesapnalamsı durdurulur, böylece belleği daha verimli kullanırız
    output=model(**imputs)
    


# modelin çıktısından son gizli durumu alalım
last_hidden_state= output.last_hidden_state  # tüm yoken çıktılarını alamk için


# ilk tokenim embeddingini alalım ve print ettirelim

first_token_embedding=last_hidden_state[0,0:].numpy()

print(f"metin temsili: {first_token_embedding}")




