from transformers import BertTokenizer, BertForQuestionAnswering
import torch

import warnings
warnings.filterwarnings("ignore")

model_name= "bert-large-uncased-whole-word-masking-finetuned-squad"

# bert tokenizer

tokenizer=BertTokenizer.from_pretrained(model_name)

# soru cevaplama gorevi için ince ayar yapilmis bert modeli
model=BertForQuestionAnswering.from_pretrained(model_name)


# cevapları tahmin eden fonksiyon

def predict_answer(context, question):
    
    """
        context= metin
        question= soru
        Amac: metin içerisinden soruyu bulmak
        
        1) tokenize
        2) metnin içerisinde soruyu ara
        3) metnin içerisinde sorunun cevabının nerede olabileceğini skorlarini return et
        4) skorlardan tokenlerin indexleri hesapladık
        5) tokenleri bulduk yani cevabı
        6) okunabilir olması için tokenlerdan stringe çevirdik
    """
    
    # metni ve soruyu tokenlara ayiralım ve modele uygun hale getirelim
    encoding=tokenizer.encode_plus(question,context,return_tensors="pt", max_length=512, truncation=True)
    
    # giris tensorlerini hazırla
    input_ids=encoding["input_ids"] # tokenleri id
    attention_mask= encoding["attention_mask"] # hangi tokenlerin dikkate alinacagini belirtir
    
    # modeli çalıştır ve skorları hesapla
    with torch.no_grad(): # gradyanların hesaplanmasını devre dışı bırakır buda hızlı olmasını sağlar
        start_scores, end_scores=model(input_ids, attention_mask, return_dict=False)
        
    # en yüksek olasılığa sahip start ve end indekslerini hesapliyor
    
    start_index=torch.argmax(start_scores, dim=1).item() # baslangic indes
    end_index= torch.argmax(end_scores, dim=1).item() # bitis indekslerimiz
    
    # token id lerini kullanarak cevap metnini elde edelim
    answer_tokens=tokenizer.convert_ids_to_tokens(input_ids[0][start_index: end_index+1])
    
    # tokenleri birlestir ve okunabilir hale getir
    answer= tokenizer.convert_tokens_to_string(answer_tokens)
    
    return answer


question="what is the capital ol france"
context="France, officiality the french Republic, is a country whose capital is Paris"


answer=predict_answer(context,question)
    
    
    
    
    
    