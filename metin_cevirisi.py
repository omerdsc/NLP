from transformers import MarianMTModel, MarianTokenizer


model_name="Helsinki-NLP/opus-mt-en-fr"


tokenizer=MarianTokenizer.from_pretrained(model_name)
model=MarianMTModel.from_pretrained(model_name)


text="hello, what is your name"

# encode edelim, sonrasinda modele input olarak verelim

translated_text= model.generate(**tokenizer(text, return_tensors="pt",padding=True))

# translated text metne dönüştürülür

translated_text=tokenizer.decode(translated_text[0], skip_specal_tokens=True)
print(f"Translated text: {translated_text}")