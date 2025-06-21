"""
metin üretimi

gpt-2 metin üretimi calismasi

"""

# import libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM # llama


# modelin tanimlanmasi
model_name="gpt2"
model_name_llama="huggyLLama/LLama-7b" # llama


# tokenizer tanimlama ve model olusturma
tokenizer=GPT2Tokenizer.from_pretrained(model_name)
tokenizer_llama=AutoTokenizer.from_pretrained(model_name_llama)


model=GPT2LMHeadModel.from_pretrained(model_name)
model_llama=AutoModelForCausalLM.from_pretrained(model_name_llama) # llama

# metin üretimi için gerekli olana baslangic text i
text="Afternoon, "

# tokenization
inputs=tokenizer.encode(text, return_tensors="pt")
inputs_llama=tokenizer_llama(text, return_tensors="pt") # llama

# metin üretimi gerçeklestirelim

outputs=model.generate(inputs, max_length=55) #☼ inputs =modelin baslangic noktasi, max_length=max token(sözcük) sayısı
outputs_llama=model_llama.generate(inputs_llama.input_ids, max_length=55) # llama


# modelin ürettiği tokenleri okunabilir hale getirmemiz lazım

generation_text=tokenizer.decode(outputs[0], skip_special_tokens=True) # skip_special_tokens= ozel tokenleri ( orn: cumle baslangic bitis tokenleri) metinden çıkart
generation_text_llama=tokenizer_llama.decode(outputs_llama[0], skip_special_tokens=True) # llama

# üretilen metni print ettirelim

print(generation_text)
print(generation_text_llama)

 


