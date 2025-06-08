import nltk #natural language tool kit

nltk.download("punkt_tab") # metni kelime ve cümle bazında tokenlara ayıra bilmek için gerekli

text="Hello, world! How are you? Hello, hi ..."

#kelime tokenizasyonu: word_tokenize: metni kelimelere ayirir, noktalam işaretleri 
 #ve boşluklar ayri birer token olarak elde edilecektir"""

word_tokens=nltk.word_tokenize(text)

#cumle tokenizasyonu: sent_tokenize: metni cumlelere ayirir. her ir cumle birer token olur.

sentence_tokens=nltk.sent_tokenize(text)



