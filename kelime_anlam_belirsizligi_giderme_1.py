import nltk
from nltk.wsd import lesk


# gerekli nltk paketlerini indirelim

nltk.download("wordnet")
nltk.download("own1.4")
nltk.download("punkt")

# ilk cumle

s1="I go to the bank to deposit money"
w1= "bank"

sense1=lesk(nltk.word_tokenize(s1),w1)
print("Cumle: {s1}")
print("word: {w1}")
print(f"Sense: {sense1.definition()}")


s2="The river bank is flooded after the heavy rain"
w2="bank"
sense2=lesk(nltk.word_tokenize(s2),w2)
print("Cumle: {s2}")
print("word: {w2}")
print(f"Sense: {sense2.definition()}")
