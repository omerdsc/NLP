""" 
Part of Spech POS: kelimelerin uygun sözcük türünü bulma calismasi
HMM

I(Zamir) am a teacher(isim)

"""

# import libraies
import nltk
from nltk.tag import hmm

# örnek training data tanimla

train_data=[
    [("I","PRP"),("am","VBP"),("a","DT"),("teacher","NN")],
    [("You","PRP"),("are","VBP"),("a","DT"),("student","NN")]]
    

# train HMM
trainer=hmm.HiddenMarkovModelTrainer()
hmm_tagger=trainer.train(train_data)


# yeni bir cümle oluştur ve cümlenin içerisinde bulunan her bir sözcüğün türünü etiketle

test_sentence= "I am a student".split()

tags=hmm_tagger.tag(test_sentence)

print(f"Yeni Cumle: {tags}")