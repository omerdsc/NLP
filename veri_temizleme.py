
# metinlerde bulunan fazla boşlukları ortadan kaldır

text="Hello,    world!    2025"

cleanend_text1=" ".join(text.split())

print(f"text: {text} ")
print(cleanend_text1)



# %% buyuk harfleri küçük harflere cevirme
text="Hello, world! 2025"

cleanend_text2=text.lower() #kucuk harfe çevir
print(f"text: {text} ")
print(cleanend_text2)


# %% noktamalama isaretlerini kaldir
import string

text="Hello, world! 2025"

#♥translete(str.maketrans("değiştirilecek karekter","yeni karekter","silinecek karekterler"))

cleanend_text3=text.translate(str.maketrans("","",string.punctuation))

print(f"text: {text} ")
print(cleanend_text3)


# %% ozel karekterleri kaldir
import re

text="Hello, world! 2025 @, /,*,#"


cleanend_text4=re.sub(r"[^A-Za-z0-9\s]","",text)

print(f"text: {text} ")
print(cleanend_text4)


# %% yazım hatalarini duzelt

from textblob import TextBlob # metin analizlerinde kullanılan bir kutuphane

text="Hellio wirld! 2035"

cleanend_text5=TextBlob(text).correct() # correct : yazim hatalarını uzeltir

print(f"text: {text} ")
print(cleanend_text5)


# %% html yada url etiketlreini kaldir

from bs4 import BeautifulSoup

html_text="<div>Hello, World! 2035</div>"

cleanend_text6=BeautifulSoup(html_text,"html.parser").get_text()
print(f"text: {html_text} ")
print(cleanend_text6)