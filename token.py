import nltk
nltk.download('punkt')
nltk.download('punkt_tab') # Add this line to download the missing resource
from nltk.tokenize import word_tokenize

sentence = "Machine learning is fun!"
tokens = word_tokenize(sentence)

print("Tokens:", tokens)
