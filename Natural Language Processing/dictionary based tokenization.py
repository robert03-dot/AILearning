##
# Steps needed for implementing Dictionary-based tokenization:
# Step 1:Collect a dictionary of words and their corresponding parts of
# speech.The dictionary can be created manually or obtained from a
# pre-existing source such as WordNet or Wikipedia.
# Step 2:Preprocess the text by removing any noise such as punctuation
# marks,stop words and HTML tags.
# Step 3:Tokenize the text into words using a whitespace tokenizer or a
# sentence tokenizer.
# Step 4:Identify the parts of speech of each word in the text using a
# part-of-speech tagger such as the Stanford POS Tagger.
# Step 5:Segment the text into tokens by comparing each word in the text with
# the words in the dictionary.If a match is found,the corresponding word
# in the dictionary is used as a token.Otherwise,the word is split into
# smaller sub-tokens based in its parts of speech
from nltk import MWETokenizer, word_tokenize

# For example, consider the following sentence:
#
# Jammu Kashmir is an integral part of India.
# My name is Pawan Kumar Gunjan.
# He is from Himachal Pradesh.#

dictionary = [("Jammu", "Kashmir"),
              ("Pawan", "Kumar", "Gunjan"),
              ("Himachal", "Pradesh")]
dictionary_tokenizer = MWETokenizer(dictionary,separator=' ')
text = """ 
Jammu Kashmir is an integral part of India. 
My name is Pawan Kumar Gunjan. 
He is from Himachal Pradesh. 
"""
tokens = word_tokenize(text)
print(tokens)

dictionary_based_token = dictionary_tokenizer.tokenize(tokens)
print(dictionary_based_token)