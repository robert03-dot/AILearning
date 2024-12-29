import nltk
from nltk import TweetTokenizer, WhitespaceTokenizer, TabTokenizer, SpaceTokenizer, SExprTokenizer, MWETokenizer, \
    LineTokenizer, ConditionalFreqDist, RegexpTokenizer, WordPunctTokenizer, SyllableTokenizer, PunktSentenceTokenizer
from nltk.corpus import webtext, stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

##
# Corpus - Body of text,singular.Corpora is the plural of this
# Lexicon - Words and their meanings
# Token - Each "entity" that is a part of whatever was split up based on rules.For examples,each word
# is a token when a sentence is "tokenized" into words.Each sentence can also be a token,if
# you tokenized the sentences out of a paragraph
# So basically tokenizing involves splitting sentences and words from the body of the text.#

text = ("Natural language processing (NLP) is a field of computer science, "
        "artificial intelligence and computational linguistics concerned with"
        " the interactions between computers and human (natural) languages, and,"
        "in particular, concerned with programming computers to fruitfully process"
        " large natural language corpora. Challenges in natural language processing"
        " frequently involve natural language understanding, natural language generation"
        " (frequently from formal, machine-readable logical forms), connecting language"
        " and machine perception, managing human-computer dialog systems, or some combination thereof.")
print(sent_tokenize(text))
print(word_tokenize(text))

# nltk.TweetTokenizer()
# With the help of NLTK nltk.TweetTokenizer() method,we are able to convert the stream of words
# into small tokens so that we can analyse the audio stream with the help of nltk.TweetTokenizer()
# method.
# Example 1:In this example when we pass audio stream in the form of string it will converted to small
# tokens from a long string with the help of nltk.TweetTokenizer() method.

# Create a reference variable for Class TweetTokenizer
tk = TweetTokenizer()
gfg = "Geeks for Geeks"
geek = tk.tokenize(gfg)
print(geek)

# Example 2
tk = TweetTokenizer()
gfg = ":-) <> () {} [] :-p"
geek = tk.tokenize(gfg)
print(geek)

##
# nltk.WhitespaceTokenizer
# With the help of nltk.tokenize.WhitespaceTokenizer() method,we are able to extract the tokens from
# string of words or sentences without whitespaces,new line and tabs by using tokenize.WhitespaceTokenizer() method
#
# Example 1:
# In this example we can see that by using tokenize.WhitespaceTokenizer() method,we are able to extract the
# tokens from stream of words#

tk = WhitespaceTokenizer()
gfg = "GeeksforGeeks \nis\t for geeks"
geek = tk.tokenize(gfg)
print(geek)

##
# nltk.tokenize.TabTokenizer()
# With the help of nltk.tokenize.TabTokenizer() method,we are able to extract the tokens
# from string of words on the basis of tabs them by using tokenize.TabTokenizer() method.
#
# Example 1
# In this we can see that by using tokenize.TabTokenizer() method,we are able to extract the tokens from
# stream to words having tabs between them.#

# Create a reference variable for Class TabTokenizer
tk = TabTokenizer()
gfg = "Geeksfor\tGeeks..\t.$$&* \nis\t for geeks"
geek = tk.tokenize(gfg)
print(geek)

# Example 2
# Create a reference variable for Class TabTokenizer
tk = TabTokenizer()
gfg = "The price\t of burger \tin BurgerKing is Rs.36.\n"
geek = tk.tokenize(gfg)
print(geek)

##
# nltk.tokenize.SpaceTokenizer()
# With the help of nltk.tokenize.SpaceTokenize() method,we are able to extract the tokens
# from string of words on the basis of space between them by using tokenize.SpaceTokenizer() method.#

# Example 1:
# In this example we can see that by using tokenize.SpaceTokenizer() method,we are able to extract
# the tokens from streams to words having space between them
tk = SpaceTokenizer()
gfg = "Geeksfor Geeks.. .$$&* \nis\t for geeks"
geek = tk.tokenize(gfg)
print(geek)

# Example 2:
tk = SpaceTokenizer()
gfg = "The price\t of burger \nin BurgerKing is Rs.36.\n"
geek = tk.tokenize(gfg)
print(geek)

##
# nltk.tokenize.SExprTokenizer()
# With the help of nltk.tokenize.SExprTokenizer() method,we are able to extract the tokens
# from string of characters of numbers by using tokenize.SExprTokenizer() method.It actually
# looking for proper brackets to make tokens
##

# Example 1:
# In this example we can see that by using tokenize.SExprTokenize() method,we are able to extract the
# tokens from stream of characters or numbers by taking brackets in consideration
tk = SExprTokenizer()
gfg = "( a * ( b + c ))ab( a-c )"
geek = tk.tokenize(gfg)
print(geek)

# Example 2:
tk = SExprTokenizer()
gfg = "(a b) c d (e f)"
geek = tk.tokenize(gfg)
print(geek)

##
# nltk.tokenize.StanfordTokenizer()
# With the help of nltk.tokenize.StanfordTokenizer() method,we are able to extract the tokens from
# string of characters or numbers by using tokenize.StandfordTokenizer() method.It follows stanford
# standard for generating tokens
# #

# Example 1:
# In this example we can see that by using tokenize.SExprTokenizer() method,we are able to extract the
# tokens from stream of characters or numbers using stanford standard

# tk = StanfordTokenizer()
# gfg = "Geeks f o r Geeks"
# geek = tk.tokenize(gfg)
# print(geek)
#
# # Example 2:
# tk = StanfordTokenizer()
# gfg = "This is your great author."
# geek = tk.tokenize(gfg)
# print(geek)

##
# nltk.tokenize.mwe()
# With the help of NLTK nltk.tokenize.mwe() method,we can tokenize the audio stream into multi_word
# expression token which helps to bind the tokens with underscore by using nltk.tokenize.mwe() method.
# Remember it is case sensitive
# #

# Example 1:
# In this example we are using  MWETokenizer.tokenize() method,
# which used to bind the tokens which is defined before.We can also
# add the predefined tokens by using tokenizer.add_mwe() method.
tk = MWETokenizer([('g', 'f', 'g'), ('geeks', 'for', 'geeks')])
gfg = "geek for geeks g f g"
geek = tk.tokenize(gfg.split())
print(geek)

# nltk.tokenize.LineTokenizer
##
# With the help of nltk.tokenize.LineTokenizer() method, we are able to
# extract the tokens from string of sentences in the form
# of single line by using tokenize.LineTokenizer() method.#

# Example 1:
# In this example we can see that by using tokenize.LineTokenizer()
# method,we are able extract the tokens from stream of sentences into
# small lines.
tk = LineTokenizer()
gfg = "GeeksforGeeks...$$&* \nis\n for geeks"
geek = tk.tokenize(gfg)
print(geek)

# Example 2:
tk = LineTokenizer(blanklines='keep')
gfg = "The price\n\n of burger \nin BurgerKing is Rs.36.\n"
geek = tk.tokenize(gfg)
print(geek)

# NLTK nltk.tokenize.ConditionalFreqDist()
##
# With the help of nltk.tokenize.ConditionalFreqDist() method,we are
# able to count the frequency of words in a sentence by using
# tokenize.ConditionalFreqDist() method#

# Example 1
# In this example we can see that by using tokenize.ConditionalFreqDist()
# method,we are able to count the occurrence of words in a sentence.
tk = ConditionalFreqDist()
gfg = "Geeks for Geeks"
for word in word_tokenize(gfg):
    condition = len(word)
    tk[condition][word] += 1
print(tk)

# Example 2:
tk = ConditionalFreqDist()
gfg = " G F G"
for word in word_tokenize(gfg):
    condition = len(word)
    tk[condition][word] += 1
print(tk)

# Python NLTK | tokenize.regexp()
##
# With the help of NLTK tokenize.regexp() module,we are able to extract
# the tokens from string by using regular expression with
# RegexpTokenizer() method.#

# Example 1:
# In this example we are using RegexpTokenizer() method to extract the
# stream of tokens with the help of regular expressions.
tk = RegexpTokenizer('\+s', gaps=True)
gfg = "I love Python"
geek = tk.tokenize(gfg)
print(geek)

# Python NLTK | tokenize.WordPunctTokenizer()
##
# With the help of nltk.tokenize.WordPunctTokenizer()() method,we are
# able to extract the tokens from string of words or sentences in the form
# of Alphabetic and Non-Alphabetic character by using
# tokenize.WordPunctTokenizer()() method#

# Example 1:
# In this example we can see that by using tokenize.WordPunctTokenizer()()
# method,we are able to extract the tokens from stream of alphabetic or
# non-alphabetic character

tk = WordPunctTokenizer()
gfg = "GeeksforGeeks...$$&* \nis\t for geeks"
geek = tk.tokenize(gfg)
print(geek)

# Example 2:
# Create a reference variable for Class WordPunctTokenizer
tk = WordPunctTokenizer()

# Create a string input
gfg = "The price\t of burger \nin BurgerKing is Rs.36.\n"

# Use tokenize method
geek = tk.tokenize(gfg)

print(geek)

# Python NLTK | nltk.tokenizer.word_tokenize()
##
# With the help of nltk.tokenize.word_tokenize() method,we are able
# to extract the tokens from string of characters by using
# tokenize.word_tokenize() method.It actually returns the syllables
# from a single word.A single word can contain one or two syllables.#

# Example 1:
# In this example we can see that by using tokenize.word_tokenize()
# method,we are able to extract the syllables from stream of words or sentences

tk = SyllableTokenizer()
gfg = "Antidisestablishmentarianism"
geek = tk.tokenize(gfg)
print(geek)

# Create a reference variable for Class word_tokenize
tk = SyllableTokenizer()

# Create a string input
gfg = "Gametophyte"

# Use tokenize method
geek = tk.tokenize(gfg)

print(geek)
# NLP | Training a tokenizer and filtering stopwords in a sentence
# Why do we need to train a sentence tokenizer?
##
# In NLTK,default sentence tokenizer works for the general
# purpose and it works very well.But there are chances that it won't
# work best for some kind of text as that text may use nonstandard
# punctuation or maybe it is having a unique format.So,to handle such
# cases,training sentences tokenizer can result in much more
# accurate sentence tokenization.
##
# 1.Training Tokenizer
text = webtext.raw('C:\\Users\\Robert\\OneDrive\\Desktop\\file.txt')
sent_tokenizer = PunktSentenceTokenizer(text)
sents_1 = sent_tokenizer.tokenize(text)

print(sents_1[0])
print("\n", sents_1[1])

# 2.Default Sentence Tokenizer
sents_2 = sent_tokenize(text)

print(sents_2[0])
print("\n", sents_2[1])

##
# This difference in the second output is a good demonstration of why
# it can be very useful to train your own sentence tokenizer,
# especially when your text isn't in the typical paragraph-sentence structure.
#
# How training works?
# The PunktSentenceTokenizer class follows an unsupervised learning
# algorithm to learn what constitutes a sentence break.It is unsupervised
# because so one need not give any labelled training data,just raw text.
#
# Filtering stopwords in a tokenized sentence.
# Stopwords are common words that are present in the text but generally
# do not contribute to the meaning of a sentence.They hold almost
# no importance for the purposes of information retrieval and natural
# language processing.For example-'the' and 'a'.Most search engines
# will filter out stop words from search queries and documents.
# NLTK library comes with a stopwords corpus - nltk_data/corpora/stopwords/
# that contains word lists for many languages.#
english_stops = set(stopwords.words('english'))
words = ["Let's", 'see', 'how', "it's", 'working']
print("Before stopwords removal: ", words)
print("\nAfter stopwords removal: ",
      [word for word in words if word not in english_stops])
# Complete list of languages used in NLTK stopwords.
stopwords.fileids()