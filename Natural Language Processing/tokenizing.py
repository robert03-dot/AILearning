import nltk
from nltk import TweetTokenizer, WhitespaceTokenizer, TabTokenizer, SpaceTokenizer, SExprTokenizer
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